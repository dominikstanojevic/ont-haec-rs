use std::path::Path;

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;

use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix2};

use tch::{CModule, IValue, IndexOp, Tensor};

use crate::{
    consensus::{ConsensusData, ConsensusWindow},
    features::SupportedPos,
};

const BASE_PADDING: u8 = 11;
const QUAL_MIN_VAL: f32 = 33.;
const QUAL_MAX_VAL: f32 = 126.;

const QUAL_RANGE_DIFF: f64 = (QUAL_MAX_VAL - QUAL_MIN_VAL) as f64;
const QUAL_SCALE: f64 = 2. / QUAL_RANGE_DIFF;
const QUAL_OFFSET: f64 = 2. * QUAL_MIN_VAL as f64 / QUAL_RANGE_DIFF + 1.;

pub(crate) const BASES_MAP: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 9, 255, 255,
    255, 255, 255, 255, 4, 255, 255, 255, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 5, 255, 6, 255, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

pub(crate) struct InferenceBatch {
    wids: Vec<u32>,
    bases: Tensor,
    quals: Tensor,
    lens: Tensor,
    indices: Vec<Tensor>,
}

impl InferenceBatch {
    fn new(
        wids: Vec<u32>,
        bases: Tensor,
        quals: Tensor,
        lens: Tensor,
        indices: Vec<Tensor>,
    ) -> Self {
        Self {
            wids,
            bases,
            quals,
            lens,
            indices,
        }
    }
}

pub(crate) struct InferenceData {
    consensus_data: ConsensusData,
    batches: Vec<InferenceBatch>,
}

impl InferenceData {
    fn new(consensus_data: ConsensusData, batches: Vec<InferenceBatch>) -> Self {
        Self {
            consensus_data,
            batches,
        }
    }
}

fn collate<'a>(batch: &[(u32, &ConsensusWindow)]) -> InferenceBatch {
    const KERNEL_SIZE: usize = 17;
    const KERNEL_RADIUS: usize = KERNEL_SIZE / 2;

    let total_candidates: usize = batch.iter().map(|(_, f)| f.supported.len()).sum();

    let size = [
        total_candidates as i64,
        KERNEL_SIZE as i64,
        batch[0].1.bases.len_of(Axis(1)) as i64,
    ]; // [N, K, R]

    let bases = Tensor::full(
        &size,
        BASE_PADDING as i64,
        (tch::Kind::Uint8, tch::Device::Cpu),
    );

    let quals = Tensor::full(
        &size,
        QUAL_MAX_VAL as i64,
        (tch::Kind::Uint8, tch::Device::Cpu),
    );
    //let quals = Tensor::zeros_like(&bases);

    let mut lens = Vec::with_capacity(batch.len());
    let mut indices = Vec::with_capacity(batch.len());
    let mut wids = Vec::with_capacity(batch.len());

    let mut current_candidate = 0;

    for (_idx, (wid, f)) in batch.iter().enumerate() {
        wids.push(*wid);
        let l = f.bases.len_of(Axis(0));

        let supported: Vec<_> = f
            .supported
            .iter()
            .map(|&sp| (f.indices[sp.pos as usize] + sp.ins as usize))
            .collect();

        let mut candidate_indices = Vec::with_capacity(supported.len());

        for &pos in &supported {
            let start = pos.saturating_sub(KERNEL_RADIUS);
            let end = (pos + KERNEL_RADIUS + 1).min(l);
            let pad_left = KERNEL_RADIUS.saturating_sub(pos);
            let pad_right = KERNEL_RADIUS.saturating_sub(l - pos);

            let mut bases_window =
                Array2::from_elem((KERNEL_SIZE, f.bases.len_of(Axis(1))), BASE_PADDING);

            let mut quals_window =
                Array2::from_elem((KERNEL_SIZE, f.quals.len_of(Axis(1))), QUAL_MAX_VAL as u8);
            quals_window.slice_mut(s![..KERNEL_RADIUS, ..]).fill(0u8);

            let data_start = KERNEL_RADIUS.saturating_sub(pos);
            let data_len = end - start;
            bases_window
                .slice_mut(s![data_start..data_start + data_len, ..])
                .assign(&f.bases.slice(s![start..end, ..]));
            quals_window
                .slice_mut(s![data_start..data_start + data_len, ..])
                .assign(&f.quals.slice(s![start..end, ..]));

            let bt = unsafe {
                let shape: Vec<_> = vec![KERNEL_SIZE as i64, f.bases.len_of(Axis(1)) as i64];
                Tensor::from_blob(
                    bases_window.as_ptr() as *const u8,
                    &shape,
                    &[shape[shape.len() - 1], 1],
                    tch::Kind::Uint8,
                    tch::Device::Cpu,
                )
            };

            let qt = unsafe {
                let shape: Vec<_> = vec![KERNEL_SIZE as i64, f.quals.len_of(Axis(1)) as i64];
                Tensor::from_blob(
                    quals_window.as_ptr() as *const u8,
                    &shape,
                    &[shape[shape.len() - 1], 1],
                    tch::Kind::Uint8,
                    tch::Device::Cpu,
                )
            };

            bases
                .i((current_candidate as i64, .., ..))
                .copy_(&bt.squeeze());
            quals
                .i((current_candidate as i64, .., ..))
                .copy_(&qt.squeeze());

            candidate_indices.push(current_candidate);
            current_candidate += 1;
        }

        lens.push(f.supported.len() as i32);

        indices.push(Tensor::try_from(candidate_indices).unwrap());
    }

    for (_, example) in batch {
        if example.rid == 14987 && example.wid == 0 {
            bases.save("bases.pt").unwrap();
            quals.save("quals.pt").unwrap();
            Tensor::try_from(&lens).unwrap().save("lens.pt").unwrap();

            println!("Saved tensors.");
        }
    }

    InferenceBatch::new(wids, bases, quals, Tensor::try_from(lens).unwrap(), indices)
}

fn inference(
    batch: InferenceBatch,
    model: &CModule,
    device: tch::Device,
) -> (Vec<u32>, Vec<Tensor>, Vec<Tensor>) {
    let quals_mask = batch.quals.eq(0).to(device);
    let quals = batch.quals.to_device_(device, tch::Kind::Float, true, true);
    let quals = quals.where_self(&quals_mask, &(QUAL_SCALE * &quals - QUAL_OFFSET));

    //let quals = QUAL_SCALE * quals - QUAL_OFFSET;

    let inputs = [
        IValue::Tensor(batch.bases.to_device_(device, tch::Kind::Int, true, true)),
        IValue::Tensor(quals),
        IValue::Tensor(batch.lens.to_device_(device, tch::Kind::Int, true, true)),
    ];

    let (info_logits, bases_logits) =
        <(Tensor, Tensor)>::try_from(model.forward_is(&inputs).unwrap()).unwrap();

    // Get number of target positions for each window
    let lens: Vec<i64> = match inputs[2] {
        IValue::Tensor(ref t) => Vec::try_from(t).unwrap(),
        _ => unreachable!(),
    };

    let info_logits = info_logits.to(tch::Device::Cpu).split_with_sizes(&lens, 0);
    let bases_logits = bases_logits
        .argmax(1, false)
        .to(tch::Device::Cpu)
        .split_with_sizes(&lens, 0);

    (batch.wids, info_logits, bases_logits)
}

pub(crate) fn inference_worker<P: AsRef<Path>>(
    model_path: P,
    device: tch::Device,
    input_channel: Receiver<InferenceData>,
    output_channel: Sender<ConsensusData>,
) {
    let _no_grad = tch::no_grad_guard();

    let mut model = tch::CModule::load_on_device(model_path, device).expect("Cannot load model.");
    model.set_eval();

    loop {
        let mut data = match input_channel.recv() {
            Ok(data) => data,
            Err(_) => break,
        };

        for batch in data.batches {
            let (wids, info_logits, bases_logits) = inference(batch, &model, device);
            wids.into_iter()
                .zip(info_logits.into_iter())
                .zip(bases_logits.into_iter())
                .for_each(|((wid, il), bl)| {
                    data.consensus_data[wid as usize]
                        .info_logits
                        .replace(Vec::try_from(il).unwrap());

                    data.consensus_data[wid as usize]
                        .bases_logits
                        .replace(Vec::try_from(bl).unwrap());
                });
        }

        output_channel.send(data.consensus_data).unwrap();
    }
}

pub(crate) fn prepare_examples(
    features: impl IntoIterator<Item = WindowExample>,
    batch_size: usize,
) -> InferenceData {
    let windows: Vec<_> = features
        .into_iter()
        .map(|mut example| {
            // Transform bases (encode) and quals (normalize)
            example.bases.mapv_inplace(|b| BASES_MAP[b as usize]);

            // Transpose: [R, L] -> [L, R]
            //bases.swap_axes(1, 0);
            //quals.swap_axes(1, 0);

            let tidx = get_target_indices(&example.bases);

            //TODO: Start here.
            ConsensusWindow::new(
                example.rid,
                example.wid,
                example.n_alns,
                example.n_total_wins,
                example.bases,
                example.quals,
                tidx,
                example.supported,
                None,
                None,
            )
        })
        .collect();

    let batches: Vec<_> = (0u32..)
        .zip(windows.iter())
        .filter(|(_, features)| features.supported.len() > 0)
        .chunks(batch_size)
        .into_iter()
        .map(|v| {
            let batch = v.collect::<Vec<_>>();
            collate(&batch)
        })
        .collect();

    InferenceData::new(windows, batches)
}

fn get_target_indices<S: Data<Elem = u8>>(bases: &ArrayBase<S, Ix2>) -> Vec<usize> {
    bases
        .slice(s![.., 0])
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| {
            if *b != BASES_MAP[b'*' as usize] {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

pub(crate) struct WindowExample {
    rid: u32,
    wid: u16,
    n_alns: u8,
    bases: Array2<u8>,
    quals: Array2<u8>,
    supported: Vec<SupportedPos>,
    n_total_wins: u16,
}

impl WindowExample {
    pub(crate) fn new(
        rid: u32,
        wid: u16,
        n_alns: u8,
        bases: Array2<u8>,
        quals: Array2<u8>,
        supported: Vec<SupportedPos>,
        n_total_wins: u16,
    ) -> Self {
        Self {
            rid,
            wid,
            n_alns,
            bases,
            quals,
            supported,
            n_total_wins,
        }
    }
}

/*#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array3};

    use super::{inference, prepare_examples};

    #[test]
    fn test() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/mm2-attn.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        /*let mut files: Vec<_> = resources
            .join("resources/example_feats")
            .read_dir()
            .unwrap()
            .filter_map(|p| {
                let p = p.unwrap().path();
                p.
                match p.extension() {
                    Some(ext) if ext == "npy" => Some(p),
                    _ => None,
                }
            })
            .collect();
        files.sort();*/

        // Create input data
        let features: Vec<_> = (0..4)
            .into_iter()
            .map(|wid| {
                let feats: Array3<u8> =
                    read_npy(format!("resources/example_feats/{}.features.npy", wid)).unwrap();
                let supported: Array1<u16> =
                    read_npy(format!("resources/example_feats/{}.supported.npy", wid)).unwrap();
                (feats, supported.iter().map(|s| *s as usize).collect())
            })
            .collect();
        let mut input_data = prepare_examples(0, features);
        let batch = input_data.batches.remove(0);

        let output = inference(batch, &model, device);
        let predicted: Array1<f32> = output
            .1
            .into_iter()
            .flat_map(|l| Vec::try_from(l).unwrap().into_iter())
            .collect();

        let target: Array1<f32> =
            read_npy(resources.join("resources/example_feats_tch_out.npy")).unwrap();

        assert_relative_eq!(predicted, target, epsilon = 1e-5);
    }

    #[test]
    fn test2() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/model.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        let mut files: Vec<_> = PathBuf::from("resources/test_rs")
            .read_dir()
            .unwrap()
            .filter_map(|p| {
                let p = p.unwrap().path();
                match p.extension() {
                    Some(ext) if ext == "npy" => Some(p),
                    _ => None,
                }
            })
            .collect();
        files.sort();

        // Create input data
        let mut features: Vec<_> = files
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let feats: Array3<u8> = read_npy(p).unwrap();
                (i as u16, feats)
            })
            .collect();
        let input_data = prepare_examples(0, &mut features);

        let output = inference(input_data, &model, device);
        let predicted: Array1<f32> = output
            .windows
            .into_iter()
            .flat_map(|(_, _, l)| l.into_iter())
            .collect();

        println!("{:?}", &predicted.to_vec()[4056 - 5..4056 + 5]);
    }
}*/
