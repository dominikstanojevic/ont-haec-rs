use ndarray::Array2;

use std::{cmp::Reverse, collections::BinaryHeap};

use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;

use ndarray::{s, Axis};
use rustc_hash::FxHashMap as HashMap;

use crate::features::SupportedPos;

use crate::inference::BASES_MAP;

const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];

// Bases, tidx, supported, logits
pub(crate) struct ConsensusWindow {
    pub(crate) rid: u32,
    pub(crate) wid: u16,
    pub(crate) n_alns: u8,
    pub(crate) n_total_wins: u16,
    pub(crate) bases: Array2<u8>,
    pub(crate) quals: Array2<u8>,
    pub(crate) indices: Vec<usize>,
    pub(crate) supported: Vec<SupportedPos>,
    pub(crate) info_logits: Option<Vec<f32>>,
    pub(crate) bases_logits: Option<Vec<u8>>,
}

impl ConsensusWindow {
    pub(crate) fn new(
        rid: u32,
        wid: u16,
        n_alns: u8,
        n_total_wins: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        indices: Vec<usize>,
        supported: Vec<SupportedPos>,
        info_logits: Option<Vec<f32>>,
        bases_logits: Option<Vec<u8>>,
    ) -> Self {
        Self {
            rid,
            wid,
            n_alns,
            n_total_wins,
            bases,
            quals,
            indices,
            supported,
            info_logits,
            bases_logits,
        }
    }
}

pub type ConsensusData = Vec<ConsensusWindow>;

#[allow(dead_code)]
fn two_most_frequent<'a, I>(elements: I) -> Vec<(usize, u8)>
where
    I: Iterator<Item = u8>,
{
    let mut map = HashMap::default();
    for x in elements {
        *map.entry(x).or_default() += 1;
    }

    let mut heap = BinaryHeap::with_capacity(3);
    for (x, count) in map.into_iter() {
        heap.push(Reverse((count, x)));
        if heap.len() > 2 {
            heap.pop();
        }
    }

    heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
}

fn consensus(data: ConsensusData, counts: &mut [u8]) -> Option<Vec<Vec<u8>>> {
    let mut corrected_seqs = Vec::new();
    let mut corrected: Vec<u8> = Vec::new();

    let minmax = data
        .iter()
        .enumerate()
        .filter_map(|(idx, win)| if win.n_alns > 1 { Some(idx) } else { None })
        .minmax();
    let (wid_st, wid_en) = match minmax {
        NoElements => {
            return None;
        }
        OneElement(wid) => (wid, wid + 1),
        MinMax(st, en) => (st, en + 1),
    };

    for window in data[wid_st..wid_en].iter() {
        /*if window.n_alns < 2 {
            let start = wid * window_size;
            let end = ((wid + 1) * window_size).min(uncorrected.len());

            corrected.extend(&uncorrected[start..end]);
            continue;
        }*/
        if window.n_alns < 2 {
            if corrected.len() > 0 {
                corrected_seqs.push(corrected);
                corrected = Vec::new();
            }

            continue;
        }

        // Don't analyze empty rows: LxR -> LxN
        //let n_rows = (window.n_alns + 1).min(TOP_K + 1);
        let n_rows = window.n_alns + 1;
        let bases = window.bases.slice(s![.., ..n_rows as usize]);
        let maybe_info = match window.supported.len() {
            0 => HashMap::default(),
            _ => window
                .supported
                .iter()
                .zip(window.info_logits.as_ref().unwrap().iter())
                .zip(window.bases_logits.as_ref().unwrap().iter())
                .map(|((supp, il), bl)| (*supp, (*il, *bl)))
                .collect(),
        };

        let (mut pos, mut ins) = (-1i32, 0);
        for col in bases.axis_iter(Axis(0)) {
            if col[0] == BASES_MAP[b'*' as usize] {
                ins += 1;
            } else {
                pos += 1;
                ins = 0;
            }

            if let Some((_, b)) = maybe_info.get(&SupportedPos::new(pos as u16, ins)) {
                let base = match *b {
                    0 => b'A',
                    1 => b'C',
                    2 => b'G',
                    3 => b'T',
                    4 => b'*',
                    _ => panic!("Unrecognized base"),
                };

                /*if *il > 0.0 {
                    println!(
                        "{}\t{}\t{}",
                        std::str::from_utf8(&read.id).unwrap(),
                        corrected_seqs.len(),
                        corrected.len(),
                    );
                }*/

                /*println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    'S',
                    base
                );*/
                if base != b'*' {
                    corrected.push(base);
                }
            } else {
                // Count bases
                counts.iter_mut().for_each(|c| *c = 0);
                col.iter().for_each(|&b| {
                    if b != BASES_MAP[b'.' as usize] {
                        counts[BASES_UPPER_COUNTER[b as usize]] += 1;
                    }
                });

                // Get two most common bases and counts - (c, b)
                let (mc0, mc1) = counts
                    .iter()
                    .enumerate()
                    .sorted_by_key(|(_, c)| Reverse(*c))
                    .take(2)
                    .map(|(i, c)| (*c, BASES_UPPER[i]))
                    .collect_tuple()
                    .unwrap();
                let tbase = BASES_UPPER[col[0] as usize];

                let base = if mc0.0 < 2 || (mc0.0 == mc1.0 && (mc0.1 == tbase || mc1.1 == tbase)) {
                    tbase
                } else {
                    mc0.1
                };

                /*println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    std::str::from_utf8(&read.id).unwrap(),
                    wid,
                    pos,
                    ins,
                    corrected_seqs.len(),
                    corrected.len(),
                    "N",
                    base,
                );*/
                if base != b'*' {
                    corrected.push(base);
                }
            }
        }
    }

    if corrected.len() > 0 {
        corrected_seqs.push(corrected);
    }

    Some(corrected_seqs)
}

pub(crate) fn consensus_worker(
    receiver: Receiver<ConsensusData>,
    sender: Sender<(usize, Vec<Vec<u8>>)>,
) {
    let mut consensus_data = HashMap::default();
    let mut counts = [0u8; 5];
    loop {
        let output = match receiver.recv() {
            Ok(output) => output,
            Err(_) => break,
        };

        for cw in output {
            let rid = cw.rid;
            let n_total_wins = cw.n_total_wins;

            let entry = consensus_data.entry(cw.rid).or_insert_with(|| Vec::new());
            entry.push(cw);

            if entry.len() == (n_total_wins as usize) {
                let mut windows = consensus_data.remove(&rid).unwrap();
                windows.sort_by_key(|cw| cw.wid);

                let seq = consensus(windows, &mut counts);

                if let Some(s) = seq {
                    sender.send((rid as usize, s)).unwrap();
                }
            }
        }
    }

    assert!(consensus_data.is_empty());
}
