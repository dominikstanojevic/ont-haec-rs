//! WFA bindings for rust using Bindgen.
//!
//! First, add `bindgen` to your dependencies:
//! ```toml
//! [build-dependencies]
//! bindgen = "0.60.1"
//! ```
//!
//! Then save this file as `build.rs` in the root of your crate.
//! Update the paths below to match your WFA installation/repository.
//!
//! The example has the WFA2-lib repository cloned in `../wfa2`.
//! Make sure to run `make lib_wfa` in the WFA repository.
//! The code below will tell cargo to link against `../wfa2/lib/libwfa.a`.
//!
//! The bindings will be writted to a special `OUT_DIR` set by cargo. See
//! `example.rs` for an example of how to include and use the generated
//! bindings.

extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn wfa() {
    // 1. Link instructions for Cargo.
    // The directory of the WFA libraries, added to the search path.
    println!("cargo:rustc-link-search=WFA2-lib/build");
    // Link the `wfa-lib` library.
    println!("cargo:rustc-link-lib=static=wfa2");
    // Also link `omp`.
    println!("cargo:rustc-link-lib=gomp");
    // Invalidate the built crate whenever the linked library changes.
    println!("cargo:rerun-if-changed=WFA2-lib/build/libwfa2.a");

    // 2. Generate bindings.
    let bindings = bindgen::Builder::default()
        // Generate bindings for this header file.
        .header("wrapper.h")
        // Add this directory to the include path to find included header files.
        .clang_args(&["-IWFA2-lib/"])
        // Generate bindings for all functions starting with `wavefront_`.
        .allowlist_function("wavefront_.*")
        // Generate bindings for all variables starting with `wavefront_`.
        .allowlist_var("wavefront_.*")
        // Invalidate the built crate whenever any of the included header files
        // changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings_wfa.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings_wfa.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    wfa();
}
