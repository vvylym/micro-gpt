//! Binary entrypoint: runs the training and inference pipeline.

use microgpt_original::{run, DEFAULT_INPUT_PATH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        std::env::var("MICROGPT_INPUT_PATH").unwrap_or_else(|_| DEFAULT_INPUT_PATH.to_string());
    run(&path)
}
