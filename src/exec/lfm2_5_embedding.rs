use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::exec::ExecModel;
use crate::models::{
    EmbeddingOptions, LoadSpec, WhichModel, lfm2_5_embedding::generate::Lfm2_5EmbeddingModel,
};
use crate::utils::get_file_path;

pub struct Lfm2_5EmbeddingExec;

impl Lfm2_5EmbeddingExec {
    fn build_text_input(input: &[String]) -> Result<String> {
        let input_text = input
            .first()
            .ok_or_else(|| anyhow!("embedding run requires one text input"))?;
        if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            Ok(std::fs::read_to_string(path)?)
        } else {
            Ok(input_text.clone())
        }
    }

    pub fn run_with_spec(
        input: &[String],
        output: Option<&str>,
        spec: &LoadSpec,
        options: EmbeddingOptions,
    ) -> Result<()> {
        let i_start = Instant::now();
        let mut model = Lfm2_5EmbeddingModel::init_from_spec(spec, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let text = Self::build_text_input(input)?;

        let i_start = Instant::now();
        let result = model.embed_with_options(&[text], options)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        let result_json = serde_json::to_string_pretty(&result)?;
        println!("{}", result_json);

        if let Some(out) = output {
            std::fs::write(out, &result_json)?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}

impl ExecModel for Lfm2_5EmbeddingExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(WhichModel::LFM2_5Embedding350M, weight_path);
        Self::run_with_spec(input, output, &spec, EmbeddingOptions::default())
    }
}
