//! LFM2.5-350M exec implementation for CLI `run` subcommand

use std::path::Path;
use std::time::Instant;

use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};

use crate::exec::ExecModel;
use crate::models::{GenerateModel, LoadSpec, WhichModel, lfm2_5::generate::Lfm2_5GenerateModel};
use crate::utils::get_file_path;

pub struct Lfm2_5Exec;

impl Lfm2_5Exec {
    fn build_text_request(input: &[String]) -> Result<ChatCompletionParameters> {
        let input_text = input.first().ok_or_else(|| {
            anyhow!("lfm2.5 run requires one text input unless --request-json is provided")
        })?;
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        Ok(serde_json::from_value(serde_json::json!({
            "model": "lfm2.5-350m",
            "messages": [
                {
                    "role": "user",
                    "content": target_text,
                }
            ]
        }))?)
    }

    fn load_request_json(path: &str) -> Result<ChatCompletionParameters> {
        let request_path = if path.starts_with("file://") {
            get_file_path(path)?
        } else {
            Path::new(path).to_path_buf()
        };
        let request_json = std::fs::read_to_string(&request_path).map_err(|err| {
            anyhow!(
                "failed to read request json file {}: {err}",
                request_path.display()
            )
        })?;
        serde_json::from_str(&request_json).map_err(|err| {
            anyhow!(
                "failed to parse request json file {}: {err}",
                request_path.display()
            )
        })
    }

    fn build_request(
        input: &[String],
        request_json: Option<&str>,
    ) -> Result<ChatCompletionParameters> {
        if let Some(path) = request_json {
            Self::load_request_json(path)
        } else {
            Self::build_text_request(input)
        }
    }

    pub fn run_with_spec(
        input: &[String],
        output: Option<&str>,
        spec: &LoadSpec,
        request_json: Option<&str>,
    ) -> Result<()> {
        let i_start = Instant::now();
        let mut model = Lfm2_5GenerateModel::init_from_spec(spec, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let mes = Self::build_request(input, request_json)?;

        let i_start = Instant::now();
        let result = model.generate(mes)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        let result_json = serde_json::to_string_pretty(&result)?;
        println!("Result: {}", result_json);

        if let Some(out) = output {
            std::fs::write(out, &result_json)?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}

impl ExecModel for Lfm2_5Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(WhichModel::LFM2_5_350M, weight_path);
        Self::run_with_spec(input, output, &spec, None)
    }
}
