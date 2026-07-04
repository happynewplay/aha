//! MiniCPM5-1B exec implementation for CLI `run` subcommand

use std::time::Instant;
use std::path::Path;

use anyhow::{Result, anyhow};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;

use crate::exec::ExecModel;
use crate::models::{
    GenerateModel, LoadSpec, WhichModel, minicpm5::generate::MiniCPM5GenerateModel,
};
use crate::utils::get_file_path;

pub struct MiniCPM5Exec;

impl MiniCPM5Exec {
    fn build_text_request(input: &[String]) -> Result<ChatCompletionParameters> {
        let input_text = input
            .first()
            .ok_or_else(|| anyhow!("minicpm5 run requires one text input unless --request-json is provided"))?;
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        Ok(serde_json::from_value(serde_json::json!({
            "model": "minicpm5-1b",
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
        let mut model = MiniCPM5GenerateModel::init_from_spec(spec, None, None)?;
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

impl ExecModel for MiniCPM5Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(WhichModel::MiniCPM5_1B, weight_path);
        Self::run_with_spec(input, output, &spec, None)
    }
}

#[cfg(test)]
mod tests {
    use super::MiniCPM5Exec;
    use anyhow::Result;
    use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;

    fn build_request_json_payload() -> serde_json::Value {
        serde_json::json!({
            "model": "minicpm5-1b",
            "messages": [
                {
                    "role": "user",
                    "content": "please call a tool"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup docs",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ],
            "tool_choice": null
        })
    }

    #[test]
    fn build_request_prefers_request_json_and_preserves_tools() -> Result<()> {
        let request_path = std::env::temp_dir().join(format!(
            "aha_minicpm5_request_{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&request_path, serde_json::to_string(&build_request_json_payload())?)?;

        let request = MiniCPM5Exec::build_request(&[], request_path.to_str())?;
        let value = serde_json::to_value(request)?;
        assert_eq!(value["model"], "minicpm5-1b");
        assert!(value["tools"].is_array());
        assert_eq!(
            value["tools"].as_array().map(|tools| tools.len()),
            Some(1)
        );

        std::fs::remove_file(&request_path)?;
        Ok(())
    }

    #[test]
    fn build_text_request_keeps_existing_plain_text_flow() -> Result<()> {
        let request = MiniCPM5Exec::build_request(&[String::from("hello")], None)?;
        let value = serde_json::to_value(request)?;
        assert_eq!(value["model"], "minicpm5-1b");
        assert_eq!(value["messages"].as_array().map(|v| v.len()), Some(1));
        assert!(value.get("tools").is_none() || value["tools"].is_null());
        Ok(())
    }

    #[test]
    fn load_request_json_reports_path_failures() {
        let err = MiniCPM5Exec::build_request(&[], Some("D:/definitely/not/found.json"))
            .expect_err("expected missing file to error");
        assert!(err.to_string().contains("failed to read request json file"));
    }

    #[test]
    fn load_request_json_parses_minimal_chat_completion_parameters() -> Result<()> {
        let payload = build_request_json_payload();
        let request: ChatCompletionParameters = serde_json::from_value(payload)?;
        let value = serde_json::to_value(request)?;
        assert!(value["tools"].is_array());
        assert!(value["tool_choice"].is_null());
        Ok(())
    }
}
