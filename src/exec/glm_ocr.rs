//! Glm-OCR exec implementation for CLI `run` subcommand
use std::{path::Path, time::Instant};

use anyhow::{Ok, Result, anyhow};
use serde_json::json;

use crate::exec::ExecModel;
use crate::models::{GenerateModel, LoadSpec, glm_ocr::generate::GlmOcrGenerateModel};

pub struct GlmOcrExec;

fn resolve_input_url(input: &str) -> Result<String> {
    if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("file://")
    {
        return Ok(input.to_string());
    }

    let path = Path::new(input);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let path = path
        .canonicalize()
        .map_err(|e| anyhow!("failed to resolve input path {}: {e}", path.display()))?;
    url::Url::from_file_path(&path)
        .map(|url| url.to_string())
        .map_err(|_| {
            anyhow!(
                "failed to convert input path to file url: {}",
                path.display()
            )
        })
}

fn build_request(
    input_url: &str,
    max_tokens: Option<u32>,
) -> Result<aha_openai_dive::v1::resources::chat::ChatCompletionParameters> {
    Ok(serde_json::from_value(json!({
        "model": "glm-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": input_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Text Recognition:"
                    }
                ]
            }
        ],
        "max_tokens": max_tokens.unwrap_or(256)
    }))?)
}

impl GlmOcrExec {
    pub fn run_with_spec(
        input: &[String],
        output: Option<&str>,
        spec: &LoadSpec,
        max_tokens: Option<u32>,
    ) -> Result<()> {
        let url = input
            .first()
            .ok_or_else(|| anyhow!("glm-ocr run requires an input image path or url"))?;
        let input_url = resolve_input_url(url)?;

        let i_start = Instant::now();
        let mut model = GlmOcrGenerateModel::init_from_spec(spec, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let mes = build_request(&input_url, max_tokens)?;

        let i_start = Instant::now();
        let result = model.generate(mes)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        println!("Result: {:?}", result);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", result))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}

impl ExecModel for GlmOcrExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(crate::models::WhichModel::GlmOCR, weight_path);
        Self::run_with_spec(input, output, &spec, None)
    }
}

#[cfg(test)]
mod tests {
    use super::{build_request, resolve_input_url};
    use aha_openai_dive::v1::resources::chat::{
        ChatMessage, ChatMessageContent, ChatMessageContentPart,
    };
    use anyhow::Result;

    #[test]
    fn resolve_input_url_converts_relative_windows_path_to_file_url() -> Result<()> {
        let url = resolve_input_url(r".\assets\img\ocr_test1.png")?;
        assert!(url.starts_with("file:///"));
        assert!(url.contains("assets/img/ocr_test1.png"));
        Ok(())
    }

    #[test]
    fn build_request_accepts_file_url_without_json_escape_issues() -> Result<()> {
        let mes = build_request("file:///D:/model_download/ocr_test1.png", Some(32))?;
        let ChatMessage::User { content, .. } = &mes.messages[0] else {
            panic!("expected user message");
        };
        let ChatMessageContent::ContentPart(parts) = content else {
            panic!("expected multipart content");
        };
        let ChatMessageContentPart::Image(image) = &parts[0] else {
            panic!("expected image content part");
        };
        assert_eq!(
            image.image_url.url,
            "file:///D:/model_download/ocr_test1.png"
        );
        assert_eq!(mes.max_tokens, Some(32));
        Ok(())
    }
}
