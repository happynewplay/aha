use std::path::Path;

use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    lfm2_5::generate::Lfm2_5GenerateModel,
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

const DEFAULT_LFM2_5_SAFETENSORS_DIR: &str = r"D:\model_download\LFM2.5-350M";
const DEFAULT_LFM2_5_230M_SAFETENSORS_DIR: &str = r"D:\model_download\LFM2.5-230M";

fn resolve_lfm2_5_dir() -> Option<String> {
    let path = std::env::var("AHA_LFM2_5_350M_DIR")
        .unwrap_or_else(|_| DEFAULT_LFM2_5_SAFETENSORS_DIR.to_string());
    if Path::new(&path).is_dir() {
        Some(path)
    } else {
        None
    }
}

fn resolve_lfm2_5_230m_dir() -> Option<String> {
    let path = std::env::var("AHA_LFM2_5_230M_DIR")
        .unwrap_or_else(|_| DEFAULT_LFM2_5_230M_SAFETENSORS_DIR.to_string());
    if Path::new(&path).is_dir() {
        Some(path)
    } else {
        None
    }
}

fn build_text_request() -> Result<ChatCompletionParameters> {
    Ok(serde_json::from_value(serde_json::json!({
        "model": "lfm2.5-350m",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话介绍 Rust。"
            }
        ]
    }))?)
}

#[test]
fn lfm2_5_safetensors_init_from_spec_can_generate() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_dir() else {
        println!("skip lfm2.5 test: dir not found, set AHA_LFM2_5_350M_DIR to run");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::LFM2_5_350M,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = Lfm2_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[test]
fn lfm2_5_230m_safetensors_response_uses_requested_model_id() -> Result<()> {
    let Some(weight_dir) = resolve_lfm2_5_230m_dir() else {
        println!("skip lfm2.5-230m test: dir not found, set AHA_LFM2_5_230M_DIR to run");
        return Ok(());
    };
    let spec = LoadSpec {
        model: WhichModel::LFM2_5_230M,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };
    let request: ChatCompletionParameters = serde_json::from_value(serde_json::json!({
        "model": "lfm2.5-230m",
        "max_tokens": 1,
        "messages": [{ "role": "user", "content": "请用一句话介绍 Rust。" }]
    }))?;

    let mut model = Lfm2_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = serde_json::to_value(model.generate(request)?)?;

    assert_eq!(response["model"], "lfm2.5-230m");
    Ok(())
}
