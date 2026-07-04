use std::path::{Path, PathBuf};

use aha::chat_template::ChatTemplate;
use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    common::gguf::load_text_bootstrap_from_gguf,
    minicpm5::generate::MiniCPM5GenerateModel,
};
use aha::tokenizer::TokenizerModel;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

const DEFAULT_MINICPM5_SAFETENSORS_DIR: &str = r"D:\model_download\MiniCPM5-1B";
const DEFAULT_MINICPM5_GGUF_DIRS: &[&str] = &[
    r"D:\model_download\MiniCPM5-1B-GGUF",
    r"D:\model_download\OpenBMB\MiniCPM5-1B-GGUF",
];

fn existing_dir(path: &str) -> bool {
    Path::new(path).is_dir()
}

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn first_file_with_extension_recursive(dir: &str, extension: &str) -> Result<Option<PathBuf>> {
    if !existing_dir(dir) {
        return Ok(None);
    }

    let mut stack = vec![PathBuf::from(dir)];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
            {
                matches.push(path);
            }
        }
    }

    matches.sort();
    Ok(matches.into_iter().next())
}

fn resolve_safetensors_dir() -> Option<String> {
    let path = env_or_default("AHA_MINICPM5_SAFETENSORS_DIR", DEFAULT_MINICPM5_SAFETENSORS_DIR);
    if existing_dir(&path) {
        return Some(path);
    }
    None
}

fn resolve_gguf_path() -> Result<Option<String>> {
    if let Ok(path) = std::env::var("AHA_MINICPM5_GGUF_PATH") {
        let p = Path::new(&path);
        if p.is_file() {
            return Ok(Some(path));
        }
        if p.is_dir()
            && let Some(found) = first_file_with_extension_recursive(&path, "gguf")?
        {
            return Ok(Some(found.to_string_lossy().to_string()));
        }
    }

    for dir in DEFAULT_MINICPM5_GGUF_DIRS {
        if let Some(found) = first_file_with_extension_recursive(dir, "gguf")? {
            return Ok(Some(found.to_string_lossy().to_string()));
        }
    }

    Ok(None)
}

fn build_text_request() -> Result<ChatCompletionParameters> {
    Ok(serde_json::from_value(serde_json::json!({
        "model": "minicpm5-1b",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话介绍 Rust。"
            }
        ]
    }))?)
}

fn response_content_text(response: &serde_json::Value) -> String {
    response
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string()
}

#[test]
fn minicpm5_safetensors_init_from_spec_can_generate() -> Result<()> {
    let Some(weight_dir) = resolve_safetensors_dir() else {
        println!(
            "skip safetensors test: dir not found, set AHA_MINICPM5_SAFETENSORS_DIR to run"
        );
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::MiniCPM5_1B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = MiniCPM5GenerateModel::init_from_spec(&spec, None, None)?;
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
fn minicpm5_gguf_init_from_spec_can_generate() -> Result<()> {
    let Some(gguf_path) = resolve_gguf_path()? else {
        println!("skip gguf test: no gguf file found, set AHA_MINICPM5_GGUF_PATH to run explicitly");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::MiniCPM5_1B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            ..Default::default()
        },
    };

    let mut model = MiniCPM5GenerateModel::init_from_spec(&spec, None, None)?;
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
fn minicpm5_gguf_generation_is_not_whitespace_only() -> Result<()> {
    let Some(gguf_path) = resolve_gguf_path()? else {
        println!("skip gguf whitespace test: no gguf file found");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::MiniCPM5_1B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            ..Default::default()
        },
    };

    let mut model = MiniCPM5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let content = response_content_text(&value);
    assert!(
        !content.trim().is_empty(),
        "expected gguf response to contain visible text, got {content:?}"
    );
    Ok(())
}

#[test]
fn minicpm5_gguf_bootstrap_matches_safetensors_prompt_and_tokens() -> Result<()> {
    let Some(weight_dir) = resolve_safetensors_dir() else {
        println!("skip prompt/token test: safetensors dir not found");
        return Ok(());
    };
    let Some(gguf_path) = resolve_gguf_path()? else {
        println!("skip prompt/token test: gguf file not found");
        return Ok(());
    };

    let request = build_text_request()?;
    let safetensors_template = ChatTemplate::init(&weight_dir)?;
    let safetensors_prompt = safetensors_template.apply_chat_template(&request)?;
    let safetensors_tokenizer = TokenizerModel::init(&weight_dir)?;
    let safetensors_ids = safetensors_tokenizer.text_encode_vec(safetensors_prompt.clone(), true)?;

    let gguf_bootstrap =
        load_text_bootstrap_from_gguf(&gguf_path, Some(false), Some(false), Some(false))?;
    let gguf_template = ChatTemplate::str_init(
        gguf_bootstrap
            .chat_template
            .as_deref()
            .unwrap_or_default(),
    )?;
    let gguf_prompt = gguf_template.apply_chat_template(&request)?;
    let gguf_ids = gguf_bootstrap
        .tokenizer
        .text_encode_vec(gguf_prompt.clone(), true)?;

    assert_eq!(gguf_prompt, safetensors_prompt, "prompt render mismatch");
    assert_eq!(gguf_ids, safetensors_ids, "token ids mismatch");
    Ok(())
}

#[test]
fn minicpm5_chat_template_renders_tool_calls() -> Result<()> {
    let Some(weight_dir) = resolve_safetensors_dir() else {
        println!("skip tool-call template test: safetensors dir not found");
        return Ok(());
    };

    let request: ChatCompletionParameters = serde_json::from_value(serde_json::json!({
        "model": "minicpm5-1b",
        "messages": [
            {
                "role": "user",
                "content": "请调用工具查找 Rust 文档"
            },
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"query\":\"rust\"}"
                        }
                    }
                ]
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
        ]
    }))?;

    let template = ChatTemplate::init(&weight_dir)?;
    let prompt = template.apply_chat_template(&request)?;
    assert!(prompt.contains("<function name=\"lookup\">"));
    assert!(prompt.contains("<|im_start|>assistant"));
    Ok(())
}
