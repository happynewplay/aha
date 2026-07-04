use std::path::{Path, PathBuf};

use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    glm_ocr::generate::GlmOcrGenerateModel,
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Context, Result, anyhow};

const GLM_OCR_GGUF_DIR: &str = r"D:\model_download\GLM-OCR-GGUF";
const GLM_OCR_ONNX_DIR: &str = r"D:\model_download\GLM-OCR-ONNX";

fn require_existing_dir(path: &str) -> Result<()> {
    let dir = Path::new(path);
    if !dir.exists() {
        return Err(anyhow!("model dir not found: {}", path));
    }
    if !dir.is_dir() {
        return Err(anyhow!("path is not a directory: {}", path));
    }
    Ok(())
}

fn first_file_with_extension(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;
    let mut matches = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found in {}", extension, dir))
}

fn first_file_with_extension_recursive(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

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
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found (recursive) in {}", extension, dir))
}

fn ocr_test_image_url() -> Result<String> {
    let path = std::env::current_dir()?
        .join("assets")
        .join("img")
        .join("ocr_test1.png");
    let path = path.canonicalize()?;
    url::Url::from_file_path(&path)
        .map(|url| url.to_string())
        .map_err(|_| anyhow!("failed to build file url for {}", path.display()))
}

#[test]
fn glm_ocr_gguf_files_can_load() -> Result<()> {
    let gguf_path = first_file_with_extension(GLM_OCR_GGUF_DIR, "gguf")?;
    let metadata = std::fs::metadata(&gguf_path)
        .with_context(|| format!("failed to read gguf metadata: {}", gguf_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("gguf file is empty: {}", gguf_path.display()));
    }
    Ok(())
}

#[test]
fn glm_ocr_onnx_files_can_load() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(GLM_OCR_ONNX_DIR, "onnx")?;
    let metadata = std::fs::metadata(&onnx_path)
        .with_context(|| format!("failed to read onnx metadata: {}", onnx_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("onnx file is empty: {}", onnx_path.display()));
    }
    Ok(())
}

#[test]
fn glm_ocr_load_spec_accepts_gguf() {
    let spec = LoadSpec {
        model: WhichModel::GlmOCR,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(GLM_OCR_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };
    spec.validate()
        .expect("glm-ocr should accept gguf artifact");
}

#[test]
fn glm_ocr_load_spec_accepts_onnx() {
    let spec = LoadSpec {
        model: WhichModel::GlmOCR,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(GLM_OCR_ONNX_DIR.to_string()),
            tokenizer_dir: Some(GLM_OCR_ONNX_DIR.to_string()),
            ..Default::default()
        },
    };
    spec.validate()
        .expect("glm-ocr should accept onnx artifact");
}

#[test]
fn glm_ocr_gguf_init_from_spec_can_init() -> Result<()> {
    if let Err(err) = require_existing_dir(GLM_OCR_GGUF_DIR) {
        println!("skip glm-ocr gguf init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::GlmOCR,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(GLM_OCR_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };
    GlmOcrGenerateModel::init_from_spec(&spec, None, None)?;
    Ok(())
}

#[test]
fn glm_ocr_onnx_init_from_spec_can_init() -> Result<()> {
    if let Err(err) = require_existing_dir(GLM_OCR_ONNX_DIR) {
        println!("skip glm-ocr onnx init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::GlmOCR,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(GLM_OCR_ONNX_DIR.to_string()),
            tokenizer_dir: Some(GLM_OCR_ONNX_DIR.to_string()),
            ..Default::default()
        },
    };
    GlmOcrGenerateModel::init_from_spec(&spec, None, None)?;
    Ok(())
}

#[test]
#[ignore = "manual real-model inference smoke test"]
fn glm_ocr_gguf_generate_smoke() -> Result<()> {
    if let Err(err) = require_existing_dir(GLM_OCR_GGUF_DIR) {
        println!("skip glm-ocr gguf generate smoke test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::GlmOCR,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(GLM_OCR_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };
    let ocr_test_image = ocr_test_image_url()?;
    let message = format!(
        r#"{{
            "model": "glm-ocr",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "image",
                            "image_url": {{
                                "url": "{ocr_test_image}"
                            }}
                        }},
                        {{
                            "type": "text",
                            "text": "Text Recognition:"
                        }}
                    ]
                }}
            ],
            "max_tokens": 1
        }}"#
    );
    let mes: ChatCompletionParameters = serde_json::from_str(&message)?;
    let mut model = GlmOcrGenerateModel::init_from_spec(&spec, None, None)?;
    let res = model.generate(mes)?;
    if let Some(usage) = res.usage {
        assert!(usage.total_tokens > 0);
    }
    Ok(())
}
