//! GLM-OCR Inference and Generation
use std::{collections::HashMap, path::Path};

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    models::{
        GenerateModel,
        artifact::{ArtifactKind, LoadSpec},
        common::{
            gguf::{Gguf, load_gguf_file, load_text_bootstrap_from_gguf},
            onnx::resolve_tokenizer_dir,
        },
        glm_ocr::{
            config::{
                GlmOcrConfig, GlmOcrGenerationConfig, GlmOcrRopeParameters, GlmOcrTextConfig,
                GlmOcrVisionConfig,
            },
            model::GlmOcrModel,
            onnx::GlmOcrOnnxBackend,
            processor::GlmOcrProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, extract_user_text,
        find_type_files, get_device, get_dtype, get_logit_processor, img_utils::extract_image_url,
    },
};

const DEFAULT_GLM_OCR_SHORTEST_EDGE: usize = 12_544;
const DEFAULT_GLM_OCR_LONGEST_EDGE: usize = 9_633_792;
const DEFAULT_GLM_OCR_IMAGE_TOKEN_ID: u32 = 59_280;
const DEFAULT_GLM_OCR_IMAGE_START_TOKEN_ID: u32 = 59_256;
const DEFAULT_GLM_OCR_IMAGE_END_TOKEN_ID: u32 = 59_257;
const DEFAULT_GLM_OCR_VIDEO_TOKEN_ID: u32 = 59_281;
const DEFAULT_GLM_OCR_VIDEO_START_TOKEN_ID: u32 = 59_258;
const DEFAULT_GLM_OCR_VIDEO_END_TOKEN_ID: u32 = 59_259;

pub struct GlmOcrGenerateModel {
    tokenizer: TokenizerModel,
    processor: GlmOcrProcessor,
    model: Option<GlmOcrModel>,
    onnx_backend: Option<GlmOcrOnnxBackend>,
    device: Device,
    eos_token_ids: Vec<u32>,
    model_name: String,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
}

impl GlmOcrGenerateModel {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path =
                    spec.paths.weight_dir.as_deref().ok_or_else(|| {
                        anyhow!("weight_path is required for glm-ocr safetensors")
                    })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let gguf_path = spec
                    .paths
                    .gguf_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("gguf_path is required for glm-ocr gguf"))?;
                Self::init_from_gguf(gguf_path, spec.paths.mmproj_path.as_deref(), device, dtype)
            }
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for glm-ocr onnx"))?;
                Self::init_from_onnx(onnx_path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: GlmOcrConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = GlmOcrProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = GlmOcrModel::new(vb, cfg.clone())?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: GlmOcrGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;

        Ok(Self {
            tokenizer,
            processor,
            model: Some(model),
            onnx_backend: None,
            device,
            eos_token_ids: generation_config.eos_token_id.clone(),
            model_name: "glm-ocr".to_string(),
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }

    pub fn init_from_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir = resolve_tokenizer_dir(
            onnx_path,
            tokenizer_dir,
            &[
                "tokenizer.json",
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
            ],
        )?;
        let base_path = tokenizer_dir.to_string_lossy().to_string();
        let tokenizer = TokenizerModel::init(&base_path)?;
        let cfg: GlmOcrConfig =
            serde_json::from_slice(&std::fs::read(tokenizer_dir.join("config.json"))?)?;
        let generation_config: GlmOcrGenerationConfig = serde_json::from_slice(&std::fs::read(
            tokenizer_dir.join("generation_config.json"),
        )?)?;
        let processor = GlmOcrProcessor::new(&base_path, &Device::Cpu, DType::F32)?;
        let onnx_backend =
            GlmOcrOnnxBackend::load(onnx_path, cfg.vision_config.spatial_merge_size)?;
        let model_name = tokenizer_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("glm-ocr")
            .to_string();

        Ok(Self {
            tokenizer,
            processor,
            model: None,
            onnx_backend: Some(onnx_backend),
            device: Device::Cpu,
            eos_token_ids: generation_config.eos_token_id,
            model_name,
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }

    pub fn init_from_gguf(
        gguf_path: &str,
        mmproj_path: Option<&str>,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let (model_file, mmproj_file) = resolve_glm_ocr_gguf_files(gguf_path, mmproj_path)?;
        let device = get_device(device);
        let dtype = dtype.unwrap_or(DType::F32);
        let bootstrap =
            load_text_bootstrap_from_gguf(&model_file, Some(false), Some(false), Some(false))?;
        let mut model_gguf = load_gguf_file(&model_file, &device)?;
        let mut mmproj_gguf = load_gguf_file(&mmproj_file, &device)?;
        let cfg = build_glm_ocr_gguf_config(&mut model_gguf, &mut mmproj_gguf)?;
        let processor = build_glm_ocr_gguf_processor(&mut mmproj_gguf, &device, dtype)?;
        let tensors = load_glm_ocr_gguf_tensors(&mut model_gguf, &mut mmproj_gguf, &device, dtype)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let model = GlmOcrModel::new(vb, cfg.clone())?;
        let eos_token_ids = build_glm_ocr_gguf_eos_tokens(&mut model_gguf)?;
        let image_token_id = resolve_special_token_id(
            &bootstrap.tokenizer,
            "<|image|>",
            DEFAULT_GLM_OCR_IMAGE_TOKEN_ID,
        );
        let image_start_token_id = resolve_special_token_id(
            &bootstrap.tokenizer,
            "<|begin_of_image|>",
            DEFAULT_GLM_OCR_IMAGE_START_TOKEN_ID,
        );
        let image_end_token_id = resolve_special_token_id(
            &bootstrap.tokenizer,
            "<|end_of_image|>",
            DEFAULT_GLM_OCR_IMAGE_END_TOKEN_ID,
        );
        let model_name = Path::new(&model_file)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("glm-ocr")
            .to_string();

        Ok(Self {
            tokenizer: bootstrap.tokenizer,
            processor,
            model: Some(model),
            onnx_backend: None,
            device,
            eos_token_ids,
            model_name,
            image_token_id,
            image_start_token_id,
            image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }

    fn clear_runtime_cache(&mut self) {
        if let Some(model) = self.model.as_mut() {
            model.clear_kv_cache();
        }
        if let Some(onnx_backend) = self.onnx_backend.as_mut() {
            onnx_backend.clear_cache();
        }
    }
}

impl GenerateModel for GlmOcrGenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);

        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        let mut prompt = extract_user_text(&mes)?;
        if prompt.is_empty() {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        if let Some(onnx_backend) = self.onnx_backend.as_mut() {
            let mut current_ids = processed.input_ids.squeeze(0)?.to_vec1::<u32>()?;
            let pixel_values = Some(processed.pixel_values);
            let image_grid_thw = Some(processed.grid_thw);
            let image_mask = Some(processed.image_mask);
            let prompt_tokens = current_ids.len() as u32;
            let sample_len = mes.max_tokens.unwrap_or(512);
            let mut position_start = 0usize;
            let mut generate = Vec::new();

            for _ in 0..sample_len {
                let logits = onnx_backend.forward_logits(
                    &current_ids,
                    if position_start == 0 {
                        image_mask.as_ref()
                    } else {
                        None
                    },
                    if position_start == 0 {
                        pixel_values.as_ref()
                    } else {
                        None
                    },
                    if position_start == 0 {
                        image_grid_thw.as_ref()
                    } else {
                        None
                    },
                    position_start,
                )?;
                let vocab = logits.len();
                let logits = Tensor::from_vec(logits, vocab, &self.device)?;
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                if self.eos_token_ids.contains(&next_token) {
                    break;
                }
                position_start += current_ids.len();
                current_ids = vec![next_token];
            }

            self.clear_runtime_cache();
            let num_token = generate.len() as u32;
            let res = self.tokenizer.token_decode(generate)?;
            return Ok(build_completion_response(
                res,
                &self.model_name,
                Some(num_token),
                Some(prompt_tokens),
            ));
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow!("glm-ocr native runtime is not initialized"))?;
        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_grid_thw = Some(processed.grid_thw);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(512);

        for _ in 0..sample_len {
            let is_first_pass = seqlen_offset == 0;
            let logits = model.forward(
                &input_ids,
                if is_first_pass {
                    pixel_values.as_ref()
                } else {
                    None
                },
                if is_first_pass {
                    image_grid_thw.as_ref()
                } else {
                    None
                },
                if is_first_pass {
                    image_mask.as_ref()
                } else {
                    None
                },
                seqlen_offset,
            )?;
            let logits = logits.i((0, seq_len - 1, ..))?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;

            generate.push(next_token);
            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        self.clear_runtime_cache();
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        Ok(build_completion_response(
            res,
            &self.model_name,
            Some(num_token),
            Some(prompt_tokens),
        ))
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);

        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        let mut prompt = extract_user_text(&mes)?;
        if prompt.is_empty() {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        if let Some(onnx_backend) = self.onnx_backend.as_mut() {
            let tokenizer = &self.tokenizer;
            let model_name = self.model_name.clone();
            let eos_token_ids = self.eos_token_ids.clone();
            let device = self.device.clone();
            let mut current_ids = processed.input_ids.squeeze(0)?.to_vec1::<u32>()?;
            let mut position_start = 0usize;
            let sample_len = mes.max_tokens.unwrap_or(512);
            let mut pixel_values = Some(processed.pixel_values);
            let mut image_grid_thw = Some(processed.grid_thw);
            let mut image_mask = Some(processed.image_mask);

            let stream = stream! {
                let mut error_tokens = Vec::new();
                for _ in 0..sample_len {
                    let logits = onnx_backend.forward_logits(
                        &current_ids,
                        image_mask.as_ref(),
                        pixel_values.as_ref(),
                        image_grid_thw.as_ref(),
                        position_start,
                    )?;
                    let vocab = logits.len();
                    let logits = Tensor::from_vec(logits, vocab, &device)?;
                    let next_token = logit_processor.sample(&logits)?;

                    let mut decode_ids = Vec::new();
                    if !error_tokens.is_empty() {
                        decode_ids.extend_from_slice(&error_tokens);
                    }
                    decode_ids.push(next_token);

                    let decoded_token = tokenizer
                        .token_decode(decode_ids)
                        .map_err(|e| anyhow!(format!("decode error: {e}")))?;
                    if decoded_token.contains("�") {
                        error_tokens.push(next_token);
                        if error_tokens.len() > 3 {
                            error_tokens.clear();
                        }
                        position_start += current_ids.len();
                        current_ids = vec![next_token];
                        pixel_values = None;
                        image_grid_thw = None;
                        image_mask = None;
                        continue;
                    }
                    error_tokens.clear();

                    let chunk = build_completion_chunk_response(decoded_token, &model_name, None, None);
                    yield Ok(chunk);

                    if eos_token_ids.contains(&next_token) {
                        break;
                    }
                    position_start += current_ids.len();
                    current_ids = vec![next_token];
                    pixel_values = None;
                    image_grid_thw = None;
                    image_mask = None;
                }
                onnx_backend.clear_cache();
            };
            return Ok(Box::new(Box::pin(stream)));
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow!("glm-ocr native runtime is not initialized"))?;
        let tokenizer = &self.tokenizer;
        let model_name = self.model_name.clone();
        let eos_token_ids = self.eos_token_ids.clone();
        let device = self.device.clone();
        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_grid_thw = Some(processed.grid_thw);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let sample_len = mes.max_tokens.unwrap_or(512);

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut pixel_values = pixel_values.as_ref();
            let image_grid_thw = image_grid_thw.as_ref();
            let mut image_mask = image_mask.as_ref();
            for _ in 0..sample_len {
                let logits = model.forward(
                    &input_ids,
                    pixel_values,
                    image_grid_thw,
                    image_mask,
                    seqlen_offset,
                ).map_err(|e| anyhow!(format!("forward error: {e}")))?;
                let logits = logits
                    .i((0, seq_len - 1, ..))
                    .map_err(|e| anyhow!(format!("index error: {e}")))?
                    .to_dtype(DType::F32)
                    .map_err(|e| anyhow!(format!("dtype error: {e}")))?;

                let next_token = logit_processor
                    .sample(&logits)
                    .map_err(|e| anyhow!(format!("sample error: {e}")))?;

                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);

                let decoded_token = tokenizer
                    .token_decode(decode_ids)
                    .map_err(|e| anyhow!(format!("decode error: {e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)
                        .map_err(|e| anyhow!(format!("tensor error: {e}")))?;
                    pixel_values = None;
                    image_mask = None;
                    continue;
                }
                error_tokens.clear();

                let chunk = build_completion_chunk_response(decoded_token, &model_name, None, None);
                yield Ok(chunk);

                if eos_token_ids.contains(&next_token) {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)
                    .map_err(|e| anyhow!(format!("tensor error: {e}")))?;
                pixel_values = None;
                image_mask = None;
            }
            model.clear_kv_cache();
        };

        Ok(Box::new(Box::pin(stream)))
    }
}

fn resolve_glm_ocr_gguf_files(
    gguf_path: &str,
    mmproj_path: Option<&str>,
) -> Result<(String, String)> {
    fn find_gguf_file(dir: &Path, marker: &str, negate: bool) -> Result<String> {
        let mut matches = std::fs::read_dir(dir)?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                    && if negate {
                        !file_name.contains(marker)
                    } else {
                        file_name.contains(marker)
                    }
            })
            .collect::<Vec<_>>();
        matches.sort();
        matches
            .into_iter()
            .next()
            .map(|path| path.to_string_lossy().to_string())
            .ok_or_else(|| {
                anyhow!(
                    "unable to locate gguf component {} under {}",
                    marker,
                    dir.display()
                )
            })
    }

    let model_path = Path::new(gguf_path);
    if !model_path.exists() {
        return Err(anyhow!("gguf model path not found: {}", gguf_path));
    }

    let model_file = if model_path.is_dir() {
        find_gguf_file(model_path, "mmproj", true)?
    } else {
        let file_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if !file_name.ends_with(".gguf") || file_name.contains("mmproj") {
            return Err(anyhow!("glm-ocr gguf model file is invalid: {}", gguf_path));
        }
        model_path.to_string_lossy().to_string()
    };

    let mmproj_file = if let Some(mmproj_path) = mmproj_path {
        let mmproj = Path::new(mmproj_path);
        if !mmproj.exists() {
            return Err(anyhow!("glm-ocr mmproj path not found: {}", mmproj_path));
        }
        if mmproj.is_dir() {
            find_gguf_file(mmproj, "mmproj", false)?
        } else {
            mmproj.to_string_lossy().to_string()
        }
    } else {
        let search_dir = Path::new(&model_file)
            .parent()
            .ok_or_else(|| anyhow!("glm-ocr gguf parent directory not found"))?;
        find_gguf_file(search_dir, "mmproj", false)?
    };

    Ok((model_file, mmproj_file))
}

fn resolve_special_token_id(tokenizer: &TokenizerModel, token: &str, fallback: u32) -> u32 {
    tokenizer.tokenizer.token_to_id(token).unwrap_or(fallback)
}

fn build_glm_ocr_gguf_eos_tokens<R: std::io::Read + std::io::Seek>(
    gguf: &mut Gguf<R>,
) -> Result<Vec<u32>> {
    let mut eos = vec![gguf.get_matedata("tokenizer.ggml.eos_token_id")?.to_u32()?];
    if let Ok(eot) = gguf.get_matedata("tokenizer.ggml.eot_token_id")
        && let Ok(eot) = eot.to_u32()
        && !eos.contains(&eot)
    {
        eos.push(eot);
    }
    Ok(eos)
}

fn build_glm_ocr_gguf_config<
    R1: std::io::Read + std::io::Seek,
    R2: std::io::Read + std::io::Seek,
>(
    model_gguf: &mut Gguf<R1>,
    mmproj_gguf: &mut Gguf<R2>,
) -> Result<GlmOcrConfig> {
    let rope_sections = model_gguf
        .get_matedata("glm4.rope.dimension_sections")?
        .to_vec()?
        .iter()
        .map(|value| {
            value
                .to_i32()
                .map(|value| value as usize)
                .or_else(|_| value.to_u32().map(|value| value as usize))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let image_size = mmproj_gguf
        .get_matedata("clip.vision.image_size")?
        .to_u32()? as usize;
    let patch_size = mmproj_gguf
        .get_matedata("clip.vision.patch_size")?
        .to_u32()? as usize;

    Ok(GlmOcrConfig {
        model_type: "glm_ocr".to_string(),
        vision_config: GlmOcrVisionConfig {
            depth: mmproj_gguf
                .get_matedata("clip.vision.block_count")?
                .to_u32()? as usize,
            hidden_size: mmproj_gguf
                .get_matedata("clip.vision.embedding_length")?
                .to_u32()? as usize,
            hidden_act: candle_nn::Activation::Silu,
            attention_bias: true,
            num_heads: mmproj_gguf
                .get_matedata("clip.vision.attention.head_count")?
                .to_u32()? as usize,
            in_channels: 3,
            image_size,
            patch_size,
            rms_norm_eps: mmproj_gguf
                .get_matedata("clip.vision.attention.layer_norm_epsilon")?
                .to_f32()? as f64,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: mmproj_gguf
                .get_matedata("clip.vision.projection_dim")?
                .to_u32()? as usize,
            intermediate_size: mmproj_gguf
                .get_matedata("clip.vision.feed_forward_length")?
                .to_u32()? as usize,
            initializer_range: 0.02,
            rope_theta: 10_000.0,
        },
        text_config: GlmOcrTextConfig {
            vocab_size: model_gguf
                .get_matedata("tokenizer.ggml.tokens")?
                .to_vec()?
                .len(),
            hidden_size: model_gguf.get_matedata("glm4.embedding_length")?.to_u32()? as usize,
            intermediate_size: model_gguf
                .get_matedata("glm4.feed_forward_length")?
                .to_u32()? as usize,
            num_hidden_layers: model_gguf.get_matedata("glm4.block_count")?.to_u32()? as usize,
            num_attention_heads: model_gguf
                .get_matedata("glm4.attention.head_count")?
                .to_u32()? as usize,
            num_key_value_heads: model_gguf
                .get_matedata("glm4.attention.head_count_kv")?
                .to_u32()? as usize,
            head_dim: Some(
                model_gguf
                    .get_matedata("glm4.attention.key_length")?
                    .to_u32()? as usize,
            ),
            max_position_embeddings: model_gguf.get_matedata("glm4.context_length")?.to_u32()?
                as usize,
            rms_norm_eps: model_gguf
                .get_matedata("glm4.attention.layer_norm_rms_epsilon")?
                .to_f32()? as f64,
            hidden_act: candle_nn::Activation::Silu,
            use_cache: true,
            rope_parameters: GlmOcrRopeParameters {
                rope_type: "default".to_string(),
                mrope_section: rope_sections,
                partial_rotary_factor: 1.0,
                rope_theta: model_gguf.get_matedata("glm4.rope.freq_base")?.to_f32()?,
            },
            eos_token_id: build_glm_ocr_gguf_eos_tokens(model_gguf)?,
            dtype: "float32".to_string(),
        },
        image_token_id: DEFAULT_GLM_OCR_IMAGE_TOKEN_ID,
        video_token_id: DEFAULT_GLM_OCR_VIDEO_TOKEN_ID,
        image_start_token_id: DEFAULT_GLM_OCR_IMAGE_START_TOKEN_ID,
        image_end_token_id: DEFAULT_GLM_OCR_IMAGE_END_TOKEN_ID,
        video_start_token_id: DEFAULT_GLM_OCR_VIDEO_START_TOKEN_ID,
        video_end_token_id: DEFAULT_GLM_OCR_VIDEO_END_TOKEN_ID,
    })
}

fn build_glm_ocr_gguf_processor<R: std::io::Read + std::io::Seek>(
    mmproj_gguf: &mut Gguf<R>,
    device: &Device,
    dtype: DType,
) -> Result<GlmOcrProcessor> {
    let image_mean = mmproj_gguf
        .get_matedata("clip.vision.image_mean")?
        .to_vec()?
        .iter()
        .map(|value| value.to_f32())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let image_std = mmproj_gguf
        .get_matedata("clip.vision.image_std")?
        .to_vec()?
        .iter()
        .map(|value| value.to_f32())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let patch_size = mmproj_gguf
        .get_matedata("clip.vision.patch_size")?
        .to_u32()? as usize;

    Ok(GlmOcrProcessor::from_params(
        image_mean,
        image_std,
        DEFAULT_GLM_OCR_SHORTEST_EDGE,
        DEFAULT_GLM_OCR_LONGEST_EDGE,
        patch_size,
        2,
        2,
        device,
        dtype,
    ))
}

fn load_glm_ocr_gguf_tensors<
    R1: std::io::Read + std::io::Seek,
    R2: std::io::Read + std::io::Seek,
>(
    model_gguf: &mut Gguf<R1>,
    mmproj_gguf: &mut Gguf<R2>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    insert_gguf_tensor(
        model_gguf,
        &mut tensors,
        "token_embd.weight",
        "model.language_model.embed_tokens.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        model_gguf,
        &mut tensors,
        "output.weight",
        "lm_head.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        model_gguf,
        &mut tensors,
        "output_norm.weight",
        "model.language_model.norm.weight",
        device,
        dtype,
    )?;

    let num_layers = model_gguf.get_matedata("glm4.block_count")?.to_u32()? as usize;
    for idx in 0..num_layers {
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.attn_q.weight"),
            &format!("model.language_model.layers.{idx}.self_attn.q_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.attn_k.weight"),
            &format!("model.language_model.layers.{idx}.self_attn.k_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.attn_v.weight"),
            &format!("model.language_model.layers.{idx}.self_attn.v_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.attn_output.weight"),
            &format!("model.language_model.layers.{idx}.self_attn.o_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.attn_norm.weight"),
            &format!("model.language_model.layers.{idx}.input_layernorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.post_attention_norm.weight"),
            &format!("model.language_model.layers.{idx}.post_self_attn_layernorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.ffn_norm.weight"),
            &format!("model.language_model.layers.{idx}.post_attention_layernorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.post_ffw_norm.weight"),
            &format!("model.language_model.layers.{idx}.post_mlp_layernorm.weight"),
            device,
            dtype,
        )?;
        insert_text_gate_up_gguf_tensor(
            model_gguf,
            &mut tensors,
            idx,
            &format!("model.language_model.layers.{idx}.mlp.gate_up_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            model_gguf,
            &mut tensors,
            &format!("blk.{idx}.ffn_down.weight"),
            &format!("model.language_model.layers.{idx}.mlp.down_proj.weight"),
            device,
            dtype,
        )?;
    }

    let patch_weight_0 = take_gguf_tensor(mmproj_gguf, "v.patch_embd.weight", device, dtype)?;
    let patch_weight_1 = take_gguf_tensor(mmproj_gguf, "v.patch_embd.weight.1", device, dtype)?;
    let patch_weight = Tensor::cat(
        &[&patch_weight_0.unsqueeze(2)?, &patch_weight_1.unsqueeze(2)?],
        2,
    )?;
    tensors.insert(
        "model.visual.patch_embed.proj.weight".to_string(),
        patch_weight,
    );
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "v.patch_embd.bias",
        "model.visual.patch_embed.proj.bias",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "v.post_ln.weight",
        "model.visual.post_layernorm.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.patch_merger.weight",
        "model.visual.downsample.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.patch_merger.bias",
        "model.visual.downsample.bias",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.model.fc.weight",
        "model.visual.merger.proj.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.post_norm.weight",
        "model.visual.merger.post_projection_norm.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.post_norm.bias",
        "model.visual.merger.post_projection_norm.bias",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.gate.weight",
        "model.visual.merger.gate_proj.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.up.weight",
        "model.visual.merger.up_proj.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        mmproj_gguf,
        &mut tensors,
        "mm.down.weight",
        "model.visual.merger.down_proj.weight",
        device,
        dtype,
    )?;

    let vision_depth = mmproj_gguf
        .get_matedata("clip.vision.block_count")?
        .to_u32()? as usize;
    for idx in 0..vision_depth {
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ln1.weight"),
            &format!("model.visual.blocks.{idx}.norm1.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ln2.weight"),
            &format!("model.visual.blocks.{idx}.norm2.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_qkv.weight"),
            &format!("model.visual.blocks.{idx}.attn.qkv.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_qkv.bias"),
            &format!("model.visual.blocks.{idx}.attn.qkv.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_out.weight"),
            &format!("model.visual.blocks.{idx}.attn.proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_out.bias"),
            &format!("model.visual.blocks.{idx}.attn.proj.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_q_norm.weight"),
            &format!("model.visual.blocks.{idx}.attn.q_norm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.attn_k_norm.weight"),
            &format!("model.visual.blocks.{idx}.attn.k_norm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_gate.weight"),
            &format!("model.visual.blocks.{idx}.mlp.gate_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_gate.bias"),
            &format!("model.visual.blocks.{idx}.mlp.gate_proj.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_up.weight"),
            &format!("model.visual.blocks.{idx}.mlp.up_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_up.bias"),
            &format!("model.visual.blocks.{idx}.mlp.up_proj.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_down.weight"),
            &format!("model.visual.blocks.{idx}.mlp.down_proj.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            mmproj_gguf,
            &mut tensors,
            &format!("v.blk.{idx}.ffn_down.bias"),
            &format!("model.visual.blocks.{idx}.mlp.down_proj.bias"),
            device,
            dtype,
        )?;
    }

    Ok(tensors)
}

fn take_gguf_tensor<R: std::io::Read + std::io::Seek>(
    gguf: &mut Gguf<R>,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let tensor = gguf
        .get_dequantized(name)
        .map_err(|err| anyhow!("failed to load gguf tensor {}: {}", name, err))?
        .to_device(device)
        .map_err(|err| anyhow!("failed to move gguf tensor {}: {}", name, err))?;
    tensor
        .to_dtype(dtype)
        .map_err(|err| anyhow!("failed to convert gguf tensor {}: {}", name, err))
}

fn insert_gguf_tensor<R: std::io::Read + std::io::Seek>(
    gguf: &mut Gguf<R>,
    tensors: &mut HashMap<String, Tensor>,
    gguf_name: &str,
    target_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let tensor = take_gguf_tensor(gguf, gguf_name, device, dtype)?;
    tensors.insert(target_name.to_string(), tensor);
    Ok(())
}

fn insert_combined_gguf_tensors<R: std::io::Read + std::io::Seek>(
    gguf: &mut Gguf<R>,
    tensors: &mut HashMap<String, Tensor>,
    gate_name: &str,
    up_name: &str,
    target_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let gate = take_gguf_tensor(gguf, gate_name, device, dtype)?;
    let up = take_gguf_tensor(gguf, up_name, device, dtype)?;
    let tensor = Tensor::cat(&[&gate, &up], 0)?;
    tensors.insert(target_name.to_string(), tensor);
    Ok(())
}

fn insert_text_gate_up_gguf_tensor<R: std::io::Read + std::io::Seek>(
    gguf: &mut Gguf<R>,
    tensors: &mut HashMap<String, Tensor>,
    layer_idx: usize,
    target_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let fused_name = format!("blk.{layer_idx}.ffn_up.weight");
    let gate_name = format!("blk.{layer_idx}.ffn_gate.weight");

    if gguf.has_tensor(&gate_name) {
        return insert_combined_gguf_tensors(
            gguf,
            tensors,
            &gate_name,
            &fused_name,
            target_name,
            device,
            dtype,
        );
    }

    insert_gguf_tensor(gguf, tensors, &fused_name, target_name, device, dtype)
}
