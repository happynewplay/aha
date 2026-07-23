use std::path::Path;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        ArtifactKind, GenerateModel, LoadSpec,
        common::{gguf::load_text_bootstrap_from_gguf, onnx::resolve_tokenizer_dir},
        lfm2_5::{
            config::Lfm2_5Config,
            model::{Lfm2_5Model, resolve_lfm2_5_gguf_file},
            onnx::Lfm2_5OnnxBackend,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

const LFM2_5_IM_END_TOKEN_ID: u32 = 7;

enum Lfm2_5Runtime {
    Safetensors(Lfm2_5Model),
    Gguf(Lfm2_5Model),
    Onnx(Lfm2_5OnnxBackend),
}

pub struct Lfm2_5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    runtime: Lfm2_5Runtime,
    device: Device,
    eos_token_id: u32,
    model_name: String,
}

impl<'a> Lfm2_5GenerateModel<'a> {
    fn forward_runtime_logits(
        runtime: &mut Lfm2_5Runtime,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let logits = match runtime {
            Lfm2_5Runtime::Safetensors(model) | Lfm2_5Runtime::Gguf(model) => model
                .forward(input_ids, seqlen_offset)?
                .squeeze(0)?
                .squeeze(0)?,
            Lfm2_5Runtime::Onnx(_) => {
                return Err(anyhow!("onnx runtime should use vector input path"));
            }
        };
        logits.to_dtype(DType::F32).map_err(Into::into)
    }

    fn clear_runtime_cache_inner(runtime: &mut Lfm2_5Runtime) {
        match runtime {
            Lfm2_5Runtime::Safetensors(model) | Lfm2_5Runtime::Gguf(model) => model.clear_cache(),
            Lfm2_5Runtime::Onnx(backend) => backend.clear_cache(),
        }
    }

    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut model = match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec.paths.weight_dir.as_deref().ok_or_else(|| {
                    anyhow!("weight_path is required for lfm2.5-350m safetensors")
                })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let path = spec
                    .paths
                    .gguf_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("gguf_path is required for lfm2.5-350m gguf"))?;
                Self::init_from_gguf(path, spec.paths.tokenizer_dir.as_deref(), device, dtype)
            }
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for lfm2.5-350m onnx"))?;
                Self::init_from_onnx(onnx_path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }?;
        model.model_name = spec.model.openai_model_id().to_string();
        Ok(model)
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Lfm2_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        if model_list.is_empty() {
            return Err(anyhow!("no safetensors files found in {path}"));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = Lfm2_5Model::new_from_vb(vb, &cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            runtime: Lfm2_5Runtime::Safetensors(model),
            device,
            eos_token_id: cfg.eos_token_id,
            model_name: "lfm2.5-350m".to_string(),
        })
    }

    pub fn init_from_gguf(
        gguf_path: &str,
        tokenizer_dir: Option<&str>,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let model_file = resolve_lfm2_5_gguf_file(gguf_path)?;
        let device = get_device(device);
        let dtype = dtype.unwrap_or(DType::F32);
        let bootstrap =
            load_text_bootstrap_from_gguf(&model_file, Some(false), Some(false), Some(false))?;
        let chat_template = if let Some(dir) = tokenizer_dir {
            ChatTemplate::init(dir)?
        } else if let Some(chat_template_str) = bootstrap.chat_template {
            ChatTemplate::str_init(&chat_template_str)?
        } else {
            let parent = Path::new(&model_file)
                .parent()
                .and_then(|path| path.to_str())
                .ok_or_else(|| anyhow!("cannot resolve gguf parent directory for {model_file}"))?;
            ChatTemplate::init(parent)?
        };
        let cfg_path = if let Some(dir) = tokenizer_dir {
            Path::new(dir).join("config.json")
        } else {
            Path::new(&model_file)
                .parent()
                .ok_or_else(|| anyhow!("cannot resolve gguf parent directory for {model_file}"))?
                .join("config.json")
        };
        let cfg: Lfm2_5Config = serde_json::from_slice(&std::fs::read(cfg_path)?)?;
        let eos_token_id = bootstrap.eos_token_id.unwrap_or(cfg.eos_token_id);
        let model = Lfm2_5Model::new_from_gguf(&model_file, &cfg, &device, dtype)?;
        let model_name = Path::new(&model_file)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("lfm2.5-350m")
            .to_string();

        Ok(Self {
            chat_template,
            tokenizer: bootstrap.tokenizer,
            runtime: Lfm2_5Runtime::Gguf(model),
            device,
            eos_token_id,
            model_name,
        })
    }

    pub fn init_from_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir = resolve_tokenizer_dir(
            onnx_path,
            tokenizer_dir,
            &["tokenizer.json", "config.json", "chat_template.jinja"],
        )?;
        let base_path = tokenizer_dir.to_string_lossy().to_string();
        let chat_template = ChatTemplate::init(&base_path)?;
        let tokenizer = TokenizerModel::init(&base_path)?;
        let cfg: Lfm2_5Config =
            serde_json::from_slice(&std::fs::read(tokenizer_dir.join("config.json"))?)?;
        let model_name = tokenizer_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("lfm2.5-350m")
            .to_string();
        let backend = Lfm2_5OnnxBackend::load(onnx_path)?;

        Ok(Self {
            chat_template,
            tokenizer,
            runtime: Lfm2_5Runtime::Onnx(backend),
            device: Device::Cpu,
            eos_token_id: cfg.eos_token_id,
            model_name,
        })
    }

    fn forward_logits(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        Self::forward_runtime_logits(&mut self.runtime, input_ids, seqlen_offset)
    }

    fn clear_runtime_cache(&mut self) {
        Self::clear_runtime_cache_inner(&mut self.runtime);
    }

    fn generate_with_onnx(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut current_ids = self.tokenizer.text_encode_vec(mes_render, true)?;
        let prompt_tokens = current_ids.len() as u32;
        let mut position_start = 0usize;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        let onnx_backend = match &mut self.runtime {
            Lfm2_5Runtime::Onnx(backend) => backend,
            _ => return Err(anyhow!("lfm2.5 onnx runtime is not initialized")),
        };
        for _ in 0..sample_len {
            let logits = onnx_backend.forward_logits(&current_ids, position_start)?;
            let vocab_size = logits.len();
            let logits = Tensor::from_vec(logits, vocab_size, &self.device)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            position_start += current_ids.len();
            current_ids = vec![next_token];
        }
        let num_token = generate.len() as u32;
        let res = decode_tokens_for_completion(&self.tokenizer, &generate, self.eos_token_id)?;
        onnx_backend.clear_cache();
        Ok(build_completion_response(
            res,
            &self.model_name,
            Some(num_token),
            Some(prompt_tokens),
        ))
    }

    fn generate_stream_with_onnx(
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let tokenizer = &self.tokenizer;
        let model_name = self.model_name.clone();
        let eos_token_id = self.eos_token_id;
        let device = self.device.clone();
        let mut current_ids = tokenizer.text_encode_vec(mes_render, true)?;
        let mut position_start = 0usize;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let onnx_backend = match &mut self.runtime {
            Lfm2_5Runtime::Onnx(backend) => backend,
            _ => return Err(anyhow!("lfm2.5 onnx runtime is not initialized")),
        };

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut stream_state = Lfm2_5StreamState::default();
            for _ in 0..sample_len {
                let logits = onnx_backend.forward_logits(&current_ids, position_start)?;
                let vocab_size = logits.len();
                let logits = Tensor::from_vec(logits, vocab_size, &device)?;
                let next_token = logit_processor.sample(&logits)?;
                if eos_token_id == LFM2_5_IM_END_TOKEN_ID && next_token == eos_token_id {
                    break;
                }
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = tokenizer
                    .token_decode_with_special_tokens(decode_ids)
                    .map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                if decoded_token.contains('�') {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    position_start += current_ids.len();
                    current_ids = vec![next_token];
                    continue;
                }
                error_tokens.clear();
                if let Some(chunk) = stream_state.push(&decoded_token, &model_name)? {
                    yield Ok(chunk);
                }
                position_start += current_ids.len();
                current_ids = vec![next_token];
                if next_token == eos_token_id {
                    break;
                }
            }
            onnx_backend.clear_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}

impl<'a> GenerateModel for Lfm2_5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        if matches!(&self.runtime, Lfm2_5Runtime::Onnx(_)) {
            return self.generate_with_onnx(mes);
        }

        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);

        for _ in 0..sample_len {
            let logits = self.forward_logits(&input_ids, seqlen_offset)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        let num_token = generate.len() as u32;
        let res = decode_tokens_for_completion(&self.tokenizer, &generate, self.eos_token_id)?;
        self.clear_runtime_cache();
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
        if matches!(&self.runtime, Lfm2_5Runtime::Onnx(_)) {
            return self.generate_stream_with_onnx(mes);
        }

        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let model_name = self.model_name.clone();
        let tokenizer = &self.tokenizer;
        let device = self.device.clone();
        let eos_token_id = self.eos_token_id;
        let runtime = &mut self.runtime;

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut stream_state = Lfm2_5StreamState::default();
            for _ in 0..sample_len {
                let logits = Self::forward_runtime_logits(runtime, &input_ids, seqlen_offset)?;
                let next_token = logit_processor.sample(&logits)?;
                if eos_token_id == LFM2_5_IM_END_TOKEN_ID && next_token == eos_token_id {
                    break;
                }
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = tokenizer
                    .token_decode_with_special_tokens(decode_ids)
                    .map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                if decoded_token.contains('�') {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                    continue;
                }
                error_tokens.clear();
                if let Some(chunk) = stream_state.push(&decoded_token, &model_name)? {
                    yield Ok(chunk);
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                if next_token == eos_token_id {
                    break;
                }
            }
            Self::clear_runtime_cache_inner(runtime);
        };
        Ok(Box::new(Box::pin(stream)))
    }
}

#[derive(Default)]
struct Lfm2_5StreamState {
    tool_call_id: Option<String>,
    tool_call_content: String,
}

impl Lfm2_5StreamState {
    fn push(
        &mut self,
        decoded_token: &str,
        model_name: &str,
    ) -> Result<Option<ChatCompletionChunkResponse>> {
        match decoded_token {
            "<|tool_call_start|>" | "<tool_call>" => {
                self.tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                self.tool_call_content.clear();
                Ok(None)
            }
            "<|tool_call_end|>" | "</tool_call>" => {
                let chunk = build_completion_chunk_response(
                    decoded_token.to_string(),
                    model_name,
                    self.tool_call_id.clone(),
                    Some(self.tool_call_content.clone()),
                );
                self.tool_call_id = None;
                self.tool_call_content.clear();
                Ok(Some(chunk))
            }
            _ if self.tool_call_id.is_some() => {
                self.tool_call_content.push_str(decoded_token);
                Ok(None)
            }
            _ => Ok(Some(build_completion_chunk_response(
                decoded_token.to_string(),
                model_name,
                None,
                None,
            ))),
        }
    }
}

trait Lfm2_5TokenDecoder {
    fn decode_with_special_tokens(&self, tokens: Vec<u32>) -> Result<String>;
}

impl Lfm2_5TokenDecoder for TokenizerModel {
    fn decode_with_special_tokens(&self, tokens: Vec<u32>) -> Result<String> {
        self.token_decode_with_special_tokens(tokens)
    }
}

fn decode_tokens_for_completion<T: Lfm2_5TokenDecoder>(
    tokenizer: &T,
    tokens: &[u32],
    eos_token_id: u32,
) -> Result<String> {
    let filtered_tokens = if eos_token_id == LFM2_5_IM_END_TOKEN_ID {
        tokens
            .iter()
            .copied()
            .filter(|token| *token != eos_token_id)
            .collect::<Vec<u32>>()
    } else {
        tokens.to_vec()
    };
    tokenizer.decode_with_special_tokens(filtered_tokens)
}

#[cfg(test)]
mod tests {
    use super::{Lfm2_5StreamState, decode_tokens_for_completion};
    use anyhow::Result;
    use std::{cell::RefCell, collections::VecDeque};

    struct MockTokenizer {
        decode_map: RefCell<VecDeque<(Vec<u32>, String)>>,
    }

    impl MockTokenizer {
        fn new(entries: Vec<(Vec<u32>, &str)>) -> Self {
            Self {
                decode_map: RefCell::new(
                    entries
                        .into_iter()
                        .map(|(tokens, output)| (tokens, output.to_string()))
                        .collect(),
                ),
            }
        }
    }

    impl super::Lfm2_5TokenDecoder for MockTokenizer {
        fn decode_with_special_tokens(&self, tokens: Vec<u32>) -> Result<String> {
            let (expected, output) = self
                .decode_map
                .borrow_mut()
                .pop_front()
                .expect("unexpected decode call in test");
            assert_eq!(tokens, expected);
            Ok(output)
        }
    }

    #[test]
    fn lfm2_5_filters_im_end_before_decode() -> Result<()> {
        let mut tokenizer = MockTokenizer::new(vec![(vec![11, 12], "decoded")]);
        let res = decode_tokens_for_completion(&mut tokenizer, &[11, 7, 12], 7)?;
        assert_eq!(res, "decoded");
        Ok(())
    }

    #[test]
    fn lfm2_5_stream_state_emits_openai_tool_calls_on_closing_tag() -> Result<()> {
        let mut state = Lfm2_5StreamState::default();
        assert!(state.push("<|tool_call_start|>", "lfm2.5-350m")?.is_none());
        assert!(
            state
                .push(
                    r#"{"name":"lookup","arguments":{"query":"rust"}}"#,
                    "lfm2.5-350m"
                )?
                .is_none()
        );

        let chunk = state
            .push("<|tool_call_end|>", "lfm2.5-350m")?
            .expect("expected chunk on closing tag");
        let payload = serde_json::to_string(&chunk)?;
        assert!(payload.contains("tool_calls"));
        assert!(payload.contains("lookup"));
        Ok(())
    }
}
