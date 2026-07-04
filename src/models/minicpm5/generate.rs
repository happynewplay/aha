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
        common::gguf::{load_text_bootstrap_from_gguf},
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

use super::{
    config::MiniCPM5Config,
    model::{MiniCPM5Model, resolve_minicpm5_gguf_file},
};

pub struct MiniCPM5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    model: MiniCPM5Model,
    device: Device,
    eos_token_ids: Vec<u32>,
    model_name: String,
}

impl<'a> MiniCPM5GenerateModel<'a> {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        spec.validate()?;
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec
                    .paths
                    .weight_dir
                    .as_deref()
                    .ok_or_else(|| anyhow!("weight_path is required for minicpm5 safetensors"))?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let path = spec
                    .paths
                    .gguf_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("gguf_path is required for minicpm5 gguf"))?;
                Self::init_from_gguf(path, device, dtype)
            }
            ArtifactKind::Onnx => Err(anyhow!(
                "model {} does not support artifact {:?}",
                spec.model.openai_model_id(),
                ArtifactKind::Onnx
            )),
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: MiniCPM5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.torch_dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = MiniCPM5Model::new_from_vb(vb, &cfg)?;
        Ok(Self {
            chat_template,
            tokenizer,
            model,
            device,
            eos_token_ids: normalize_eos_token_ids(cfg.eos_token_id.clone()),
            model_name: "minicpm5-1b".to_string(),
        })
    }

    pub fn init_from_gguf(
        model_file: &str,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let model_file = resolve_minicpm5_gguf_file(model_file)?;
        let device = get_device(device);
        let dtype = dtype.unwrap_or(DType::F16);
        let bootstrap = load_text_bootstrap_from_gguf(&model_file, Some(false), Some(false), Some(false))?;
        let chat_template = bootstrap
            .chat_template
            .ok_or_else(|| anyhow!("tokenizer.chat_template metadata is missing in {model_file}"))?;
        let tokenizer = bootstrap.tokenizer;
        let model = MiniCPM5Model::new_from_gguf(&model_file, &device, dtype)?;
        let mut eos_token_ids = vec![1, 130073];
        if let Some(eos) = bootstrap.eos_token_id
            && !eos_token_ids.contains(&eos)
        {
            eos_token_ids.push(eos);
        }

        Ok(Self {
            chat_template: ChatTemplate::str_init(&chat_template)?,
            tokenizer,
            model,
            device,
            eos_token_ids,
            model_name: "minicpm5-1b".to_string(),
        })
    }
}

impl<'a> GenerateModel for MiniCPM5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
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
            let logits = self.model.forward(&input_ids, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.model.clear_kv_cache();
        let response = build_completion_response(
            res,
            &self.model_name,
            Some(num_token),
            Some(prompt_tokens),
        );
        Ok(response)
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let eos_token_ids = self.eos_token_ids.clone();
        let model_name = self.model_name.clone();
        let tokenizer = &self.tokenizer;
        let device = self.device.clone();
        let model = &mut self.model;

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut stream_state = MiniCPM5StreamState::default();
            for _ in 0..sample_len {
                let logits = model.forward(&input_ids, seqlen_offset)?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = tokenizer
                    .token_decode(decode_ids)
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
                if eos_token_ids.contains(&next_token) {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
            }
            model.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}

#[derive(Default)]
struct MiniCPM5StreamState {
    tool_call_id: Option<String>,
    tool_call_content: String,
}

impl MiniCPM5StreamState {
    fn push(
        &mut self,
        decoded_token: &str,
        model_name: &str,
    ) -> Result<Option<ChatCompletionChunkResponse>> {
        match decoded_token {
            "<tool_call>" => {
                self.tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                self.tool_call_content.clear();
                Ok(None)
            }
            "</tool_call>" => {
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

fn normalize_eos_token_ids(mut eos_token_ids: Vec<u32>) -> Vec<u32> {
    for token in [1_u32, 130073_u32] {
        if !eos_token_ids.contains(&token) {
            eos_token_ids.push(token);
        }
    }
    eos_token_ids
}

#[cfg(test)]
mod tests {
    use super::MiniCPM5StreamState;
    use anyhow::Result;

    #[test]
    fn minicpm5_stream_state_buffers_tool_call_until_closing_tag() -> Result<()> {
        let mut state = MiniCPM5StreamState::default();

        assert!(state.push("hello", "minicpm5-1b")?.is_some());
        assert!(state.push("<tool_call>", "minicpm5-1b")?.is_none());
        assert!(state
            .push(r#"{"name":"lookup","arguments":{"query":"rust"}}"#, "minicpm5-1b")?
            .is_none());

        let chunk = state
            .push("</tool_call>", "minicpm5-1b")?
            .expect("expected chunk on closing tag");
        let payload = serde_json::to_string(&chunk)?;
        assert!(payload.contains("lookup"));
        assert!(payload.contains("tool_calls"));
        Ok(())
    }
}
