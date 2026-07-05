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
        lfm2_5::{config::Lfm2_5Config, model::Lfm2_5Model},
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

const LFM2_5_IM_END_TOKEN_ID: u32 = 7;

pub struct Lfm2_5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    model: Lfm2_5Model,
    device: Device,
    eos_token_id: u32,
    model_name: String,
}

impl<'a> Lfm2_5GenerateModel<'a> {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec.paths.weight_dir.as_deref().ok_or_else(|| {
                    anyhow!("weight_path is required for lfm2.5-350m safetensors")
                })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => Err(anyhow!("lfm2.5-350m gguf runtime is not implemented yet")),
            ArtifactKind::Onnx => Err(anyhow!("lfm2.5-350m onnx runtime is not implemented yet")),
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
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
            model,
            device,
            eos_token_id: cfg.eos_token_id,
            model_name: "lfm2.5-350m".to_string(),
        })
    }
}

impl<'a> GenerateModel for Lfm2_5GenerateModel<'a> {
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
            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        let num_token = generate.len() as u32;
        let res = decode_tokens_for_completion(&self.tokenizer, &generate, self.eos_token_id)?;
        self.model.clear_cache();
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let model_name = self.model_name.clone();
        let tokenizer = &self.tokenizer;
        let device = self.device.clone();
        let eos_token_id = self.eos_token_id;
        let model = &mut self.model;

        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut stream_state = Lfm2_5StreamState::default();
            for _ in 0..sample_len {
                let logits = model.forward(&input_ids, seqlen_offset)?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
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
            model.clear_cache();
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
