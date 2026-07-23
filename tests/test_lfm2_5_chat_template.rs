use aha::chat_template::ChatTemplate;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn lfm2_5_230m_template_renders_generation_block() -> Result<()> {
    let template = include_str!("fixtures/lfm2_5_230m_chat_template.jinja");
    let request: ChatCompletionParameters = serde_json::from_value(serde_json::json!({
        "model": "lfm2.5-230m",
        "messages": [
            { "role": "user", "content": "What is Rust?" }
        ]
    }))?;

    let chat_template = ChatTemplate::str_init(template)?;
    let prompt = chat_template.apply_chat_template(&request)?;

    assert_eq!(
        prompt,
        "<|im_start|>user\nWhat is Rust?<|im_end|>\n<|im_start|>assistant\n"
    );
    Ok(())
}
