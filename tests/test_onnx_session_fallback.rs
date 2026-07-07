#[cfg(feature = "onnx-runtime")]
#[path = "../src/models/common/onnx.rs"]
mod onnx_impl;

#[cfg(feature = "onnx-runtime")]
mod tests {
    use super::onnx_impl::try_create_session_with_optimization_fallbacks;
    use anyhow::anyhow;
    use ort::session::builder::GraphOptimizationLevel;

    #[test]
    fn onnx_optimization_fallback_retries_from_all_to_level1() {
        let mut attempts = Vec::new();

        let result = try_create_session_with_optimization_fallbacks(|level| {
            attempts.push(level);
            match level {
                GraphOptimizationLevel::All => Err(anyhow!("optimizer crash")),
                GraphOptimizationLevel::Level1 => Ok("ok"),
                _ => Err(anyhow!("unexpected fallback")),
            }
        })
        .expect("level1 fallback should recover");

        assert_eq!(result, "ok");
        assert_eq!(
            attempts,
            vec![GraphOptimizationLevel::All, GraphOptimizationLevel::Level1]
        );
    }
}
