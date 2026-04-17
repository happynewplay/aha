use std::path::Path;
#[cfg(feature = "onnx-runtime")]
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "onnx-runtime")]
use aha::exec::yolo::YoloExec;
#[cfg(feature = "onnx-runtime")]
use aha::models::yolo::model::YoloPredictOptions;
use aha::models::{
    ArtifactKind, LoadSpec, ModelPaths, WhichModel,
    yolo::{
        generate::YoloGenerateModel,
        model::{YoloBox, YoloResults, YoloSpeed},
    },
};
use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, Rgba, RgbaImage};

#[cfg(feature = "onnx-runtime")]
use aha::models::common::onnx::ensure_ort_dylib_path;

const DEFAULT_YOLO_ONNX_PATH: &str = r"D:\model_download\yolo26m-ONNX\onnx\model_q4f16.onnx";
#[cfg(feature = "onnx-runtime")]
const TEST_IMAGE_REL_PATH: &str = "assets/img/gougou.jpg";
#[cfg(feature = "onnx-runtime")]
const TEST_VIDEO_REL_PATH: &str = "assets/video/video_test.mp4";

fn yolo_onnx_path() -> String {
    std::env::var("AHA_YOLO_ONNX_PATH").unwrap_or_else(|_| DEFAULT_YOLO_ONNX_PATH.to_string())
}

fn require_existing_file(path: &str) -> Result<()> {
    let file = Path::new(path);
    if !file.exists() {
        return Err(anyhow!("file not found: {}", path));
    }
    if !file.is_file() {
        return Err(anyhow!("path is not a file: {}", path));
    }
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
fn absolute_repo_path(relative: &str) -> Result<PathBuf> {
    Ok(std::env::current_dir()?.join(relative))
}

fn yolo_spec_from(path: String) -> LoadSpec {
    LoadSpec {
        model: WhichModel::Yolo11Detect,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(path),
            ..Default::default()
        },
    }
}

#[cfg(feature = "onnx-runtime")]
fn unique_temp_dir(prefix: &str) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("system clock error: {err}"))?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("aha_{prefix}_{nanos}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(feature = "onnx-runtime")]
fn first_png_in_dir(dir: &Path) -> Result<Option<PathBuf>> {
    let mut pngs = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        })
        .collect::<Vec<_>>();
    pngs.sort();
    Ok(pngs.into_iter().next())
}

#[cfg(feature = "onnx-runtime")]
fn init_yolo_model_or_skip() -> Result<Option<YoloGenerateModel>> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip yolo onnx tests: {err}");
        return Ok(None);
    }

    let onnx_path = yolo_onnx_path();
    if !Path::new(&onnx_path).exists() {
        println!(
            "skip yolo onnx tests: model file not found at {}, set AHA_YOLO_ONNX_PATH to run",
            onnx_path
        );
        return Ok(None);
    }

    let spec = yolo_spec_from(onnx_path);
    Ok(Some(YoloGenerateModel::init_from_spec(&spec)?))
}

#[test]
fn yolo_onnx_file_can_load() -> Result<()> {
    let onnx_path = yolo_onnx_path();
    require_existing_file(&onnx_path)?;
    let metadata = std::fs::metadata(&onnx_path)
        .with_context(|| format!("failed to read metadata for {}", onnx_path))?;
    if metadata.len() == 0 {
        return Err(anyhow!("onnx file is empty: {}", onnx_path));
    }
    if Path::new(&onnx_path)
        .extension()
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("onnx"))
    {
        return Err(anyhow!("onnx path is not an .onnx file: {}", onnx_path));
    }
    Ok(())
}

#[test]
fn yolo_load_spec_accepts_onnx() {
    let spec = yolo_spec_from(yolo_onnx_path());
    spec.validate().expect("yolo should accept onnx artifact");
}

#[test]
fn yolo_model_metadata_is_registered() {
    assert_eq!(WhichModel::Yolo11Detect.openai_model_id(), "yolo11-detect");
    assert_eq!(WhichModel::Yolo11Detect.model_type(), "image");
}

#[test]
fn yolo_results_are_json_serializable() {
    let result = YoloResults {
        boxes: vec![YoloBox {
            xyxy: [1.0, 2.0, 3.0, 4.0],
            xywh: [2.0, 3.0, 2.0, 2.0],
            conf: 0.9,
            cls: 0,
            label: "person".to_string(),
        }],
        path: "image.jpg".to_string(),
        names: vec!["person".to_string()],
        speed: YoloSpeed {
            preprocess_ms: 1.0,
            inference_ms: 2.0,
            postprocess_ms: 3.0,
        },
        width: 640,
        height: 480,
        orig_img: None,
    };

    let json =
        YoloGenerateModel::results_to_json(&[result]).expect("yolo results should serialize");
    assert!(json.contains("person"));
    assert!(json.contains("image.jpg"));
}

#[test]
fn yolo_results_plot_draws_box() -> Result<()> {
    let base = DynamicImage::ImageRgba8(RgbaImage::from_pixel(32, 32, Rgba([0, 0, 0, 255])));
    let result = YoloResults {
        boxes: vec![YoloBox {
            xyxy: [4.0, 4.0, 20.0, 20.0],
            xywh: [12.0, 12.0, 16.0, 16.0],
            conf: 0.88,
            cls: 0,
            label: "person".to_string(),
        }],
        path: "synthetic.png".to_string(),
        names: vec!["person".to_string()],
        speed: YoloSpeed::default(),
        width: 32,
        height: 32,
        orig_img: Some(base),
    };
    let plotted = result.plot()?;
    assert_eq!(plotted.get_pixel(4, 4).0, [255, 64, 64, 255]);
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_onnx_init_from_spec_can_init() -> Result<()> {
    let Some(_model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_onnx_single_image_predict_smoke() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    let image_path = absolute_repo_path(TEST_IMAGE_REL_PATH)?;
    if !image_path.exists() {
        println!(
            "skip yolo single-image smoke test: test image not found at {}",
            image_path.display()
        );
        return Ok(());
    }
    let results = model.predict(&image_path.to_string_lossy())?;
    assert_eq!(
        results.len(),
        1,
        "single image input should return one result"
    );
    assert!(results[0].width > 0 && results[0].height > 0);
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_onnx_batch_directory_predict_smoke() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    let source_image = absolute_repo_path(TEST_IMAGE_REL_PATH)?;
    if !source_image.exists() {
        println!(
            "skip yolo batch smoke test: source image not found at {}",
            source_image.display()
        );
        return Ok(());
    }
    let batch_dir = unique_temp_dir("yolo_batch_input")?;
    let img1 = batch_dir.join("img1.jpg");
    let img2 = batch_dir.join("img2.jpg");
    std::fs::copy(&source_image, &img1)?;
    std::fs::copy(&source_image, &img2)?;

    let results = model.predict(&batch_dir.to_string_lossy())?;
    assert_eq!(
        results.len(),
        2,
        "batch directory should produce two results"
    );
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_onnx_batch_directory_predict_with_workers_smoke() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    let source_image = absolute_repo_path(TEST_IMAGE_REL_PATH)?;
    if !source_image.exists() {
        println!(
            "skip yolo batch+workers smoke test: source image not found at {}",
            source_image.display()
        );
        return Ok(());
    }
    let batch_dir = unique_temp_dir("yolo_batch_workers_input")?;
    for idx in 0..4 {
        let dst = batch_dir.join(format!("img_{idx}.jpg"));
        std::fs::copy(&source_image, dst)?;
    }

    let options = YoloPredictOptions {
        workers: 2,
        batch_size: 2,
        ..Default::default()
    };
    let results = model.predict_with_options(&batch_dir.to_string_lossy(), &options)?;
    assert_eq!(
        results.len(),
        4,
        "batch directory should produce four results"
    );
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_exec_persist_outputs_writes_json_png_and_txt() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip yolo exec output test: {err}");
        return Ok(());
    }

    let onnx_path = yolo_onnx_path();
    if !Path::new(&onnx_path).exists() {
        println!(
            "skip yolo exec output test: model file not found at {}",
            onnx_path
        );
        return Ok(());
    }

    let image_path = absolute_repo_path(TEST_IMAGE_REL_PATH)?;
    if !image_path.exists() {
        println!(
            "skip yolo exec output test: image not found at {}",
            image_path.display()
        );
        return Ok(());
    }

    let output_dir = unique_temp_dir("yolo_exec_output")?;
    let spec = yolo_spec_from(onnx_path);
    let input = vec![image_path.to_string_lossy().to_string()];
    YoloExec::run_with_spec(&input, Some(&output_dir.to_string_lossy()), &spec)?;

    let json_path = output_dir.join("results.json");
    let txt_path = output_dir.join("result_0.txt");
    if !json_path.exists() {
        return Err(anyhow!(
            "missing expected output json: {}",
            json_path.display()
        ));
    }
    if !txt_path.exists() {
        return Err(anyhow!(
            "missing expected output txt: {}",
            txt_path.display()
        ));
    }
    let Some(png_path) = first_png_in_dir(&output_dir)? else {
        return Err(anyhow!(
            "missing expected annotated png in {}",
            output_dir.display()
        ));
    };
    if !png_path.exists() {
        return Err(anyhow!("annotated png not found: {}", png_path.display()));
    }

    let json_value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path)?)?;
    let array_len = json_value.as_array().map_or(0, |items| items.len());
    assert_eq!(array_len, 1, "expected one json result entry");
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_stream_sources_require_stream_mode() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };

    let rtsp_err = model
        .predict("rtsp://127.0.0.1:8554/live")
        .expect_err("rtsp should require stream mode");
    assert!(
        rtsp_err.to_string().contains("requires stream mode"),
        "unexpected rtsp error: {rtsp_err}"
    );

    let webcam_err = model
        .predict("0")
        .expect_err("webcam index should still be rejected");
    assert!(
        webcam_err
            .to_string()
            .contains("webcam numeric index input requires stream mode"),
        "unexpected webcam error: {webcam_err}"
    );

    let video_path = absolute_repo_path(TEST_VIDEO_REL_PATH)?;
    if !video_path.exists() {
        println!(
            "skip yolo video rejection assertion: video file not found at {}",
            video_path.display()
        );
        return Ok(());
    }
    let video_err = model
        .predict(&video_path.to_string_lossy())
        .expect_err("video source should require stream mode");
    assert!(
        video_err
            .to_string()
            .contains("video sources require stream mode"),
        "unexpected video error: {video_err}"
    );

    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn yolo_unbounded_stream_vec_prediction_is_rejected() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    let options = YoloPredictOptions {
        stream: true,
        max_frames: None,
        ..Default::default()
    };
    let err = model
        .predict_with_options("rtsp://127.0.0.1:8554/live", &options)
        .expect_err("unbounded stream Vec prediction should be rejected");
    assert!(
        err.to_string().contains("requires max_frames"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[cfg(all(feature = "onnx-runtime", feature = "ffmpeg"))]
#[test]
fn yolo_stream_video_predict_smoke() -> Result<()> {
    let Some(mut model) = init_yolo_model_or_skip()? else {
        return Ok(());
    };
    let video_path = absolute_repo_path(TEST_VIDEO_REL_PATH)?;
    if !video_path.exists() {
        println!(
            "skip yolo stream-video smoke test: video file not found at {}",
            video_path.display()
        );
        return Ok(());
    }
    let options = YoloPredictOptions {
        stream: true,
        max_frames: Some(3),
        frame_stride: 8,
        ..Default::default()
    };
    let results = model.predict_with_options(&video_path.to_string_lossy(), &options)?;
    assert!(
        !results.is_empty(),
        "stream mode should decode at least one frame"
    );
    assert!(
        results.len() <= 3,
        "stream mode should honor max_frames, got {}",
        results.len()
    );
    Ok(())
}
