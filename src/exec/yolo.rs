use std::{
    fs::OpenOptions,
    io::{Cursor, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use anyhow::{Result, anyhow};
use image::ImageFormat;

use crate::{
    exec::ExecModel,
    models::{
        ArtifactKind, LoadSpec, ModelPaths, WhichModel,
        yolo::{
            generate::YoloGenerateModel,
            model::{YoloPredictOptions, YoloResults},
        },
    },
};

pub struct YoloExec;

#[derive(Debug, Clone, Default)]
pub struct YoloExecOptions {
    pub show: bool,
    pub stream: bool,
    pub workers: Option<usize>,
    pub batch_size: Option<usize>,
    pub max_frames: Option<usize>,
    pub frame_stride: Option<usize>,
}

impl YoloExec {
    pub fn run_with_spec(input: &[String], output: Option<&str>, spec: &LoadSpec) -> Result<()> {
        Self::run_with_spec_and_options(input, output, spec, &YoloExecOptions::default())
    }

    pub fn run_with_spec_and_options(
        input: &[String],
        output: Option<&str>,
        spec: &LoadSpec,
        options: &YoloExecOptions,
    ) -> Result<()> {
        let source = input
            .first()
            .ok_or_else(|| anyhow!("yolo run requires one source input"))?;
        let load_start = Instant::now();
        let mut model = YoloGenerateModel::init_from_spec(spec)?;
        println!("Time elapsed in load model is: {:?}", load_start.elapsed());

        if options.stream {
            return Self::run_stream_mode(source, output, options, &mut model);
        }

        let predict_options = YoloPredictOptions {
            stream: options.stream,
            workers: options.workers.unwrap_or(1).max(1),
            batch_size: options.batch_size.unwrap_or(16).max(1),
            max_frames: options.max_frames,
            frame_stride: options.frame_stride.unwrap_or(1).max(1),
            stop_flag: None,
        };

        let predict_start = Instant::now();
        let results = model.predict_with_options(source, &predict_options)?;
        println!("Time elapsed in predict is: {:?}", predict_start.elapsed());

        let output_json = YoloGenerateModel::results_to_json(&results)?;
        println!("{}", output_json);

        if let Some(output) = output {
            persist_outputs(&results, &output_json, output)?;
        }

        if options.show {
            show_results_window(&results)?;
        }

        Ok(())
    }

    fn run_stream_mode(
        source: &str,
        output: Option<&str>,
        options: &YoloExecOptions,
        model: &mut YoloGenerateModel,
    ) -> Result<()> {
        let stop_flag = install_stream_interrupt_handler()?;
        stop_flag.store(false, Ordering::SeqCst);

        let predict_options = YoloPredictOptions {
            stream: true,
            workers: options.workers.unwrap_or(1).max(1),
            batch_size: options.batch_size.unwrap_or(1).max(1),
            max_frames: options.max_frames,
            frame_stride: options.frame_stride.unwrap_or(1).max(1),
            stop_flag: Some(stop_flag.clone()),
        };

        let mut viewer = if options.show {
            Some(FrameViewer::spawn()?)
        } else {
            None
        };

        let predict_start = Instant::now();
        let mut frame_count = 0_usize;
        model.predict_stream_with_options(source, &predict_options, |result| {
            frame_count += 1;
            let frame_json = YoloGenerateModel::results_to_json(std::slice::from_ref(result))?;
            println!("{}", frame_json);

            if let Some(output_path) = output {
                persist_stream_result(result, &frame_json, output_path, frame_count)?;
            }

            if let Some(viewer) = viewer.as_mut() {
                viewer.write_result(result)?;
            }

            if stop_flag.load(Ordering::Relaxed) {
                return Ok(false);
            }
            Ok(true)
        })?;
        println!(
            "Time elapsed in stream predict is: {:?}",
            predict_start.elapsed()
        );
        println!("Processed stream frames: {}", frame_count);

        if let Some(viewer) = viewer.take() {
            viewer.finish()?;
        }

        Ok(())
    }
}

impl ExecModel for YoloExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec {
            model: WhichModel::Yolo11Detect,
            artifact: ArtifactKind::Onnx,
            paths: ModelPaths {
                onnx_path: Some(weight_path.to_string()),
                ..Default::default()
            },
        };
        Self::run_with_spec(input, output, &spec)
    }
}

fn persist_outputs(
    results: &[crate::models::yolo::model::YoloResults],
    output_json: &str,
    output: &str,
) -> Result<()> {
    let path = Path::new(output);
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        std::fs::write(path, output_json)?;
        println!("Output saved to: {}", path.display());
        return Ok(());
    }

    let output_dir = if path.extension().is_some() {
        path.parent().unwrap_or_else(|| Path::new("."))
    } else {
        path
    };
    std::fs::create_dir_all(output_dir)?;

    let json_path = if path.extension().is_some() {
        path.with_extension("json")
    } else {
        output_dir.join("results.json")
    };
    std::fs::write(&json_path, output_json)?;
    println!("Output saved to: {}", json_path.display());

    for (index, result) in results.iter().enumerate() {
        if let Ok(annotated) = result.plot() {
            let stem = Path::new(&result.path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("result");
            let image_path = output_dir.join(format!("{}_{}.png", stem, index));
            annotated.save(&image_path)?;
            println!("Annotated image saved to: {}", image_path.display());
        }
        let txt_path = output_dir.join(format!("result_{}.txt", index));
        result.save_txt(&txt_path.to_string_lossy())?;
    }

    Ok(())
}

fn show_results_window(results: &[YoloResults]) -> Result<()> {
    let mut viewer = FrameViewer::spawn()?;
    for result in results {
        viewer.write_result(result)?;
    }
    viewer.finish()
}

fn persist_stream_result(
    result: &YoloResults,
    frame_json: &str,
    output: &str,
    frame_index: usize,
) -> Result<()> {
    let path = Path::new(output);
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("jsonl"))
    {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", frame_json)?;
        return Ok(());
    }

    let output_dir = if path.extension().is_some() {
        path.parent().unwrap_or_else(|| Path::new("."))
    } else {
        path
    };
    std::fs::create_dir_all(output_dir)?;

    let mut jsonl = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("results.jsonl"))?;
    writeln!(jsonl, "{}", frame_json)?;

    if let Ok(annotated) = result.plot() {
        let image_path = output_dir.join(format!("frame_{:06}.png", frame_index));
        annotated.save(&image_path)?;
    }
    let txt_path = output_dir.join(format!("frame_{:06}.txt", frame_index));
    result.save_txt(&txt_path.to_string_lossy())?;
    Ok(())
}

fn install_stream_interrupt_handler() -> Result<Arc<AtomicBool>> {
    static STREAM_STOP_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    if let Some(flag) = STREAM_STOP_FLAG.get() {
        return Ok(flag.clone());
    }

    let flag = Arc::new(AtomicBool::new(false));
    let handler_flag = flag.clone();
    if let Err(err) = ctrlc::set_handler(move || {
        handler_flag.store(true, Ordering::SeqCst);
    }) {
        return Err(anyhow!(
            "failed to install Ctrl+C stream stop handler: {err}"
        ));
    }
    let _ = STREAM_STOP_FLAG.set(flag.clone());
    Ok(flag)
}

struct FrameViewer {
    child: Child,
    stdin: ChildStdin,
}

impl FrameViewer {
    fn spawn() -> Result<Self> {
        let mut child = Command::new("ffplay")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-framerate",
                "24",
                "-vcodec",
                "png",
                "-i",
                "-",
            ])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|err| anyhow!("failed to launch ffplay for --show mode: {err}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("failed to open ffplay stdin for --show mode"))?;
        Ok(Self { child, stdin })
    }

    fn write_result(&mut self, result: &YoloResults) -> Result<()> {
        let plotted = result.plot()?;
        let mut frame_png = Vec::new();
        image::DynamicImage::ImageRgba8(plotted)
            .write_to(&mut Cursor::new(&mut frame_png), ImageFormat::Png)?;
        self.stdin.write_all(&frame_png)?;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.stdin.flush()?;
        drop(self.stdin);
        let status = self.child.wait()?;
        if !status.success() {
            return Err(anyhow!("ffplay exited with non-zero status in --show mode"));
        }
        Ok(())
    }
}
