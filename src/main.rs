use std::sync::atomic::{AtomicBool, Ordering};
use std::{net::IpAddr, str::FromStr, sync::Arc};

use aha::{
    models::{
        ArtifactKind, EmbeddingOptions, LISTED_MODELS, LoadSpec, ModelPaths, WhichModel,
        default_artifact,
    },
    process::{cleanup_pid_file, create_pid_file},
    utils::{download_model, get_default_save_dir},
};
use anyhow::anyhow;
use clap::{Args, Parser, Subcommand, ValueEnum};
use rocket::{
    Config,
    data::{ByteUnit, Limits},
    routes,
};
use serde::Serialize;

use crate::api::{init, set_server_port};
mod api;

#[derive(Parser, Debug)]
#[command(name = "aha")]
#[command(version, about, long_about = None)]
struct Cli {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: Option<String>,

    /// Service listen port
    #[arg(short, long)]
    port: Option<u16>,

    /// Model type (required for backward compatibility)
    #[arg(short, long)]
    model: Option<WhichModel>,

    /// Local model weight path
    #[arg(long)]
    weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    mmproj_path: Option<String>,

    /// Local ONNX model path
    #[arg(long)]
    onnx_path: Option<String>,

    /// Tokenizer/config directory for GGUF/ONNX artifacts
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Model artifact format
    #[arg(long)]
    artifact_format: Option<ArtifactArg>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download model and start service (default)
    Cli(CliArgs),
    /// Start service only (--weight-path is optional, defaults to ~/.aha/{model_id})
    Serv(ServArgs),
    /// List all running aha services
    Ps(ServListArgs),
    /// Delete a downloaded model from the default location (~/.aha/{model_id})
    Delete(DeleteArgs),
    /// Download model only
    Download(DownloadArgs),
    /// Run model inference directly
    Run(RunArgs),
    /// List all supported models
    List(ListArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ArtifactArg {
    Auto,
    Safetensors,
    Gguf,
    Onnx,
}

impl From<ArtifactArg> for ArtifactKind {
    fn from(value: ArtifactArg) -> Self {
        match value {
            ArtifactArg::Auto => ArtifactKind::Auto,
            ArtifactArg::Safetensors => ArtifactKind::Safetensors,
            ArtifactArg::Gguf => ArtifactKind::Gguf,
            ArtifactArg::Onnx => ArtifactKind::Onnx,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RunEmbeddingPromptArg {
    Query,
    Document,
}

impl From<RunEmbeddingPromptArg> for aha::models::EmbeddingPromptName {
    fn from(value: RunEmbeddingPromptArg) -> Self {
        match value {
            RunEmbeddingPromptArg::Query => Self::Query,
            RunEmbeddingPromptArg::Document => Self::Document,
        }
    }
}

/// Common/shared arguments for server operations
#[derive(Args, Debug)]
struct CommonArgs {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: String,

    /// Service listen port
    #[arg(short, long, default_value_t = 10100)]
    port: u16,

    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Allow remote shutdown requests (default: local only, use with caution)
    #[arg(long)]
    allow_remote_shutdown: bool,
}

/// Arguments for the 'cli' subcommand (download + serve)
#[derive(Args, Debug)]
struct CliArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Local model weight path (skip download if provided)
    #[arg(long)]
    weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    mmproj_path: Option<String>,

    /// Local ONNX model path
    #[arg(long)]
    onnx_path: Option<String>,

    /// Tokenizer/config directory for GGUF/ONNX artifacts
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Model artifact format
    #[arg(long)]
    artifact_format: Option<ArtifactArg>,
}

/// Arguments for the 'serv start' subcommand
#[derive(Args, Debug)]
struct ServArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    weight_path: Option<String>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    mmproj_path: Option<String>,

    /// Local ONNX model path
    #[arg(long)]
    onnx_path: Option<String>,

    /// Tokenizer/config directory for GGUF/ONNX artifacts
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Model artifact format
    #[arg(long)]
    artifact_format: Option<ArtifactArg>,
}

/// Arguments for the 'serv list' subcommand
#[derive(Args, Debug)]
struct ServListArgs {
    /// Compact output format
    #[arg(short, long)]
    compact: bool,
}

/// Arguments for the 'download' subcommand (download only)
#[derive(Args, Debug)]
struct DownloadArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Model download save directory
    #[arg(short, long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,
}

/// Arguments for the 'run' subcommand (direct inference)
#[derive(Args, Debug)]
struct RunArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Input text or file path
    #[arg(short, long, num_args = 1..=2, value_delimiter = ' ', required_unless_present = "request_json")]
    input: Vec<String>,

    /// Full chat completion request JSON file
    #[arg(long)]
    request_json: Option<String>,

    /// Output file path (optional)
    #[arg(short, long)]
    output: Option<String>,

    /// Maximum number of tokens to generate
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Embedding prompt name for embedding-capable models
    #[arg(long = "prompt-name")]
    prompt_name: Option<RunEmbeddingPromptArg>,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    weight_path: Option<String>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    mmproj_path: Option<String>,

    /// Local ONNX model path
    #[arg(long)]
    onnx_path: Option<String>,

    /// Tokenizer/config directory for GGUF/ONNX artifacts
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Model artifact format
    #[arg(long)]
    artifact_format: Option<ArtifactArg>,

    /// Show prediction results in a live window (requires ffplay in PATH)
    #[arg(long, default_value_t = false)]
    show: bool,

    /// Enable stream mode for video/rtsp inputs
    #[arg(long, default_value_t = false)]
    stream: bool,

    /// Worker threads for batch inference acceleration (YOLO only)
    #[arg(long)]
    workers: Option<usize>,

    /// Chunk size for distributing work across parallel workers (YOLO only, default: 16). Does NOT control the ONNX batch dimension (always 1).
    #[arg(long)]
    chunk_size: Option<usize>,

    /// Limit frames processed in stream mode (YOLO only)
    #[arg(long)]
    max_frames: Option<usize>,

    /// Keep one frame every N frames in stream mode (YOLO only)
    #[arg(long)]
    frame_stride: Option<usize>,

    /// Log realtime stream FPS/latency every N processed frames (YOLO only)
    #[arg(long)]
    stream_log_every: Option<usize>,

    /// Confidence threshold for detection filtering (YOLO only, default: 0.25)
    #[arg(long)]
    conf: Option<f32>,

    /// IoU threshold for NMS (YOLO only, default: 0.45)
    #[arg(long)]
    iou: Option<f32>,

    /// Maximum number of detections per image (YOLO only, default: 300)
    #[arg(long)]
    max_detections: Option<usize>,

    /// Force YOLO task kind instead of auto-detection: detect|segment|pose|classify|obb
    #[arg(long)]
    task: Option<aha::models::yolo::config::YoloTaskKind>,

    /// Enable class-agnostic NMS: suppress overlapping boxes across classes (YOLO only)
    #[arg(long, default_value_t = false)]
    nms_class_agnostic: bool,

    /// Keypoint confidence threshold for pose estimation (YOLO only, default: 0.1)
    #[arg(long)]
    keypoint_conf: Option<f32>,

    /// Keep original images in results for plotting (YOLO only, default: true)
    #[arg(long)]
    keep_images: Option<bool>,

    /// Custom class names for YOLO, comma-separated (e.g. "face,person"). Overrides default COCO 80-class names.
    #[arg(long, value_delimiter = ',')]
    class_names: Option<Vec<String>>,
}

/// Arguments for the 'delete' subcommand (delete model from default location)
#[derive(Args, Debug)]
struct DeleteArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,
}

/// Arguments for the 'list' subcommand (list all supported models)
#[derive(Args, Debug)]
struct ListArgs {
    /// Output models in JSON format (includes name, model_id, and type fields)
    #[arg(short, long)]
    json: bool,
}

/// Get the default weight path for a given model
/// Returns ~/.aha/{model_id} e.g., ~/.aha/OpenBMB/VoxCPM1.5
fn get_default_weight_path(model: WhichModel) -> String {
    let model_id = model.model_id();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    format!("{}/{}", save_dir, model_id)
}

/// Check if a model is downloaded by verifying the model directory exists
/// Returns true if ~/.aha/{model_id} directory exists, false otherwise
fn is_model_downloaded(model: WhichModel) -> bool {
    let model_id = model.model_id();
    let save_dir = match get_default_save_dir() {
        Some(dir) => dir,
        None => return false,
    };
    let model_path = format!("{}/{}", save_dir, model_id);
    std::path::Path::new(&model_path).exists()
}

/// Model information for JSON output
#[derive(Serialize)]
struct ModelInfo {
    name: String,
    model_id: String,
    #[serde(rename = "type")]
    model_type: String,
    downloaded: bool,
}

/// List all supported models
fn run_list(args: ListArgs) -> anyhow::Result<()> {
    if args.json {
        // JSON output
        let model_infos: Vec<ModelInfo> = LISTED_MODELS
            .iter()
            .map(|model| {
                let possible_value = model.to_possible_value().unwrap();
                ModelInfo {
                    name: possible_value.get_name().to_string(),
                    model_id: model.model_id().to_string(),
                    model_type: model.model_type().to_string(),
                    downloaded: is_model_downloaded(*model),
                }
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&model_infos)?);
    } else {
        // Table output (default)
        println!("Available models:");
        println!();
        println!(
            "{:<30} {:<40} {:<10}",
            "Model Name", "ModelScope ID", "Download"
        );
        println!("{}", "-".repeat(80));
        for model in LISTED_MODELS {
            let possible_value = model.to_possible_value().unwrap();
            let name = possible_value.get_name();
            let id = model.model_id();
            let download_status = if is_model_downloaded(*model) {
                "  ✔"
            } else {
                ""
            };
            println!("{:<30} {:<40} {:<10}", name, id, download_status);
        }
    }

    Ok(())
}

fn resolve_artifact_kind(
    model: WhichModel,
    artifact: Option<ArtifactArg>,
    gguf_path: Option<&str>,
    onnx_path: Option<&str>,
) -> ArtifactKind {
    artifact.map(Into::into).unwrap_or_else(|| {
        if gguf_path.is_some() {
            ArtifactKind::Gguf
        } else if onnx_path.is_some() {
            ArtifactKind::Onnx
        } else {
            default_artifact(model)
        }
    })
}

async fn resolve_load_spec_for_server(
    model: WhichModel,
    weight_path: Option<String>,
    save_dir: Option<String>,
    download_retries: Option<u32>,
    gguf_path: Option<String>,
    mmproj_path: Option<String>,
    onnx_path: Option<String>,
    tokenizer_dir: Option<String>,
    artifact: Option<ArtifactArg>,
    allow_download: bool,
) -> anyhow::Result<LoadSpec> {
    let artifact =
        resolve_artifact_kind(model, artifact, gguf_path.as_deref(), onnx_path.as_deref());
    let weight_dir = match artifact {
        ArtifactKind::Safetensors => match weight_path {
            Some(path) => Some(path),
            None if allow_download && model.is_download_managed() => {
                let model_id = model.model_id();
                let save_dir = match save_dir {
                    Some(dir) => dir,
                    None => get_default_save_dir().expect("Failed to get home directory"),
                };
                let max_retries = download_retries.unwrap_or(3);
                download_model(model_id, &save_dir, max_retries).await?;
                Some(save_dir + "/" + model_id)
            }
            None => Some(get_default_weight_path(model)),
        },
        ArtifactKind::Gguf | ArtifactKind::Onnx | ArtifactKind::Auto => None,
    };

    let spec = LoadSpec {
        model,
        artifact,
        paths: ModelPaths {
            weight_dir,
            gguf_path,
            mmproj_path,
            onnx_path,
            tokenizer_dir,
        },
    };
    spec.validate()?;
    Ok(spec)
}

fn resolve_load_spec_for_run(args: &RunArgs) -> anyhow::Result<LoadSpec> {
    let artifact = resolve_artifact_kind(
        args.model,
        args.artifact_format,
        args.gguf_path.as_deref(),
        args.onnx_path.as_deref(),
    );
    let weight_dir = match artifact {
        ArtifactKind::Safetensors => Some(
            args.weight_path
                .clone()
                .unwrap_or_else(|| get_default_weight_path(args.model)),
        ),
        ArtifactKind::Gguf | ArtifactKind::Onnx | ArtifactKind::Auto => None,
    };
    let spec = LoadSpec {
        model: args.model,
        artifact,
        paths: ModelPaths {
            weight_dir,
            gguf_path: args.gguf_path.clone(),
            mmproj_path: args.mmproj_path.clone(),
            onnx_path: args.onnx_path.clone(),
            tokenizer_dir: args.tokenizer_dir.clone(),
        },
    };
    spec.validate()?;
    Ok(spec)
}

fn run_target_model_with_spec(args: &RunArgs, spec: &LoadSpec) -> anyhow::Result<bool> {
    if args.request_json.is_some()
        && args.model != WhichModel::MiniCPM5_1B
        && args.model != WhichModel::LFM2_5_350M
    {
        return Err(anyhow!(
            "--request-json is only supported for minicpm5-1b and lfm2.5-350m"
        ));
    }
    match args.model {
        WhichModel::AllMiniLML6V2 => {
            use aha::exec::all_minilm_l6_v2::AllMiniLML6V2Exec;
            AllMiniLML6V2Exec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::MxbaiEmbedXsmallV1 => {
            use aha::exec::mxbai_embed_xsmall_v1::MxbaiEmbedXsmallV1Exec;
            MxbaiEmbedXsmallV1Exec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::MiniCPM5_1B => {
            use aha::exec::minicpm5::MiniCPM5Exec;
            MiniCPM5Exec::run_with_spec(
                &args.input,
                args.output.as_deref(),
                spec,
                args.request_json.as_deref(),
            )?;
            Ok(true)
        }
        WhichModel::Qwen3_0_6B => {
            use aha::exec::qwen3::Qwen3Exec;
            Qwen3Exec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => {
            use aha::exec::qwen3_embedding::Qwen3EmbeddingExec;
            Qwen3EmbeddingExec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => {
            use aha::exec::qwen3_reranker::Qwen3RerankerExec;
            Qwen3RerankerExec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::Qwen3_5_0_8B
        | WhichModel::Qwen3_5_2B
        | WhichModel::Qwen3_5_4B
        | WhichModel::Qwen3_5_9B
        | WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf => {
            use aha::exec::qwen3_5::Qwen3_5Exec;
            Qwen3_5Exec::run_with_spec(&args.input, args.output.as_deref(), spec)?;
            Ok(true)
        }
        WhichModel::LFM2_5_350M => {
            use aha::exec::lfm2_5::Lfm2_5Exec;
            Lfm2_5Exec::run_with_spec(
                &args.input,
                args.output.as_deref(),
                spec,
                args.request_json.as_deref(),
            )?;
            Ok(true)
        }
        WhichModel::LFM2_5Embedding350M => {
            use aha::exec::lfm2_5_embedding::Lfm2_5EmbeddingExec;
            let options = EmbeddingOptions {
                prompt_name: args.prompt_name.map(Into::into).unwrap_or_default(),
            };
            Lfm2_5EmbeddingExec::run_with_spec(&args.input, args.output.as_deref(), spec, options)?;
            Ok(true)
        }
        WhichModel::GlmOCR => {
            use aha::exec::glm_ocr::GlmOcrExec;
            GlmOcrExec::run_with_spec(&args.input, args.output.as_deref(), spec, args.max_tokens)?;
            Ok(true)
        }
        WhichModel::Yolo11Detect => {
            use aha::exec::yolo::YoloExec;
            use aha::exec::yolo::YoloExecOptions;
            let options = YoloExecOptions {
                show: args.show,
                stream: args.stream,
                workers: args.workers,
                batch_size: args.chunk_size,
                max_frames: args.max_frames,
                frame_stride: args.frame_stride,
                stream_log_every: args.stream_log_every,
                conf_threshold: args.conf,
                iou_threshold: args.iou,
                max_detections: args.max_detections,
                task_kind: args.task,
                nms_class_agnostic: if args.nms_class_agnostic {
                    Some(true)
                } else {
                    None
                },
                keypoint_confidence_threshold: args.keypoint_conf,
                keep_images: args.keep_images,
                class_names: args.class_names.clone(),
            };
            YoloExec::run_with_spec_and_options(
                &args.input,
                args.output.as_deref(),
                spec,
                &options,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Run the 'cli' subcommand: download model (if needed) and start service
async fn run_cli(args: CliArgs) -> anyhow::Result<()> {
    let CliArgs {
        common,
        weight_path,
        save_dir,
        download_retries,
        gguf_path,
        mmproj_path,
        onnx_path,
        tokenizer_dir,
        artifact_format,
    } = args;
    let spec = resolve_load_spec_for_server(
        common.model,
        weight_path,
        save_dir,
        download_retries,
        gguf_path,
        mmproj_path,
        onnx_path,
        tokenizer_dir,
        artifact_format,
        true,
    )
    .await?;

    init(spec)?;
    start_http_server(common.address, common.port, common.allow_remote_shutdown).await?;

    Ok(())
}

/// Run the 'serv' subcommand: start service only (no download)
async fn run_serv(args: ServArgs) -> anyhow::Result<()> {
    let ServArgs {
        common,
        weight_path,
        gguf_path,
        mmproj_path,
        onnx_path,
        tokenizer_dir,
        artifact_format,
    } = args;
    let spec = resolve_load_spec_for_server(
        common.model,
        weight_path,
        None,
        None,
        gguf_path,
        mmproj_path,
        onnx_path,
        tokenizer_dir,
        artifact_format,
        false,
    )
    .await?;

    init(spec)?;
    start_http_server(common.address, common.port, common.allow_remote_shutdown).await?;

    Ok(())
}

/// Run the 'ps' subcommand: list running AHA services
fn run_ps(args: ServListArgs) -> anyhow::Result<()> {
    use aha::process::find_aha_services;

    let services = find_aha_services()?;

    if services.is_empty() {
        println!("No aha services found running.");
        return Ok(());
    }

    if args.compact {
        // Compact format: one service per line
        for svc in services {
            println!("{}", svc.service_id);
        }
    } else {
        // Table format
        println!(
            "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
            "Service ID", "PID", "Model", "Port", "Address", "Status"
        );
        println!("{}", "-".repeat(85));

        for svc in services {
            let model = svc.model.as_deref().unwrap_or("N/A");
            let status = match svc.status {
                aha::process::ServiceStatus::Running => "Running",
                aha::process::ServiceStatus::Stopping => "Stopping",
                aha::process::ServiceStatus::Unknown => "Unknown",
            };
            println!(
                "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
                svc.service_id, svc.pid, model, svc.port, svc.address, status,
            );
        }
    }

    Ok(())
}

/// Run the 'download' subcommand: download model only (no server)
async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let DownloadArgs {
        model,
        save_dir,
        download_retries,
    } = args;
    let model_id = model.model_id();
    if !model.is_download_managed() {
        return Err(anyhow!(
            "{} does not use managed model download. Please provide local artifact path directly when serving/running.",
            model.openai_model_id()
        ));
    }

    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_default_save_dir().expect("Failed to get home directory"),
    };
    let max_retries = download_retries.unwrap_or(3);

    download_model(model_id, &save_dir, max_retries).await?;

    Ok(())
}

/// Run the 'run' subcommand: direct model inference
fn run_run(args: RunArgs) -> anyhow::Result<()> {
    use aha::exec::ExecModel;

    let spec = resolve_load_spec_for_run(&args)?;
    if run_target_model_with_spec(&args, &spec)? {
        return Ok(());
    }

    let RunArgs {
        model,
        input,
        output,
        weight_path,
        ..
    } = args;

    // Use default weight path if not specified
    let weight_path = match weight_path {
        Some(path) => path,
        None => get_default_weight_path(model),
    };
    match model {
        WhichModel::AllMiniLML6V2 => {
            use aha::exec::all_minilm_l6_v2::AllMiniLML6V2Exec;
            AllMiniLML6V2Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::MiniCPM5_1B => {
            use aha::exec::minicpm5::MiniCPM5Exec;
            MiniCPM5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::MiniCPM4_0_5B => {
            use aha::exec::minicpm4::MiniCPM4Exec;
            MiniCPM4Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5vl3B => {
            use aha::exec::qwen2_5vl::Qwen2_5vlExec;
            Qwen2_5vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5vl7B => {
            use aha::exec::qwen2_5vl::Qwen2_5vlExec;
            Qwen2_5vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_0_6B => {
            use aha::exec::qwen3::Qwen3Exec;
            Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => {
            use aha::exec::qwen3_embedding::Qwen3EmbeddingExec;
            Qwen3EmbeddingExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => {
            use aha::exec::qwen3_reranker::Qwen3RerankerExec;
            Qwen3RerankerExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_0_8B => {
            use aha::exec::qwen3_5::Qwen3_5Exec;
            Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_2B => {
            use aha::exec::qwen3_5::Qwen3_5Exec;
            Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_4B => {
            use aha::exec::qwen3_5::Qwen3_5Exec;
            Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_9B => {
            use aha::exec::qwen3_5::Qwen3_5Exec;
            Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3ASR0_6B => {
            use aha::exec::qwen3_asr::Qwen3ASRExec;
            Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3ASR1_7B => {
            use aha::exec::qwen3_asr::Qwen3ASRExec;
            Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl2B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl4B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl8B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl32B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR => {
            use aha::exec::deepseek_ocr::DeepSeekORExec;
            DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR2 => {
            use aha::exec::deepseek_ocr::DeepSeekORExec;
            DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::HunyuanOCR => {
            use aha::exec::hunyuan_ocr::HunyuanORExec;
            HunyuanORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL => {
            use aha::exec::paddleocr_vl::PaddleOVLExec;
            PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL1_5 => {
            use aha::exec::paddleocr_vl::PaddleOVLExec;
            PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::RMBG2_0 => {
            use aha::exec::rmbg2_0::RMBG2_0Exec;
            RMBG2_0Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM => {
            use aha::exec::voxcpm::VoxCPMExec;
            VoxCPMExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM1_5 => {
            use aha::exec::voxcpm1_5::VoxCPM1_5Exec;
            VoxCPM1_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmASRNano2512 => {
            use aha::exec::glm_asr_nano::GlmASRNanoExec;
            GlmASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::FunASRNano2512 => {
            use aha::exec::fun_asr_nano::FunASRNanoExec;
            FunASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmOCR => {
            use aha::exec::glm_ocr::GlmOcrExec;
            GlmOcrExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf
        | WhichModel::LFM2_5_350M
        | WhichModel::LFM2_5Embedding350M
        | WhichModel::MxbaiEmbedXsmallV1
        | WhichModel::Yolo11Detect => unreachable!(
            "qwen3.5 gguf, lfm2.5, mxbai embedding, and yolo models should already be handled by run_target_model_with_spec"
        ),
    }

    Ok(())
}

/// Run the 'delete' subcommand: delete model from default location
fn run_delete(args: DeleteArgs) -> anyhow::Result<()> {
    let DeleteArgs { model } = args;
    let model_id = model.model_id();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    let model_path = format!("{}/{}", save_dir, model_id);

    let path = std::path::Path::new(&model_path);

    if !path.exists() {
        println!("Model not found: {} does not exist", model_path);
        return Ok(());
    }

    // Show model info
    println!("Model ID: {}", model_id);
    println!("Location: {}", model_path);

    // Calculate size if possible
    if let Ok(metadata) = std::fs::metadata(path)
        && metadata.is_dir()
        && let Ok(total_size) = dir_size(path)
    {
        println!("Size: {}", bytes_to_human(total_size));
    }

    // Confirm deletion
    print!("Are you sure you want to delete this model? (y/N): ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let input = input.trim().to_lowercase();
    if input != "y" && input != "yes" {
        println!("Deletion cancelled.");
        return Ok(());
    }

    // Delete the directory
    std::fs::remove_dir_all(path)?;

    println!("Model deleted successfully: {}", model_path);

    Ok(())
}

/// Calculate total size of a directory recursively
fn dir_size(path: &std::path::Path) -> anyhow::Result<u64> {
    let mut total = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                total += dir_size(&entry_path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    } else {
        total = std::fs::metadata(path)?.len();
    }
    Ok(total)
}

/// Convert bytes to human readable format
fn bytes_to_human(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Cli(args)) => run_cli(args).await,
        Some(Commands::Serv(args)) => run_serv(args).await,
        Some(Commands::Ps(args)) => run_ps(args),
        Some(Commands::Delete(args)) => run_delete(args),
        Some(Commands::Download(args)) => run_download(args).await,
        Some(Commands::Run(args)) => run_run(args),
        Some(Commands::List(args)) => run_list(args),
        None => {
            // Backward compatibility: when no subcommand is provided, use 'cli' behavior
            let model = cli.model.expect("Model is required (use -m or --model)");
            let args = CliArgs {
                common: CommonArgs {
                    address: cli.address.unwrap_or_else(|| "127.0.0.1".to_string()),
                    port: cli.port.unwrap_or(10100),
                    model,
                    allow_remote_shutdown: false,
                },
                weight_path: cli.weight_path,
                save_dir: cli.save_dir,
                download_retries: cli.download_retries,
                gguf_path: cli.gguf_path,
                mmproj_path: cli.mmproj_path,
                onnx_path: cli.onnx_path,
                tokenizer_dir: cli.tokenizer_dir,
                artifact_format: cli.artifact_format,
            };
            run_cli(args).await
        }
    }
}

pub(crate) async fn start_http_server(
    address: String,
    port: u16,
    allow_remote_shutdown: bool,
) -> anyhow::Result<()> {
    // Set server port for shutdown endpoint
    set_server_port(port, allow_remote_shutdown);

    // Create PID file for service tracking
    let pid = std::process::id();
    create_pid_file(pid, port)?;

    // Set up shutdown flag
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    // Configure Ctrl+C handler for graceful shutdown
    let port_for_cleanup = port;
    let shutdown_handler = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("Received shutdown signal, gracefully shutting down...");
        shutdown_flag_clone.store(true, Ordering::SeqCst);
        // Give time for existing requests to complete
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Cleanup PID file
        let _ = cleanup_pid_file(port_for_cleanup);
        std::process::exit(0);
    });

    let mut builder = rocket::build().configure(Config {
        address: IpAddr::from_str(&address)?,
        port,
        limits: Limits::default()
            .limit("string", ByteUnit::Mebibyte(5))
            .limit("json", ByteUnit::Mebibyte(5))
            .limit("data-form", ByteUnit::Mebibyte(100))
            .limit("file", ByteUnit::Mebibyte(100)),
        ..Config::default()
    });

    builder = builder.mount("/v1/chat", routes![api::chat]);
    builder = builder.mount("/v1", routes![api::chat]);
    builder = builder.mount("/chat", routes![api::chat]);
    // /images/remove_background
    builder = builder.mount("/images", routes![api::remove_background]);
    // /audio/speech and /audio/transcriptions (ASR transcription endpoint)
    builder = builder.mount("/audio", routes![api::speech, api::transcriptions]);
    // /v1/audio/transcriptions (OpenAI standard ASR transcription endpoint)
    builder = builder.mount("/v1/audio", routes![api::transcriptions]);
    // /embeddings and /v1/embeddings (OpenAI-compatible embeddings endpoint)
    builder = builder.mount("/", routes![api::embeddings]);
    builder = builder.mount("/v1", routes![api::embeddings]);
    // /rerank and /v1/rerank
    builder = builder.mount("/", routes![api::rerank]);
    builder = builder.mount("/v1", routes![api::rerank]);
    // Health check and model info endpoints
    builder = builder.mount("/", routes![api::health, api::models]);
    builder = builder.mount("/v1", routes![api::health, api::models]);
    // Shutdown endpoint
    builder = builder.manage(shutdown_flag);
    builder = builder.mount("/", routes![api::shutdown]);

    let _rocket = builder.launch().await?;

    // Cleanup PID file when server exits
    cleanup_pid_file(port)?;
    shutdown_handler.abort();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_run_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "qwen3-embedding-0.6b",
            "--input",
            "hello",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\Qwen3-Embedding-0.6B-ONNX",
            "--tokenizer-dir",
            "D:\\model_download\\Qwen3-Embedding-0.6B-ONNX",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\Qwen3-Embedding-0.6B-ONNX")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\Qwen3-Embedding-0.6B-ONNX")
        );
    }

    #[test]
    fn parse_run_minicpm5_gguf_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "minicpm5-1b",
            "--input",
            "hello",
            "--artifact-format",
            "gguf",
            "--gguf-path",
            "D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::MiniCPM5_1B);
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Gguf)));
        assert_eq!(
            args.gguf_path.as_deref(),
            Some("D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf")
        );
    }

    #[test]
    fn parse_run_minicpm5_weight_path_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "minicpm5-1b",
            "--input",
            "hello",
            "--weight-path",
            "D:\\model_download\\MiniCPM5-1B",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::MiniCPM5_1B);
        assert_eq!(
            args.weight_path.as_deref(),
            Some("D:\\model_download\\MiniCPM5-1B")
        );
    }

    #[test]
    fn parse_run_lfm2_5_weight_path_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "lfm2.5-350m",
            "--input",
            "hello",
            "--weight-path",
            "D:\\model_download\\LFM2.5-350M",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::LFM2_5_350M);
        assert_eq!(
            args.weight_path.as_deref(),
            Some("D:\\model_download\\LFM2.5-350M")
        );
    }

    #[test]
    fn resolve_artifact_kind_prefers_gguf_path_when_format_is_omitted() {
        let artifact = resolve_artifact_kind(
            WhichModel::MiniCPM5_1B,
            None,
            Some("D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf"),
            None,
        );

        assert_eq!(artifact, ArtifactKind::Gguf);
    }

    #[test]
    fn parse_all_minilm_run_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "all-minilm-l6-v2",
            "--input",
            "hello",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\all-MiniLM-L6-v2\\onnx",
            "--tokenizer-dir",
            "D:\\model_download\\all-MiniLM-L6-v2",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(args.model, WhichModel::AllMiniLML6V2);
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\all-MiniLM-L6-v2\\onnx")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\all-MiniLM-L6-v2")
        );
    }

    #[test]
    fn parse_mxbai_embed_xsmall_v1_run_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "mxbai-embed-xsmall-v1",
            "--input",
            "hello",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\mxbai-embed-xsmall-v1\\onnx",
            "--tokenizer-dir",
            "D:\\model_download\\mxbai-embed-xsmall-v1",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(args.model, WhichModel::MxbaiEmbedXsmallV1);
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1\\onnx")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1")
        );
    }

    #[test]
    fn parse_mxbai_embed_xsmall_v1_run_gguf_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "mxbai-embed-xsmall-v1",
            "--input",
            "hello",
            "--artifact-format",
            "gguf",
            "--gguf-path",
            "D:\\model_download\\mxbai-embed-xsmall-v1\\gguf\\mxbai-embed-xsmall-v1-f16.gguf",
            "--tokenizer-dir",
            "D:\\model_download\\mxbai-embed-xsmall-v1",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Gguf)));
        assert_eq!(args.model, WhichModel::MxbaiEmbedXsmallV1);
        assert_eq!(
            args.gguf_path.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1\\gguf\\mxbai-embed-xsmall-v1-f16.gguf")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1")
        );
    }

    #[test]
    fn parse_run_minicpm5_request_json_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "minicpm5-1b",
            "--request-json",
            "D:\\model_download\\minicpm5_request.json",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::MiniCPM5_1B);
        assert_eq!(
            args.request_json.as_deref(),
            Some("D:\\model_download\\minicpm5_request.json")
        );
        assert!(args.input.is_empty());
    }

    #[test]
    fn parse_run_prompt_name_query() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "lfm2.5-embedding-350m",
            "--input",
            "hello",
            "--prompt-name",
            "query",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::LFM2_5Embedding350M);
        assert!(matches!(
            args.prompt_name,
            Some(RunEmbeddingPromptArg::Query)
        ));
    }

    #[test]
    fn parse_glm_ocr_run_gguf_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "glm-ocr",
            "--input",
            "ocr.png",
            "--artifact-format",
            "gguf",
            "--gguf-path",
            "D:\\model_download\\GLM-OCR-GGUF",
            "--max-tokens",
            "8",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Gguf)));
        assert_eq!(args.model, WhichModel::GlmOCR);
        assert_eq!(
            args.gguf_path.as_deref(),
            Some("D:\\model_download\\GLM-OCR-GGUF")
        );
        assert_eq!(args.max_tokens, Some(8));
    }

    #[test]
    fn parse_glm_ocr_run_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "glm-ocr",
            "--input",
            "ocr.png",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\GLM-OCR-ONNX",
            "--tokenizer-dir",
            "D:\\model_download\\GLM-OCR-ONNX",
        ])
        .expect("run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(args.model, WhichModel::GlmOCR);
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\GLM-OCR-ONNX")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\GLM-OCR-ONNX")
        );
    }

    #[test]
    fn parse_yolo_run_stream_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "run",
            "--model",
            "yolo11-detect",
            "--input",
            "rtsp://127.0.0.1/live",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\yolo26m-ONNX\\onnx\\model_q4f16.onnx",
            "--stream",
            "--show",
            "--workers",
            "4",
            "--chunk-size",
            "8",
            "--max-frames",
            "120",
            "--frame-stride",
            "2",
            "--stream-log-every",
            "15",
        ])
        .expect("yolo run args should parse");

        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run subcommand");
        };
        assert_eq!(args.model, WhichModel::Yolo11Detect);
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\yolo26m-ONNX\\onnx\\model_q4f16.onnx")
        );
        assert!(args.stream);
        assert!(args.show);
        assert_eq!(args.workers, Some(4));
        assert_eq!(args.chunk_size, Some(8));
        assert_eq!(args.max_frames, Some(120));
        assert_eq!(args.frame_stride, Some(2));
        assert_eq!(args.stream_log_every, Some(15));
    }

    #[test]
    fn parse_serv_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "serv",
            "--model",
            "qwen3-reranker-0.6b",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\Qwen3-Reranker-0.6B-ONNX",
            "--tokenizer-dir",
            "D:\\model_download\\Qwen3-Reranker-0.6B-ONNX",
        ])
        .expect("serv args should parse");

        let Some(Commands::Serv(args)) = cli.command else {
            panic!("expected serv subcommand");
        };
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\Qwen3-Reranker-0.6B-ONNX")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\Qwen3-Reranker-0.6B-ONNX")
        );
    }

    #[test]
    fn parse_serv_mxbai_embed_xsmall_v1_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "serv",
            "--model",
            "mxbai-embed-xsmall-v1",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\mxbai-embed-xsmall-v1\\onnx",
            "--tokenizer-dir",
            "D:\\model_download\\mxbai-embed-xsmall-v1",
        ])
        .expect("serv args should parse");

        let Some(Commands::Serv(args)) = cli.command else {
            panic!("expected serv subcommand");
        };
        assert_eq!(args.common.model, WhichModel::MxbaiEmbedXsmallV1);
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(
            args.onnx_path.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1\\onnx")
        );
        assert_eq!(
            args.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\mxbai-embed-xsmall-v1")
        );
    }

    #[test]
    fn parse_serv_minicpm5_gguf_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "serv",
            "--model",
            "minicpm5-1b",
            "--artifact-format",
            "gguf",
            "--gguf-path",
            "D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf",
        ])
        .expect("serv args should parse");

        let Some(Commands::Serv(args)) = cli.command else {
            panic!("expected serv subcommand");
        };
        assert_eq!(args.common.model, WhichModel::MiniCPM5_1B);
        assert!(matches!(args.artifact_format, Some(ArtifactArg::Gguf)));
        assert_eq!(
            args.gguf_path.as_deref(),
            Some("D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf")
        );
    }

    #[test]
    fn parse_backward_compatible_root_onnx_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "--model",
            "qwen3.5-0.8b",
            "--artifact-format",
            "onnx",
            "--onnx-path",
            "D:\\model_download\\Qwen3.5-0.8B-ONNX",
            "--tokenizer-dir",
            "D:\\model_download\\Qwen3.5-0.8B-ONNX",
        ])
        .expect("root args should parse");

        assert!(cli.command.is_none());
        assert!(matches!(cli.artifact_format, Some(ArtifactArg::Onnx)));
        assert_eq!(
            cli.onnx_path.as_deref(),
            Some("D:\\model_download\\Qwen3.5-0.8B-ONNX")
        );
        assert_eq!(
            cli.tokenizer_dir.as_deref(),
            Some("D:\\model_download\\Qwen3.5-0.8B-ONNX")
        );
    }

    #[test]
    fn parse_backward_compatible_root_minicpm5_gguf_flags() {
        let cli = Cli::try_parse_from([
            "aha",
            "--model",
            "minicpm5-1b",
            "--artifact-format",
            "gguf",
            "--gguf-path",
            "D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf",
        ])
        .expect("root args should parse");

        assert!(cli.command.is_none());
        assert_eq!(cli.model, Some(WhichModel::MiniCPM5_1B));
        assert!(matches!(cli.artifact_format, Some(ArtifactArg::Gguf)));
        assert_eq!(
            cli.gguf_path.as_deref(),
            Some("D:\\model_download\\MiniCPM5-1B-GGUF\\MiniCPM5-1B-Q4_K_M.gguf")
        );
    }
}
