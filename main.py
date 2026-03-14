import yaml
import argparse
import os
import time
import platform
import socket
import getpass
import subprocess
import sys
import re
import shlex
import inspect
import csv
import fcntl
import importlib
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
# import gym_maze
# import gym_navigation
from envs import nim, pong


def load_runner_modules():
    return {
        "basic": importlib.import_module("runner.llm_num_optim_runner"),
        "semantics": importlib.import_module("runner.llm_num_optim_semantics_runner"),
        "feedback": importlib.import_module("runner.llm_num_optim_semantics_feedback_runner"),
    }


def apply_template_variant(config):
    variant = config.get("optim_template_variant")
    if not variant:
        return

    template_dir = config.get("template_dir")
    if not template_dir:
        return

    keys = ["llm_si_template_name", "llm_output_conversion_template_name"]
    for key in keys:
        template_name = config.get(key)
        if not template_name:
            continue

        base_name, ext = os.path.splitext(template_name)
        candidate_name = f"{base_name}_{variant}{ext}" if ext else f"{template_name}_{variant}"
        candidate_path = os.path.join(template_dir, candidate_name)

        if os.path.exists(candidate_path):
            config[key] = candidate_name
        else:
            print(
                f"[WARNING] Requested template variant '{variant}' for {key}, but '{candidate_name}' was not found in '{template_dir}'. Using '{template_name}'."
            )


def resolve_repetition_id(cli_repetition_id=None, cli_run_id=None):
    if cli_repetition_id:
        return str(cli_repetition_id)
    if cli_run_id:
        return str(cli_run_id)

    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_array_job_id and slurm_array_task_id:
        return f"{slurm_array_job_id}_{slurm_array_task_id}"

    env_candidates = [
        slurm_array_task_id,
        slurm_array_job_id,
        os.getenv("SLURM_JOB_ID"),
        os.getenv("JOB_ID"),
        os.getenv("PBS_JOBID"),
    ]
    for candidate in env_candidates:
        if candidate:
            return str(candidate)

    return str(int(time.time()))


def resolve_job_id(resolved_repetition_id=None):
    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if slurm_array_job_id:
        return str(slurm_array_job_id)

    if resolved_repetition_id:
        text = str(resolved_repetition_id)
        if "_" in text:
            return text.split("_", 1)[0]

    env_candidates = [
        os.getenv("SLURM_JOB_ID"),
        os.getenv("JOB_ID"),
        os.getenv("PBS_JOBID"),
    ]
    for candidate in env_candidates:
        if candidate:
            return str(candidate)

    return f"manual_{int(time.time())}"


def normalize_repetition_id(repetition_id, job_id):
    rep = str(repetition_id)
    if rep.startswith(f"{job_id}_"):
        return rep
    if rep.isdigit() and str(job_id).isdigit():
        return f"{job_id}_{rep}"
    return rep


def validate_identifier(name, value):
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(
            f"Invalid {name} '{value}'. Only [A-Za-z0-9_.-] are allowed to prevent path mixing."
        )


def validate_output_paths(base_logdir, job_logdir, repetition_logdir, job_id, repetition_id):
    abs_base = os.path.abspath(base_logdir)
    abs_job = os.path.abspath(job_logdir)
    abs_rep = os.path.abspath(repetition_logdir)

    expected_job_name = f"job_{job_id}"
    expected_rep_name = f"repetition_{repetition_id}"

    if os.path.basename(abs_job) != expected_job_name:
        raise ValueError(
            f"Job folder mismatch: expected '{expected_job_name}', got '{os.path.basename(abs_job)}'."
        )
    if os.path.basename(abs_rep) != expected_rep_name:
        raise ValueError(
            f"Repetition folder mismatch: expected '{expected_rep_name}', got '{os.path.basename(abs_rep)}'."
        )
    if os.path.dirname(abs_job) != abs_base:
        raise ValueError(
            "Invalid job path layout. Job folder must be directly under the family logdir."
        )
    if os.path.dirname(abs_rep) != abs_job:
        raise ValueError(
            "Invalid repetition path layout. Repetition folder must be directly under the job folder."
        )
    if not abs_rep.startswith(abs_job + os.sep):
        raise ValueError("Path mixing detected: repetition folder is not inside its own job folder.")


def run_command(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        return output
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def collect_relevant_environment():
    prefixes = (
        "SLURM_",
        "CUDA",
        "NVIDIA",
        "VLLM",
        "OPENAI",
        "OLLAMA",
        "HF_",
        "PYTHON",
    )
    secret_tokens = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    env_out = {}
    for key, value in os.environ.items():
        if key.startswith(prefixes):
            if any(token in key.upper() for token in secret_tokens):
                env_out[key] = "<redacted>"
            else:
                env_out[key] = value
    return dict(sorted(env_out.items(), key=lambda item: item[0]))


def collect_cpu_info():
    cpu_model = "<unknown>"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    memory_gb = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        memory_gb = round((page_size * phys_pages) / (1024 ** 3), 2)
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "cpu_count_logical": os.cpu_count(),
        "memory_gb": memory_gb,
    }


def collect_gpu_info():
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    raw_nvidia_smi = run_command(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", raw_nvidia_smi)
    cuda_version = cuda_match.group(1) if cuda_match else None

    gpus = []
    if query and not query.startswith("<unavailable"):
        for row in query.splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total": parts[2],
                        "driver_version": parts[3],
                    }
                )

    return {
        "gpu_count": len(gpus),
        "gpus": gpus,
        "cuda_version": cuda_version,
    }


def parse_launch_script(project_root):
    launch_script_path = Path(project_root) / "run_sol.sh"
    if not launch_script_path.exists():
        return {
            "path": str(launch_script_path),
            "exists": False,
            "content": None,
            "sbatch_directives": {},
            "vllm_apptainer_args": {},
            "python_launch_snippet": None,
        }

    script_content = launch_script_path.read_text(encoding="utf-8")
    sbatch_directives = {}
    lines = script_content.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue

        directive = stripped[len("#SBATCH"):].strip()
        if not directive:
            continue

        if directive.startswith("--") and "=" in directive:
            key, value = directive.split("=", 1)
            sbatch_directives[key.strip()] = value.strip()
        else:
            parts = directive.split(maxsplit=1)
            key = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            sbatch_directives[key] = value

    def extract_multiline_command(start_pattern):
        for i, line in enumerate(lines):
            if re.search(start_pattern, line):
                chunk = [line.rstrip()]
                j = i + 1
                while j < len(lines):
                    chunk.append(lines[j].rstrip())
                    prev = lines[j - 1].rstrip()
                    curr = lines[j].rstrip()
                    if prev.endswith("&") or curr.endswith("&"):
                        break
                    if not prev.endswith("\\"):
                        break
                    j += 1
                return "\n".join(chunk).strip()
        return None

    apptainer_snippet = extract_multiline_command(r"\bapptainer\s+run\b")
    vllm_args = {}
    if apptainer_snippet:
        cmd_text = apptainer_snippet.replace("\\\n", " ").replace("&", " ")
        try:
            tokens = shlex.split(cmd_text)
            idx = 0
            while idx < len(tokens):
                token = tokens[idx]
                if token.startswith("--"):
                    key = token
                    next_value = None
                    if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                        next_value = tokens[idx + 1]
                        idx += 1
                    vllm_args[key] = next_value if next_value is not None else True
                idx += 1
        except Exception:
            vllm_args = {}

    python_launch_snippet = extract_multiline_command(r"\bpython\s+main\.py\b")

    return {
        "path": str(launch_script_path),
        "exists": True,
        "content": script_content,
        "sbatch_directives": sbatch_directives,
        "vllm_apptainer_args": vllm_args,
        "vllm_apptainer_snippet": apptainer_snippet,
        "python_launch_snippet": python_launch_snippet,
    }


def build_concise_python_command(argv):
    if not argv:
        return "python"
    pieces = ["python"] + [shlex.quote(arg) for arg in argv[1:]]
    return " ".join(pieces)


def summarize_launch_script_for_manifest(launch_script_data, launch_artifacts):
    script_content = launch_script_data.get("content")
    script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest() if script_content else None
    return {
        "path": launch_script_data.get("path"),
        "exists": launch_script_data.get("exists"),
        "saved_copy_path": launch_artifacts.get("run_sol_copy_path"),
        "sha256": script_hash,
        "sbatch_directives": launch_script_data.get("sbatch_directives", {}),
        "vllm_apptainer_args": launch_script_data.get("vllm_apptainer_args", {}),
        "vllm_apptainer_snippet": launch_script_data.get("vllm_apptainer_snippet"),
        "python_launch_snippet": launch_script_data.get("python_launch_snippet"),
    }


def extract_vllm_knobs(config, launch_script_data, env_data):
    keys = [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "n",
        "best_of",
        "stop",
        "stop_token_ids",
        "max_tokens",
        "min_tokens",
        "max_model_len",
        "context_size",
        "num_ctx",
        "truncate_prompt_tokens",
        "ignore_eos",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "dtype",
        "quantization",
        "kv_cache_dtype",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_prefix_caching",
        "chunked_prefill_enabled",
        "seed",
        "enforce_eager",
        "attention_backend",
        "model_revision",
        "tokenizer_revision",
        "trust_remote_code",
        "request_timeout",
        "num_retries",
        "served_model_name",
        "llm_model_name",
    ]

    knobs = {}
    launch_args = launch_script_data.get("vllm_apptainer_args", {})
    for key in keys:
        value = config.get(key)
        if value is not None:
            knobs[key] = value
            continue

        dashed = f"--{key.replace('_', '-')}"
        underscored = f"--{key}"
        if dashed in launch_args:
            knobs[key] = launch_args[dashed]
            continue
        if underscored in launch_args:
            knobs[key] = launch_args[underscored]
            continue

        env_key = f"VLLM_{key.upper()}"
        if env_key in env_data:
            knobs[key] = env_data.get(env_key)

    return knobs


def extract_vllm_knobs_with_sources(config, launch_script_data, env_data):
    default_values = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "n": 1,
        "max_tokens": 16,
        "min_tokens": 0,
        "ignore_eos": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "dtype": "auto",
        "max_num_seqs": 256,
    }

    keys = list(set(list(default_values.keys()) + list(extract_vllm_knobs(config, launch_script_data, env_data).keys())))
    launch_args = launch_script_data.get("vllm_apptainer_args", {})
    with_sources = {}

    for key in sorted(keys):
        value = None
        source = None

        if key in config and config.get(key) is not None:
            value = config.get(key)
            source = "config"
        else:
            dashed = f"--{key.replace('_', '-')}"
            underscored = f"--{key}"
            if dashed in launch_args:
                value = launch_args[dashed]
                source = "run_sol.sh(vllm_args)"
            elif underscored in launch_args:
                value = launch_args[underscored]
                source = "run_sol.sh(vllm_args)"
            else:
                env_key = f"VLLM_{key.upper()}"
                if env_key in env_data:
                    value = env_data.get(env_key)
                    source = "environment"
                elif key in default_values:
                    value = default_values[key]
                    source = "default"

        if source is not None:
            with_sources[key] = {"value": value, "source": source}

    return with_sources


def apply_runner_defaults(task, run_config, config_path_fallback, runner_modules=None):
    if runner_modules is None:
        runner_modules = load_runner_modules()

    if task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        runner_fn = runner_modules["basic"].run_training_loop
    elif task in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        runner_fn = runner_modules["semantics"].run_training_loop
    elif task in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]:
        runner_fn = runner_modules["feedback"].run_training_loop
    else:
        raise ValueError(f"Task {task} not recognized.")

    effective_config = dict(run_config)
    injected_defaults = {}
    signature = inspect.signature(runner_fn)
    for param_name, param in signature.parameters.items():
        if param.default is not inspect._empty and param_name not in effective_config:
            effective_config[param_name] = param.default
            injected_defaults[param_name] = param.default

    if (
        task in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]
        and "config_path" not in effective_config
    ):
        effective_config["config_path"] = config_path_fallback
        injected_defaults["config_path"] = config_path_fallback

    return effective_config, injected_defaults


def probe_http_endpoint(url, timeout_sec=3):
    try:
        req = Request(url, headers={"User-Agent": "props-llm-manifest-probe"})
        with urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": True,
                "status": resp.status,
                "body_preview": body,
            }
    except HTTPError as http_exc:
        return {
            "ok": False,
            "status": http_exc.code,
            "error": str(http_exc),
        }
    except URLError as url_exc:
        return {
            "ok": False,
            "status": None,
            "error": str(url_exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "error": str(exc),
        }


def collect_vllm_live_status(openai_base_url):
    if not openai_base_url:
        return {"probed": False, "reason": "OPENAI_BASE_URL is not set"}

    base = openai_base_url.rstrip("/")
    server_root = base[:-3] if base.endswith("/v1") else base

    status = {
        "probed": True,
        "openai_base_url": openai_base_url,
        "models": probe_http_endpoint(f"{base}/models"),
        "health": probe_http_endpoint(f"{server_root}/health"),
        "metrics": probe_http_endpoint(f"{server_root}/metrics"),
    }
    return status


def update_latest_pointer(base_logdir, job_dir):
    latest_text_path = Path(base_logdir) / "latest_job.txt"
    latest_text_path.write_text(f"{Path(job_dir).name}\n", encoding="utf-8")

    latest_symlink_path = Path(base_logdir) / "latest"
    try:
        if latest_symlink_path.is_symlink() or latest_symlink_path.exists():
            latest_symlink_path.unlink()
        latest_symlink_path.symlink_to(Path(job_dir).name)
    except Exception:
        pass


def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_jsonable(data), f, sort_keys=False)


def save_launch_artifacts(job_logdir, launch_script_data, exact_command):
    Path(job_logdir).mkdir(parents=True, exist_ok=True)
    command_path = Path(job_logdir) / "exact_command.txt"
    command_path.write_text(f"{exact_command}\n", encoding="utf-8")

    if launch_script_data.get("exists") and launch_script_data.get("content"):
        run_sol_copy_path = Path(job_logdir) / "run_sol_used.sh"
        run_sol_copy_path.write_text(launch_script_data.get("content"), encoding="utf-8")
        return {
            "exact_command_path": str(command_path),
            "run_sol_copy_path": str(run_sol_copy_path),
        }

    return {
        "exact_command_path": str(command_path),
        "run_sol_copy_path": None,
    }


def update_repetition_master(job_logdir, job_id, repetition_id, updates):
    master_csv_path = Path(job_logdir) / "repetition_master.csv"
    lock_path = Path(job_logdir) / ".repetition_master.lock"
    Path(job_logdir).mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            repetitions = {}
            if master_csv_path.exists():
                with open(master_csv_path, "r", newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        rep_id = row.get("repetition_id")
                        if rep_id:
                            repetitions[rep_id] = dict(row)

            record = repetitions.get(str(repetition_id), {})
            record.update(_to_jsonable(updates))
            repetitions[str(repetition_id)] = record

            with open(master_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                fieldnames = [
                    "repetition_id",
                    "array_task_id",
                    "host",
                    "pid",
                    "start_time_utc",
                    "end_time_utc",
                    "status",
                    "duration_seconds",
                    "error",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for rep_id, rep_record in sorted(repetitions.items()):
                    writer.writerow(
                        {
                            "repetition_id": rep_id,
                            "array_task_id": rep_record.get("array_task_id"),
                            "host": rep_record.get("host"),
                            "pid": rep_record.get("pid"),
                            "start_time_utc": rep_record.get("start_time_utc"),
                            "end_time_utc": rep_record.get("end_time_utc"),
                            "status": rep_record.get("status"),
                            "duration_seconds": rep_record.get("duration_seconds"),
                            "error": rep_record.get("error"),
                        }
                    )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_job_manifest_markdown(manifest):
    lines = []
    lines.append("# Job Manifest")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Job ID: `{manifest['run_identity']['job_id']}`")
    lines.append(f"- Manifest Generated At (UTC): `{manifest['run_identity']['timestamp_utc']}`")
    lines.append(f"- User/Host: `{manifest['run_identity']['user']}@{manifest['run_identity']['hostname']}`")
    lines.append(f"- Python: `{manifest['run_identity']['python_version']}`")
    lines.append("")
    lines.append("## Paths")
    lines.append(f"- Family logdir: `{manifest['paths']['base_logdir']}`")
    lines.append(f"- Job directory: `{manifest['paths']['job_logdir']}`")
    lines.append(f"- Source config snapshot: `{manifest['paths']['source_config_snapshot']}`")
    lines.append(f"- vLLM server config: `{manifest['paths']['vllm_server_snapshot']}`")
    lines.append("")
    lines.append("## Launch")
    lines.append(f"- Exact command: `{manifest['execution']['exact_command']}`")
    lines.append(f"- OpenAI base URL: `{manifest['execution']['openai_base_url']}`")
    lines.append(f"- Walltime limit: `{manifest['execution'].get('walltime_limit', '')}`")
    lines.append("")
    lines.append("## Hardware")
    hardware = manifest["hardware"]
    lines.append(f"- CPU: `{hardware['cpu'].get('cpu_model')}`")
    lines.append(f"- CPU logical cores: `{hardware['cpu'].get('cpu_count_logical')}`")
    lines.append(f"- RAM (GB): `{hardware['cpu'].get('memory_gb')}`")
    lines.append(f"- GPUs: `{hardware['gpu'].get('gpu_count')}`")
    lines.append(f"- CUDA version: `{hardware['gpu'].get('cuda_version')}`")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("- Use the exact command under Launch or the template below.")
    lines.append(f"- Template: `{manifest['rerun']['repetition_command_template']}`")
    lines.append(f"- Exact command file: `{manifest['rerun'].get('exact_command_path')}`")
    lines.append(f"- Used run_sol.sh copy: `{manifest['rerun'].get('run_sol_copy_path')}`")
    lines.append(f"- Repetition timeline (master): `{manifest['rerun'].get('repetition_master_csv')}`")
    lines.append("- Full details are in this file, `exact_command.txt`, and `vllm_server_effective.yaml`.")
    lines.append("")
    lines.append("## vLLM and LLM Knobs 🎛️🎛️🎛️")
    knobs_with_sources = manifest.get("vllm_llm_knobs_with_sources", {})
    for key, item in knobs_with_sources.items():
        value = item.get("value") if isinstance(item, dict) else item
        source = item.get("source") if isinstance(item, dict) else "unknown"
        lines.append(f"- {key}: `{value}` (source: `{source}`)")

    live = manifest.get("vllm_live_status", {})
    lines.append("")
    lines.append("## vLLM Live Probe")
    lines.append(f"- Probed: `{live.get('probed')}`")
    if live.get("probed"):
        models = live.get("models", {})
        health = live.get("health", {})
        metrics = live.get("metrics", {})
        lines.append(f"- /v1/models: status=`{models.get('status')}`, ok=`{models.get('ok')}`")
        lines.append(f"- /health: status=`{health.get('status')}`, ok=`{health.get('ok')}`")
        lines.append(f"- /metrics: status=`{metrics.get('status')}`, ok=`{metrics.get('ok')}`")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--repetition_id",
        type=str,
        default=None,
        help="Repetition/job id used for log folder naming",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Deprecated alias for --repetition_id",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local OpenAI-compatible server (e.g., vLLM)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Create folders/manifests/metadata only, then exit before training.",
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config.pop("num_repeats", None)
    config.pop("num_runs", None)
    base_logdir = config.get("logdir")
    repetition_id = resolve_repetition_id(args.repetition_id, args.run_id)
    job_id = resolve_job_id(repetition_id)
    repetition_id = normalize_repetition_id(repetition_id, job_id)
    validate_identifier("job_id", job_id)
    validate_identifier("repetition_id", repetition_id)

    run_config = dict(config)
    runner_modules = None
    job_logdir = None
    repetition_logdir = None
    repetition_start_time_utc = datetime.now(timezone.utc).isoformat()
    repetition_start_epoch = time.time()
    run_status = "failed"
    run_error_summary = None

    # Route OpenAI-compatible local servers (like vLLM) via env var.
    # Existing agent code reads OPENAI_BASE_URL to switch from Ollama mode.
    if args.port:
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"

    if base_logdir:
        job_logdir = os.path.join(base_logdir, f"job_{job_id}")
        repetition_logdir = os.path.join(job_logdir, f"repetition_{repetition_id}")
        validate_output_paths(base_logdir, job_logdir, repetition_logdir, job_id, repetition_id)
        os.makedirs(repetition_logdir, exist_ok=True)
        run_config["logdir"] = repetition_logdir

        source_config_snapshot = os.path.join(job_logdir, "config_source.yaml")
        vllm_server_snapshot = os.path.join(job_logdir, "vllm_server_effective.yaml")
        exact_command = build_concise_python_command([sys.executable] + sys.argv)
        exact_command_full = " ".join([shlex.quote(sys.executable)] + [shlex.quote(arg) for arg in sys.argv])

        config_overrides = {
            "logdir": {
                "value": repetition_logdir,
                "source": "main.py(runtime_path_layout)",
            },
            "repetition_id": {
                "value": repetition_id,
                "source": "main.py(cli_or_scheduler)",
            },
            "job_id": {
                "value": job_id,
                "source": "main.py(cli_or_scheduler)",
            },
            "openai_base_url": {
                "value": os.getenv("OPENAI_BASE_URL"),
                "source": "main.py(--port)",
            },
        }

        save_yaml(source_config_snapshot, config)

        apply_template_variant(run_config)
        if run_config.get("llm_si_template_name") != config.get("llm_si_template_name"):
            config_overrides["llm_si_template_name"] = {
                "value": run_config.get("llm_si_template_name"),
                "source": "main.py(optim_template_variant)",
            }
        if run_config.get("llm_output_conversion_template_name") != config.get("llm_output_conversion_template_name"):
            config_overrides["llm_output_conversion_template_name"] = {
                "value": run_config.get("llm_output_conversion_template_name"),
                "source": "main.py(optim_template_variant)",
            }

        update_latest_pointer(base_logdir, job_logdir)

        project_root = os.getcwd()
        launch_script_data = parse_launch_script(project_root)
        launch_artifacts = save_launch_artifacts(job_logdir, launch_script_data, exact_command)
        launch_script_summary = summarize_launch_script_for_manifest(launch_script_data, launch_artifacts)
        env_data = collect_relevant_environment()
        dependencies = {
            "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
            "framework_versions": {
                "torch": run_command([sys.executable, "-c", "import torch; print(torch.__version__)"]),
                "gymnasium": run_command([sys.executable, "-c", "import gymnasium; print(gymnasium.__version__)"]),
                "vllm": run_command([sys.executable, "-c", "import vllm; print(vllm.__version__)"]),
                "tokenizers": run_command([sys.executable, "-c", "import tokenizers; print(tokenizers.__version__)"]),
                "transformers": run_command([sys.executable, "-c", "import transformers; print(transformers.__version__)"]),
            },
        }
        sampling_keys = [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "n",
            "best_of",
            "stop",
            "stop_token_ids",
            "max_tokens",
            "min_tokens",
            "ignore_eos",
            "truncate_prompt_tokens",
        ]
        update_repetition_master(
            job_logdir,
            job_id,
            repetition_id,
            {
                "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "start_time_utc": repetition_start_time_utc,
                "status": "running",
                "command": exact_command,
            },
        )

        sampling_and_control = {
            key: run_config.get(key)
            for key in sampling_keys
            if run_config.get(key) is not None
        }

        try:
            runner_modules = load_runner_modules()
            effective_run_config, injected_defaults = apply_runner_defaults(
                run_config["task"],
                run_config,
                source_config_snapshot,
                runner_modules=runner_modules,
            )
        except Exception as exc:
            if args.dry_run:
                print(f"[DRY RUN] Could not import runner modules for default inference: {exc}")
                effective_run_config, injected_defaults = dict(run_config), {}
            else:
                raise

        # Save a focused snapshot of what the vLLM server is actually running with.
        _launch_args = launch_script_data.get("vllm_apptainer_args", {})
        _vllm_knobs_resolved = extract_vllm_knobs_with_sources(effective_run_config, launch_script_data, env_data)
        vllm_server_effective = {}
        if "--model" in _launch_args:
            vllm_server_effective["model_path"] = _launch_args["--model"]
        for _k, _v in sorted(_vllm_knobs_resolved.items()):
            vllm_server_effective[_k] = _v["value"]
        if args.port:
            vllm_server_effective["port"] = int(args.port)
        save_yaml(vllm_server_snapshot, vllm_server_effective)

        run_config = effective_run_config

        if injected_defaults:
            for key, value in injected_defaults.items():
                if key not in config_overrides:
                    config_overrides[key] = {
                        "value": value,
                        "source": "runner_default",
                    }

        prompting_and_eval = {
            "template_dir": run_config.get("template_dir"),
            "llm_si_template_name": run_config.get("llm_si_template_name"),
            "llm_output_conversion_template_name": run_config.get("llm_output_conversion_template_name"),
            "env_desc_file": run_config.get("env_desc_file"),
            "include_trajectories": run_config.get("include_trajectories"),
            "reward_shaping_flags": {
                "optimum": run_config.get("optimum"),
                "search_step_size": run_config.get("search_step_size"),
                "bias": run_config.get("bias"),
                "param_min": run_config.get("param_min"),
                "param_max": run_config.get("param_max"),
            },
            "eval_protocol": {
                "num_evaluation_episodes": run_config.get("num_evaluation_episodes"),
                "max_traj_length": run_config.get("max_traj_length"),
                "feedback_interval": run_config.get("feedback_interval"),
            },
        }

        manifest = {
            "run_identity": {
                "job_id": str(job_id),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "user": getpass.getuser(),
                "hostname": socket.gethostname(),
                "os": platform.platform(),
                "python_version": platform.python_version(),
            },
            "paths": {
                "base_logdir": base_logdir,
                "job_logdir": job_logdir,
                "source_config_snapshot": source_config_snapshot,
                "vllm_server_snapshot": vllm_server_snapshot,
            },
            "execution": {
                "exact_command": exact_command,
                "exact_command_full": exact_command_full,
                "python_executable": sys.executable,
                "argv": [sys.executable] + sys.argv,
                "openai_base_url": os.getenv("OPENAI_BASE_URL"),
                "walltime_limit": os.getenv("SLURM_TIMELIMIT") or launch_script_data.get("sbatch_directives", {}).get("--time"),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "slurm_script": launch_script_summary,
                "exact_command_path": launch_artifacts.get("exact_command_path"),
                "run_sol_copy_path": launch_artifacts.get("run_sol_copy_path"),
            },
            "seeds": {
                "global_seed": run_config.get("seed"),
                "numpy_seed": run_config.get("numpy_seed"),
                "torch_seed": run_config.get("torch_seed"),
                "env_seed": run_config.get("env_seed"),
            },
            "hardware": {
                "cpu": collect_cpu_info(),
                "gpu": collect_gpu_info(),
            },
            "dependencies": dependencies,
            "environment": env_data,
            "config_overrides_effective": config_overrides,
            "runner_defaults_injected": injected_defaults,
            "sampling_and_control": sampling_and_control,
            "prompting_and_eval": prompting_and_eval,
            "vllm_llm_knobs": extract_vllm_knobs(run_config, launch_script_data, env_data),
            "vllm_llm_knobs_with_sources": extract_vllm_knobs_with_sources(run_config, launch_script_data, env_data),
            "vllm_live_status": collect_vllm_live_status(os.getenv("OPENAI_BASE_URL")),
            "vllm_server_effective": vllm_server_effective,
            "rerun": {
                "repetition_command_template": (
                    f"python main.py --config {source_config_snapshot} --repetition_id {job_id}_<TASK_ID>"
                    + (f" --port {args.port}" if args.port else "")
                ),
                "job_array_hint": "sbatch run_sol.sh",
                "print_command_utility": "python utils/print_rerun_command.py --job-dir <job_dir>",
                "exact_command_path": launch_artifacts.get("exact_command_path"),
                "run_sol_copy_path": launch_artifacts.get("run_sol_copy_path"),
                "repetition_master_csv": os.path.join(job_logdir, "repetition_master.csv"),
            },
        }


        manifest_md_path = os.path.join(job_logdir, "JOB_MANIFEST.md")
        Path(manifest_md_path).write_text(build_job_manifest_markdown(manifest), encoding="utf-8")

        repetition_context_path = os.path.join(repetition_logdir, "run_context.yaml")
        save_yaml(
            repetition_context_path,
            {
                "job_id": str(job_id),
                "repetition_id": str(repetition_id),
                "repetition_logdir": repetition_logdir,
                "job_manifest_path": manifest_md_path,
                "source_config_snapshot": source_config_snapshot,
                "exact_command": exact_command,
            },
        )

        if run_config["task"] in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]:
            run_config["config_path"] = source_config_snapshot
    else:
        apply_template_variant(run_config)

    try:
        if args.dry_run:
            print("[DRY RUN] Artifacts generated. Skipping training loop.")
        else:
            if runner_modules is None:
                runner_modules = load_runner_modules()

            if run_config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
                runner_modules["basic"].run_training_loop(**run_config)
            elif run_config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
                runner_modules["semantics"].run_training_loop(**run_config)
            elif run_config["task"] in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]:
                if "config_path" not in run_config:
                    run_config["config_path"] = args.config
                runner_modules["feedback"].run_training_loop(**run_config)
            else:
                raise ValueError(f"Task {run_config['task']} not recognized.")
        run_status = "success"
    except Exception as exc:
        run_error_summary = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if job_logdir and repetition_logdir:
            repetition_end_time_utc = datetime.now(timezone.utc).isoformat()
            duration_seconds = round(time.time() - repetition_start_epoch, 3)
            update_repetition_master(
                job_logdir,
                job_id,
                repetition_id,
                {
                    "end_time_utc": repetition_end_time_utc,
                    "status": run_status,
                    "duration_seconds": duration_seconds,
                    "error": run_error_summary,
                },
            )
            save_yaml(
                os.path.join(repetition_logdir, "repetition_timing.yaml"),
                {
                    "job_id": job_id,
                    "repetition_id": repetition_id,
                    "start_time_utc": repetition_start_time_utc,
                    "end_time_utc": repetition_end_time_utc,
                    "status": run_status,
                    "duration_seconds": duration_seconds,
                    "error": run_error_summary,
                },
            )


if __name__ == "__main__":
    main()
