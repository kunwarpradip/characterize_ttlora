from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cttlora.config import DataConfig, ModelConfig
from cttlora.data import prepare_phase1_data
from cttlora.modeling import load_sequence_classification_model
from cttlora.tasks import get_task_spec
from cttlora.training import resolve_device


def set_dropout_zero(module) -> None:
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0


def load_checkpoint_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensor_path = checkpoint_dir / "model.safetensors"
    if safetensor_path.exists():
        return load_safetensors_file(str(safetensor_path))
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No supported checkpoint file found in {checkpoint_dir}")


def timed_forward(model, batch: dict[str, torch.Tensor], warmup_steps: int, measure_steps: int, device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_steps):
        with torch.no_grad():
            model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latencies = []
    logits_reference = None
    for _ in range(measure_steps):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        if logits_reference is None:
            logits_reference = outputs.logits.detach().cpu()
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return {
        "latency_mean_ms": statistics.mean(latencies) * 1000.0,
        "latency_std_ms": statistics.pstdev(latencies) * 1000.0 if len(latencies) > 1 else 0.0,
        "peak_memory_gb": peak_memory_gb,
        "logits": logits_reference,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark contraction vs reconstruction on the same trained checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--output-dir", default=str(PHASE_ROOT / "analysis" / "inference"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_dir = run_dir / "checkpoints" / "best"
    checkpoint_state = load_checkpoint_state(checkpoint_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    model_cfg = config["model"]
    dataset_root = data_cfg["dataset_root"]
    dataset_name = data_cfg["dataset_name"]
    tokenizer_name = model_cfg["tokenizer_name_or_path"]
    task_spec = get_task_spec(dataset_name)
    device = resolve_device(args.device)

    _, _, val_dataset, _, val_loader = prepare_phase1_data(
        data_config=DataConfig(**data_cfg),
        tokenizer_name_or_path=tokenizer_name,
        task_spec=task_spec,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,
    )
    batch = next(iter(val_loader))
    batch = {key: value.to(device) for key, value in batch.items()}

    results = {}
    logits_by_mode = {}
    for mode in ("contraction", "reconstruction"):
        cfg = dict(model_cfg)
        cfg["ttlora_variant"] = mode
        model = load_sequence_classification_model(ModelConfig(**cfg), task_spec.num_labels).to(device).float()
        model.apply(set_dropout_zero)
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch for mode={mode}. missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        model.eval()
        timed = timed_forward(model, batch, args.warmup_steps, args.measure_steps, device)
        logits_by_mode[mode] = timed.pop("logits")
        results[mode] = timed

    results["logits_max_abs_diff"] = float((logits_by_mode["contraction"] - logits_by_mode["reconstruction"]).abs().max().item())
    results["run_dir"] = str(run_dir)
    results["dataset_name"] = dataset_name
    results["batch_size"] = args.batch_size
    output_path = output_dir / f"{run_dir.name}_inference_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
