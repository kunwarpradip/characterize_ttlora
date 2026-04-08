from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cttlora.config import DataConfig, ModelConfig
from cttlora.data import prepare_phase1_data
from cttlora.modeling import load_sequence_classification_model, named_ttlora_modules
from cttlora.tasks import get_task_spec
from cttlora.training import resolve_device, seed_everything


def set_dropout_zero(module) -> None:
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0


def max_tensor_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def tensors_close(left: torch.Tensor, right: torch.Tensor, rtol: float, atol: float) -> bool:
    return bool(torch.allclose(left, right, rtol=rtol, atol=atol))


def ttlora_grad_state(model) -> dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if "tt_cores" in name:
            if param.grad is None:
                raise ValueError(f"Missing gradient for TT core parameter {name}")
            state[name] = param.grad.detach().cpu().clone()
    return state


def compare_grad_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, float]:
    diffs = {}
    for name in sorted(left):
        if name not in right:
            raise KeyError(f"Gradient state missing key {name} in right model.")
        diffs[name] = max_tensor_diff(left[name], right[name])
    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify contraction and reconstruction are functionally equivalent.")
    parser.add_argument("--dataset-name", default="mrpc")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "roberta-base" / "checkpoints"))
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-modules", nargs="*", default=("query", "value"))
    parser.add_argument("--ttlora-rank", type=int, default=5)
    parser.add_argument("--ttlora-alpha", type=float, default=8.0)
    parser.add_argument("--ttlora-shape", nargs="*", type=int, default=(64, 4, 3, 3, 4, 64))
    parser.add_argument("--ttlora-input-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--ttlora-output-factors", nargs="*", type=int, default=(64, 4, 3))
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output-dir", default=str(PHASE_ROOT / "analysis" / "equivalence"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = resolve_device(args.device)
    task_spec = get_task_spec(args.dataset_name)
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        max_length=args.max_length,
        max_train_samples=args.batch_size,
        max_eval_samples=args.batch_size,
    )
    tokenizer_name = args.tokenizer_path or args.model_path
    _, _, _, train_loader, _ = prepare_phase1_data(
        data_config=data_config,
        tokenizer_name_or_path=tokenizer_name,
        task_spec=task_spec,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,
    )
    batch = next(iter(train_loader))
    batch = {key: value.to(device) for key, value in batch.items()}

    common_kwargs = dict(
        model_name_or_path=args.model_path,
        tokenizer_name_or_path=tokenizer_name,
        adaptation_method="ttlora",
        target_modules=tuple(args.target_modules),
        ttlora_rank=args.ttlora_rank,
        ttlora_alpha=args.ttlora_alpha,
        ttlora_shape=tuple(args.ttlora_shape),
        ttlora_input_factors=tuple(args.ttlora_input_factors),
        ttlora_output_factors=tuple(args.ttlora_output_factors),
    )
    contraction_cfg = ModelConfig(ttlora_variant="contraction", **common_kwargs)
    reconstruction_cfg = ModelConfig(ttlora_variant="reconstruction", **common_kwargs)

    contraction_model = load_sequence_classification_model(contraction_cfg, task_spec.num_labels).to(device).float()
    reconstruction_model = load_sequence_classification_model(reconstruction_cfg, task_spec.num_labels).to(device).float()
    contraction_model.apply(set_dropout_zero)
    reconstruction_model.apply(set_dropout_zero)
    reconstruction_model.load_state_dict(contraction_model.state_dict(), strict=True)
    contraction_model.eval()
    reconstruction_model.eval()

    contraction_modules = named_ttlora_modules(contraction_model)
    reconstruction_modules = named_ttlora_modules(reconstruction_model)
    if not contraction_modules or not reconstruction_modules:
        raise ValueError("No TT-LoRA modules were found after model construction.")

    first_contraction_name, first_contraction = contraction_modules[0]
    _, first_reconstruction = reconstruction_modules[0]
    synthetic_input = torch.randn(2, 5, first_contraction.original.in_features, device=device, dtype=torch.float32)

    with torch.no_grad():
        standalone_contraction = first_contraction(synthetic_input)
        standalone_reconstruction = first_reconstruction(synthetic_input)
        contraction_outputs = contraction_model(**batch)
        reconstruction_outputs = reconstruction_model(**batch)

    contraction_model.zero_grad(set_to_none=True)
    reconstruction_model.zero_grad(set_to_none=True)
    contraction_loss = contraction_model(**batch).loss
    reconstruction_loss = reconstruction_model(**batch).loss
    contraction_loss.backward()
    reconstruction_loss.backward()

    gradient_diffs = compare_grad_states(ttlora_grad_state(contraction_model), ttlora_grad_state(reconstruction_model))
    standalone_forward_diff = max_tensor_diff(standalone_contraction, standalone_reconstruction)
    logits_diff = max_tensor_diff(contraction_outputs.logits, reconstruction_outputs.logits)
    loss_diff = float(abs(contraction_outputs.loss.item() - reconstruction_outputs.loss.item()))
    max_grad_diff = max(gradient_diffs.values()) if gradient_diffs else 0.0

    result = {
        "first_ttlora_module": first_contraction_name,
        "standalone_forward_max_abs_diff": standalone_forward_diff,
        "logits_max_abs_diff": logits_diff,
        "loss_abs_diff": loss_diff,
        "max_tt_core_grad_abs_diff": max_grad_diff,
        "num_tt_core_grad_tensors": len(gradient_diffs),
        "rtol": args.rtol,
        "atol": args.atol,
        "standalone_forward_allclose": tensors_close(
            standalone_contraction, standalone_reconstruction, rtol=args.rtol, atol=args.atol
        ),
        "logits_allclose": tensors_close(contraction_outputs.logits, reconstruction_outputs.logits, rtol=args.rtol, atol=args.atol),
    }
    result["equivalent_within_tolerance"] = (
        result["standalone_forward_allclose"]
        and result["logits_allclose"]
        and result["loss_abs_diff"] <= args.atol
        and result["max_tt_core_grad_abs_diff"] <= args.atol
    )

    output_path = output_dir / f"{args.dataset_name}_equivalence_report.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["equivalent_within_tolerance"]:
        raise SystemExit("Contraction and reconstruction are not equivalent within tolerance.")


if __name__ == "__main__":
    main()
