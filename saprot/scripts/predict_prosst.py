import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

_SAPROT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SAPROT_DIR.parent
for _path in [str(_REPO_ROOT), str(_SAPROT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from saprot.data.pdb2prosst import (
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        pad_structure_input_ids,
        validate_sequence_and_structure,
    )
    from saprot.model.prosst.prosst_classification_model import ProSSTClassificationModel
    from saprot.model.prosst.prosst_regression_model import ProSSTRegressionModel
    from saprot.model.prosst.specs import resolve_structure_vocab_size
except ImportError:
    from data.pdb2prosst import (
        encode_structure_tokens,
        get_structure_tokens_from_entry,
        pad_structure_input_ids,
        validate_sequence_and_structure,
    )
    from model.prosst.prosst_classification_model import ProSSTClassificationModel
    from model.prosst.prosst_regression_model import ProSSTRegressionModel
    from model.prosst.specs import resolve_structure_vocab_size


def _has_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _sequence_column(columns: Sequence[str]) -> str:
    lower_columns = {column.lower(): column for column in columns}
    if "sequence" in lower_columns:
        return lower_columns["sequence"]
    if "protein" in lower_columns:
        return lower_columns["protein"]
    raise ValueError("ProSST prediction CSV must contain `sequence` or `protein`.")


def _row_structure_entry(
    row: pd.Series,
    csv_dir: Path,
    structure_base_dir: str = None,
) -> Dict[str, Any]:
    lower_columns = {column.lower(): column for column in row.index}
    structure_column = lower_columns.get("structure_tokens")
    if structure_column is not None and _has_value(row[structure_column]):
        entry = {"structure_tokens": row[structure_column]}
        vocab_column = lower_columns.get("structure_vocab_size")
        if vocab_column is not None and _has_value(row[vocab_column]):
            entry["structure_vocab_size"] = row[vocab_column]
        return entry

    path_column = None
    for name in ["structure_path", "pdb_path"]:
        if name in lower_columns and _has_value(row[lower_columns[name]]):
            path_column = lower_columns[name]
            break
    if path_column is None:
        raise ValueError(
            "Each ProSST prediction row needs structure_tokens, structure_path, "
            "or pdb_path."
        )

    structure_path = Path(str(row[path_column]).strip())
    if not structure_path.is_absolute():
        candidates = [csv_dir / structure_path]
        if structure_base_dir is not None and str(structure_base_dir).strip():
            candidates.append(Path(structure_base_dir) / structure_path)
        structure_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])

    entry = {"pdb_path": str(structure_path)}
    for name in ["chain_id", "chain"]:
        column = lower_columns.get(name)
        if column is not None and _has_value(row[column]):
            entry["chain_id"] = str(row[column]).strip()
            break
    vocab_column = lower_columns.get("structure_vocab_size")
    if vocab_column is not None and _has_value(row[vocab_column]):
        entry["structure_vocab_size"] = row[vocab_column]

    return entry


def _prepare_batch(
    tokenizer,
    sequences: Sequence[str],
    structure_tokens_list: Sequence[Sequence[int]],
    max_length: int,
    structure_vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    sequences = [sequence[:max_length] for sequence in sequences]
    structure_tokens_list = [tokens[:max_length] for tokens in structure_tokens_list]

    inputs = tokenizer.batch_encode_plus(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length + 2,
    )
    target_length = inputs["input_ids"].shape[1]
    ss_ids = [
        encode_structure_tokens(tokens, structure_vocab_size)
        for tokens in structure_tokens_list
    ]
    inputs["ss_input_ids"] = torch.tensor(
        pad_structure_input_ids(ss_ids, target_length),
        dtype=torch.long,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _load_model(
    task_type: str,
    model_path: str,
    checkpoint_path: str,
    num_labels: int,
    structure_vocab_size: int,
    device: torch.device,
    load_pretrained: bool = False,
):
    common_kwargs = {
        "config_path": model_path,
        "structure_vocab_size": structure_vocab_size,
        "from_checkpoint": checkpoint_path,
        "load_pretrained": load_pretrained,
        "lr_scheduler_kwargs": {
            "class": "ConstantLRScheduler",
            "init_lr": 0.0,
        },
        "optimizer_kwargs": {
            "class": "AdamW",
            "betas": [0.9, 0.98],
            "weight_decay": 0.01,
        },
    }
    if task_type == "classification":
        model = ProSSTClassificationModel(
            num_labels=num_labels,
            **common_kwargs,
        )
    elif task_type == "regression":
        model = ProSSTRegressionModel(**common_kwargs)
    else:
        raise ValueError("task_type must be `classification` or `regression`.")

    model = model.to(device)
    model.eval()
    return model


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    task_type: str,
    model_path: str,
    structure_vocab_size: int,
) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        # The regular model loader will report unreadable or invalid checkpoints.
        return

    if not isinstance(checkpoint, dict):
        return
    metadata = checkpoint.get("colabprosst")
    if not isinstance(metadata, dict):
        return

    expected = {
        "task": task_type,
        "base_model": model_path,
        "structure_vocab_size": int(structure_vocab_size),
    }
    mismatches = [
        f"{key}={metadata[key]!r} (checkpoint), expected {value!r}"
        for key, value in expected.items()
        if key in metadata and metadata[key] != value
    ]
    if mismatches:
        raise ValueError(
            "The ProSST checkpoint is incompatible with the selected settings: "
            + "; ".join(mismatches)
            + ". Select the checkpoint's original base model and task."
        )


@torch.no_grad()
def predict_csv(
    input_csv: str,
    output_csv: str,
    task_type: str,
    checkpoint_path: str,
    model_path: str = "AI4Protein/ProSST-2048",
    num_labels: int = 2,
    batch_size: int = 1,
    cache_dir: str = None,
    structure_vocab_size: Optional[int] = None,
    max_length: int = 2046,
    device: str = None,
    structure_base_dir: str = None,
    load_pretrained: bool = False,
) -> pd.DataFrame:
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be `classification` or `regression`.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if task_type == "classification" and num_labels < 2:
        raise ValueError("Classification num_labels must be at least 2.")
    structure_vocab_size = resolve_structure_vocab_size(
        model_path,
        structure_vocab_size,
    )
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for ProSST prediction.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"ProSST checkpoint does not exist: {checkpoint}")
    validate_checkpoint_compatibility(
        str(checkpoint),
        task_type,
        model_path,
        structure_vocab_size,
    )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("ProSST prediction CSV contains no rows.")
    sequence_column = _sequence_column(df.columns)
    csv_dir = Path(input_csv).resolve().parent

    sequences: List[str] = []
    structure_tokens_list: List[List[int]] = []
    for row_idx, row in df.iterrows():
        sequence = str(row[sequence_column]).strip().upper()
        if not sequence or sequence == "NAN":
            raise ValueError(f"ProSST prediction row {row_idx} has an empty sequence.")
        entry = _row_structure_entry(
            row,
            csv_dir,
            structure_base_dir=structure_base_dir,
        )
        structure_tokens = get_structure_tokens_from_entry(
            entry,
            cache_dir=cache_dir,
            structure_vocab_size=structure_vocab_size,
        )
        validate_sequence_and_structure(sequence, structure_tokens, context=f"row {row_idx}")
        sequences.append(sequence)
        structure_tokens_list.append([int(token) for token in structure_tokens])

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = _load_model(
        task_type,
        model_path,
        checkpoint_path,
        num_labels,
        structure_vocab_size,
        device,
        load_pretrained=load_pretrained,
    )

    output_chunks = []
    for start in range(0, len(sequences), batch_size):
        stop = start + batch_size
        batch_inputs = _prepare_batch(
            tokenizer,
            sequences[start:stop],
            structure_tokens_list[start:stop],
            max_length=max_length,
            structure_vocab_size=structure_vocab_size,
            device=device,
        )
        logits = model.forward(batch_inputs)
        output_chunks.append(logits.detach().cpu())

    outputs = torch.cat(output_chunks, dim=0)
    result = df.copy()
    if task_type == "classification":
        probabilities = torch.softmax(outputs, dim=-1)
        result["pred"] = probabilities.argmax(dim=-1).numpy()
        for idx in range(probabilities.shape[-1]):
            result[f"prob_{idx}"] = probabilities[:, idx].numpy()
    else:
        result["pred"] = outputs.reshape(-1).numpy()

    result.to_csv(output_csv, index=False)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["classification", "regression"],
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2046)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure_base_dir", default=None)
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help=(
            "Load the full base ProSST weights before applying the checkpoint. "
            "By default prediction builds the model from config because "
            "ColabProSST checkpoints contain the full model state dict."
        ),
    )
    return parser.parse_args()


def main():
    args = get_args()
    predict_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        task_type=args.task_type,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        max_length=args.max_length,
        device=args.device,
        structure_base_dir=args.structure_base_dir,
        load_pretrained=args.load_pretrained,
    )


if __name__ == "__main__":
    main()
