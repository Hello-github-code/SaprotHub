import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

_SAPROT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SAPROT_DIR.parent
for _path in [str(_REPO_ROOT), str(_SAPROT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from saprot.data.pdb2prosst import (
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from saprot.data.prosst_inputs import (
        find_sequence_column,
        structure_entry_from_row,
    )
    from saprot.model.prosst.specs import resolve_structure_vocab_size
    from saprot.scripts.mutation_zeroshot_prosst import score_sequence
except ImportError:
    from data.pdb2prosst import (
        get_structure_tokens_from_entry,
        validate_sequence_and_structure,
    )
    from data.prosst_inputs import (
        find_sequence_column,
        structure_entry_from_row,
    )
    from model.prosst.specs import resolve_structure_vocab_size
    from scripts.mutation_zeroshot_prosst import score_sequence


CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def plot_saturation_heatmap(
    score_matrix: torch.Tensor,
    sequence: str,
    output_png: str,
) -> str:
    import matplotlib.pyplot as plt

    values = score_matrix.detach().float().cpu().numpy()
    max_abs = float(abs(values).max())
    if max_abs == 0:
        max_abs = 1.0

    figure_width = min(40.0, max(8.0, len(sequence) * 0.18))
    figure, axis = plt.subplots(figsize=(figure_width, 7.0))
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axis.set_yticks(range(len(CANONICAL_AMINO_ACIDS)))
    axis.set_yticklabels(CANONICAL_AMINO_ACIDS)
    axis.set_ylabel("Mutant amino acid")

    tick_stride = max(1, math.ceil(len(sequence) / 50))
    tick_positions = list(range(0, len(sequence), tick_stride))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [f"{sequence[index]}{index + 1}" for index in tick_positions],
        rotation=90,
        fontsize=8,
    )
    axis.set_xlabel("Wild-type residue and position")
    axis.set_title("ProSST single-site saturation mutagenesis")
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label("log P(mutant) - log P(wild type)")
    figure.tight_layout()

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


@torch.no_grad()
def score_saturation_mutagenesis(
    input_csv: str,
    output_csv: str,
    output_matrix_csv: str,
    output_heatmap_png: str,
    model_path: str = "AI4Protein/ProSST-2048",
    cache_dir: str = None,
    structure_vocab_size: Optional[int] = None,
    device: str = None,
    structure_base_dir: str = None,
) -> dict:
    structure_vocab_size = resolve_structure_vocab_size(
        model_path,
        structure_vocab_size,
    )
    table = pd.read_csv(input_csv)
    if len(table) != 1:
        raise ValueError(
            "A saturation mutagenesis CSV must contain exactly one protein row."
        )
    sequence_column = find_sequence_column(table.columns)
    row = table.iloc[0]
    sequence = str(row[sequence_column]).strip().upper()
    if not sequence or sequence == "NAN":
        raise ValueError("The saturation mutagenesis sequence is empty.")
    unsupported = sorted(set(sequence) - set(CANONICAL_AMINO_ACIDS))
    if unsupported:
        raise ValueError(
            "Saturation mutagenesis requires canonical amino acids only; "
            f"unsupported residues: {unsupported}."
        )

    entry = structure_entry_from_row(
        row,
        Path(input_csv).resolve().parent,
        structure_base_dir=structure_base_dir,
    )
    structure_tokens = get_structure_tokens_from_entry(
        entry,
        cache_dir=cache_dir,
        structure_vocab_size=structure_vocab_size,
    )
    validate_sequence_and_structure(
        sequence,
        structure_tokens,
        context="saturation protein",
    )

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = model.to(target_device)
    model.eval()
    residue_log_probabilities = score_sequence(
        model,
        tokenizer,
        sequence,
        structure_tokens,
        structure_vocab_size,
        target_device,
    )

    vocabulary = tokenizer.get_vocab()
    missing_amino_acids = [
        amino_acid
        for amino_acid in CANONICAL_AMINO_ACIDS
        if amino_acid not in vocabulary
    ]
    if missing_amino_acids:
        raise ValueError(
            "The selected ProSST tokenizer is missing canonical amino acids: "
            f"{missing_amino_acids}."
        )
    mutant_ids = torch.tensor(
        [vocabulary[amino_acid] for amino_acid in CANONICAL_AMINO_ACIDS],
        dtype=torch.long,
        device=residue_log_probabilities.device,
    )
    wild_type_ids = torch.tensor(
        [vocabulary[amino_acid] for amino_acid in sequence],
        dtype=torch.long,
        device=residue_log_probabilities.device,
    )
    mutant_log_probabilities = residue_log_probabilities[:, mutant_ids]
    wild_type_log_probabilities = residue_log_probabilities.gather(
        1,
        wild_type_ids.unsqueeze(1),
    )
    score_matrix = (
        mutant_log_probabilities - wild_type_log_probabilities
    ).transpose(0, 1).detach().float().cpu()
    if not torch.isfinite(score_matrix).all():
        raise ValueError("ProSST saturation scores contain non-finite values.")

    records = []
    for position, wild_type in enumerate(sequence, start=1):
        for mutant_index, mutant in enumerate(CANONICAL_AMINO_ACIDS):
            records.append(
                {
                    "position": position,
                    "wild_type": wild_type,
                    "mutant": mutant,
                    "mutation": f"{wild_type}{position}{mutant}",
                    "score": score_matrix[mutant_index, position - 1].item(),
                    "is_wild_type": mutant == wild_type,
                }
            )
    score_table = pd.DataFrame(records)

    matrix_table = pd.DataFrame(
        score_matrix.numpy(),
        columns=[
            f"{wild_type}{position}"
            for position, wild_type in enumerate(sequence, start=1)
        ],
    )
    matrix_table.insert(0, "mutant", CANONICAL_AMINO_ACIDS)

    output_path = Path(output_csv)
    matrix_path = Path(output_matrix_csv)
    for path in [output_path, matrix_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(output_path, index=False)
    matrix_table.to_csv(matrix_path, index=False)
    plot_saturation_heatmap(
        score_matrix,
        sequence,
        output_heatmap_png,
    )

    return {
        "sequence": sequence,
        "score_table": score_table,
        "matrix_table": matrix_table,
        "score_matrix": score_matrix,
        "output_csv": str(output_path),
        "output_matrix_csv": str(matrix_path),
        "output_heatmap_png": str(Path(output_heatmap_png)),
        "model_path": model_path,
        "structure_vocab_size": int(structure_vocab_size),
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_matrix_csv", required=True)
    parser.add_argument("--output_heatmap_png", required=True)
    parser.add_argument("--model_path", default="AI4Protein/ProSST-2048")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--structure_vocab_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure_base_dir", default=None)
    return parser.parse_args()


def main():
    args = get_args()
    result = score_saturation_mutagenesis(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_matrix_csv=args.output_matrix_csv,
        output_heatmap_png=args.output_heatmap_png,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        structure_vocab_size=args.structure_vocab_size,
        device=args.device,
        structure_base_dir=args.structure_base_dir,
    )
    print("saved saturation scores:", result["output_csv"])
    print("saved saturation matrix:", result["output_matrix_csv"])
    print("saved saturation heatmap:", result["output_heatmap_png"])


if __name__ == "__main__":
    main()
