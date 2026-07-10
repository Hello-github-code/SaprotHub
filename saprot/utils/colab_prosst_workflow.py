import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from easydict import EasyDict

from saprot.data.pdb2prosst import (
    load_or_quantize_structure,
    serialize_structure_tokens,
)
from saprot.scripts.mutation_zeroshot_prosst import score_mutants
from saprot.scripts.predict_prosst import predict_csv
from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb
from saprot.utils.module_loader import load_trainer, my_load_dataset, my_load_model


MODEL_PROSST_2048 = "AI4Protein/ProSST-2048"


class ColabProSSTWorkflow:
    """Small Colab-facing workflow wrapper for ProSST tasks.

    ProSST always uses amino-acid tokenizer input ids plus official ProSST
    structure token ids. This helper never builds SaProt-style AA+3Di merged
    sequences.
    """

    def __init__(
        self,
        output_dir: str = "/content/colabprosst_outputs",
        upload_dir: str = "/content/prosst_uploads",
        asset_dir: str = "/content/prosst_structure_assets",
        cache_dir: str = "/content/prosst_structure_cache",
        saprothub_dir: str = "/content/SaprotHub",
    ):
        self.output_dir = Path(output_dir)
        self.upload_dir = Path(upload_dir)
        self.asset_dir = Path(asset_dir)
        self.cache_dir = Path(cache_dir)
        self.saprothub_dir = Path(saprothub_dir)
        self.lmdb_dir = self.saprothub_dir / "LMDB"
        self.weight_dir = self.saprothub_dir / "weights" / "prosst"

        for path in [
            self.output_dir,
            self.upload_dir,
            self.asset_dir,
            self.cache_dir,
            self.lmdb_dir,
            self.weight_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.last_structure = None

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _download(path: Path) -> None:
        try:
            from google.colab import files

            files.download(str(path))
        except Exception:
            print("saved:", path)

    def maybe_upload_path(self, current_path: str, upload_enabled: bool) -> str:
        current_path = str(current_path).strip()
        if current_path:
            return current_path
        if not upload_enabled:
            raise ValueError("Set a file path or enable upload.")

        try:
            from google.colab import files
        except Exception as exc:
            raise RuntimeError("Colab file upload is only available in Google Colab.") from exc

        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No file was uploaded.")

        saved_paths = []
        for filename, content in uploaded.items():
            safe_name = Path(filename).name
            save_path = self.upload_dir / safe_name
            save_path.write_bytes(content)
            saved_paths.append(save_path)

        return str(saved_paths[0])

    def maybe_extract_asset_zip(
        self,
        zip_path: str = "",
        upload_enabled: bool = False,
    ) -> Optional[str]:
        zip_path = str(zip_path).strip()
        if not zip_path and not upload_enabled:
            return None
        if not zip_path:
            zip_path = self.maybe_upload_path("", True)

        archive_path = Path(zip_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Structure asset zip does not exist: {archive_path}")

        target_dir = self.asset_dir / archive_path.stem
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_root = target_dir.resolve()

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_target = (target_dir / member.filename).resolve()
                if target_root not in [member_target, *member_target.parents]:
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
                archive.extract(member, target_dir)

        print("extracted structure assets to", target_dir)
        return str(target_dir)

    def create_csv_templates(
        self,
        template_dir: str = "/content/prosst_templates",
        download: bool = False,
    ) -> Path:
        template_home = Path(template_dir)
        template_home.mkdir(parents=True, exist_ok=True)
        for old_template in template_home.glob("prosst_*_template.csv"):
            old_template.unlink()

        pd.DataFrame(
            [
                {"sequence": "ACD", "mutant": "D3A", "structure_tokens": "0 1 2"},
                {"sequence": "ACDE", "mutant": "D3A:E4A", "structure_tokens": "0 1 2 3"},
            ]
        ).to_csv(template_home / "prosst_zero_shot_template.csv", index=False)

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "mutant": "D3A",
                    "pdb_path": "protein.pdb",
                    "chain_id": "",
                }
            ]
        ).to_csv(template_home / "prosst_zero_shot_pdb_template.csv", index=False)

        pd.DataFrame(
            [
                {"sequence": "ACD", "label": 1, "stage": "train", "structure_tokens": "0 1 2"},
                {"sequence": "ACE", "label": 0, "stage": "valid", "structure_tokens": "0 1 3"},
                {"sequence": "ACF", "label": 1, "stage": "test", "structure_tokens": "0 1 4"},
            ]
        ).to_csv(template_home / "prosst_classification_template.csv", index=False)

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 1,
                    "stage": "train",
                    "pdb_path": "train.pdb",
                    "chain_id": "",
                },
                {
                    "sequence": "ACE",
                    "label": 0,
                    "stage": "valid",
                    "pdb_path": "valid.pdb",
                    "chain_id": "",
                },
                {
                    "sequence": "ACF",
                    "label": 1,
                    "stage": "test",
                    "pdb_path": "test.pdb",
                    "chain_id": "",
                },
            ]
        ).to_csv(template_home / "prosst_classification_pdb_template.csv", index=False)

        pd.DataFrame(
            [
                {"sequence": "ACD", "label": 0.5, "stage": "train", "structure_tokens": "0 1 2"},
                {"sequence": "ACE", "label": 0.2, "stage": "valid", "structure_tokens": "0 1 3"},
                {"sequence": "ACF", "label": 0.8, "stage": "test", "structure_tokens": "0 1 4"},
            ]
        ).to_csv(template_home / "prosst_regression_template.csv", index=False)

        pd.DataFrame(
            [
                {
                    "sequence": "ACD",
                    "label": 0.5,
                    "stage": "train",
                    "pdb_path": "train.pdb",
                    "chain_id": "",
                },
                {
                    "sequence": "ACE",
                    "label": 0.2,
                    "stage": "valid",
                    "pdb_path": "valid.pdb",
                    "chain_id": "",
                },
                {
                    "sequence": "ACF",
                    "label": 0.8,
                    "stage": "test",
                    "pdb_path": "test.pdb",
                    "chain_id": "",
                },
            ]
        ).to_csv(template_home / "prosst_regression_pdb_template.csv", index=False)

        pd.DataFrame(
            [
                {"sequence": "ACD", "structure_tokens": "0 1 2"},
                {"sequence": "ACE", "structure_tokens": "0 1 3"},
            ]
        ).to_csv(template_home / "prosst_prediction_template.csv", index=False)

        pd.DataFrame(
            [
                {"sequence": "ACD", "pdb_path": "protein_1.pdb", "chain_id": ""},
                {"sequence": "ACE", "pdb_path": "protein_2.pdb", "chain_id": ""},
            ]
        ).to_csv(template_home / "prosst_prediction_pdb_template.csv", index=False)

        template_zip = template_home / "prosst_csv_templates.zip"
        with zipfile.ZipFile(template_zip, "w") as archive:
            for csv_path in sorted(template_home.glob("*.csv")):
                archive.write(csv_path, arcname=csv_path.name)

        print("template directory:", template_home)
        print("template zip:", template_zip)
        if download:
            self._download(template_zip)

        return template_zip

    def convert_structure(
        self,
        structure_path: str,
        upload_structure: bool = False,
        chain_id: str = "",
        structure_vocab_size: int = 2048,
        output_csv: Optional[str] = None,
        download: bool = True,
    ) -> pd.DataFrame:
        structure_path = self.maybe_upload_path(structure_path, upload_structure)
        chain = chain_id.strip() or None
        result = load_or_quantize_structure(
            structure_path,
            cache_dir=str(self.cache_dir),
            chain_id=chain,
            structure_vocab_size=structure_vocab_size,
        )
        self.last_structure = result

        output_path = Path(output_csv) if output_csv else self.output_dir / "prosst_structure_tokens.csv"
        df = pd.DataFrame(
            [
                {
                    "sequence": result["sequence"],
                    "structure_tokens": serialize_structure_tokens(result["structure_tokens"]),
                    "pdb_path": structure_path,
                    "chain_id": chain or "",
                }
            ]
        )
        df.to_csv(output_path, index=False)

        print("sequence length:", len(result["sequence"]))
        print("structure token length:", len(result["structure_tokens"]))
        print("first 20 tokens:", result["structure_tokens"][:20])
        print("saved structure token csv:", output_path)
        if download:
            self._download(output_path)

        return df

    def attach_last_structure_tokens(self, csv_path: str, output_path: str) -> str:
        if self.last_structure is None:
            raise RuntimeError(
                "Run structure conversion first, or provide structure_tokens/"
                "structure_path/pdb_path in the CSV."
            )

        df = pd.read_csv(csv_path)
        lower_columns = {column.lower(): column for column in df.columns}
        has_structure = any(
            column in lower_columns for column in ["structure_tokens", "structure_path", "pdb_path"]
        )

        if not has_structure:
            sequence_column = lower_columns.get("sequence", lower_columns.get("protein"))
            if sequence_column is None:
                raise ValueError(
                    "CSV needs a sequence/protein column before reusing the last "
                    "structure tokens."
                )

            sequences = df[sequence_column].astype(str).str.strip().str.upper()
            expected_sequence = str(self.last_structure["sequence"]).strip().upper()
            mismatched = sequences[sequences != expected_sequence]
            if not mismatched.empty:
                raise ValueError(
                    "Cannot reuse one structure token sequence for CSV rows with "
                    "different protein sequences. Add per-row structure_tokens/"
                    "structure_path/pdb_path columns."
                )

            df["structure_tokens"] = serialize_structure_tokens(
                self.last_structure["structure_tokens"]
            )

        df.to_csv(output_path, index=False)
        return output_path

    def _prepare_input_csv(
        self,
        input_csv: str,
        upload_csv: bool,
        use_last_structure_tokens: bool,
        suffix: str,
    ) -> str:
        input_csv = self.maybe_upload_path(input_csv, upload_csv)
        if use_last_structure_tokens:
            output_path = self.output_dir / f"prosst_{suffix}_with_structure.csv"
            input_csv = self.attach_last_structure_tokens(input_csv, str(output_path))
        return input_csv

    @staticmethod
    def _validate_training_labels(input_csv: str, task_type: str, num_labels: int) -> None:
        df = pd.read_csv(input_csv)
        lower_columns = {column.lower(): column for column in df.columns}
        label_column = lower_columns.get("label", lower_columns.get("fitness"))
        if label_column is None:
            raise ValueError("Training CSV must contain a label column.")

        if task_type == "classification":
            labels = df[label_column].dropna()
            unique_labels = sorted(labels.astype(int).unique().tolist())
            if len(unique_labels) != int(num_labels):
                raise ValueError(
                    "Classification NUM_LABELS does not match the uploaded dataset: "
                    f"NUM_LABELS={num_labels}, observed_categories={len(unique_labels)}, "
                    f"labels={unique_labels}. If these labels are continuous scores, "
                    "choose the regression workflow instead."
                )
        elif task_type == "regression":
            pd.to_numeric(df[label_column], errors="raise")

    @staticmethod
    def _close_lmdb_datamodule(data_module) -> None:
        for name in ["train_dataset", "valid_dataset", "test_dataset"]:
            dataset = getattr(data_module, name, None)
            if dataset is not None and hasattr(dataset, "_close_lmdb"):
                dataset._close_lmdb()
        if hasattr(data_module, "_close_lmdb"):
            data_module._close_lmdb()

    def run_zero_shot(
        self,
        input_csv: str,
        upload_csv: bool = False,
        use_last_structure_tokens: bool = False,
        structure_zip: str = "",
        upload_structure_zip: bool = False,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: int = 2048,
        output_csv: Optional[str] = None,
        download: bool = True,
    ) -> pd.DataFrame:
        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            use_last_structure_tokens,
            "mutation",
        )
        structure_base_dir = self.maybe_extract_asset_zip(
            structure_zip,
            upload_structure_zip,
        )
        output_path = Path(output_csv) if output_csv else self.output_dir / "prosst_mutation_scores.csv"

        df = score_mutants(
            input_csv=input_csv,
            output_csv=str(output_path),
            model_path=model_path,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=structure_base_dir,
        )
        print("saved mutation scores:", output_path)
        if download:
            self._download(output_path)
        return df

    def train_downstream(
        self,
        task_type: str,
        input_csv: str,
        upload_csv: bool = False,
        use_last_structure_tokens: bool = False,
        structure_zip: str = "",
        upload_structure_zip: bool = False,
        task_name: str = "ProSSTUserTask",
        num_labels: int = 2,
        max_epochs: int = 2,
        batch_size: int = 1,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: int = 2048,
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        load_pretrained: bool = True,
        download: bool = True,
    ) -> dict:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be classification or regression.")

        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            use_last_structure_tokens,
            f"{task_type}_train",
        )
        structure_base_dir = self.maybe_extract_asset_zip(
            structure_zip,
            upload_structure_zip,
        )

        self._validate_training_labels(input_csv, task_type, num_labels)

        construct_prosst_lmdb(
            input_csv,
            str(self.lmdb_dir),
            task_name,
            task_type,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=structure_base_dir,
        )

        model_py = (
            "prosst/prosst_classification_model"
            if task_type == "classification"
            else "prosst/prosst_regression_model"
        )
        dataset_py = (
            "prosst/prosst_classification_dataset"
            if task_type == "classification"
            else "prosst/prosst_regression_dataset"
        )

        checkpoint_path = self.weight_dir / f"{task_name}.pt"
        test_result_csv = self.output_dir / f"{task_name}_{task_type}_test_predictions.csv"
        model_kwargs = {
            "config_path": model_path,
            "load_pretrained": load_pretrained,
            "freeze_backbone": freeze_backbone,
            "gradient_checkpointing": gradient_checkpointing,
            "save_path": str(checkpoint_path),
            "test_result_path": str(test_result_csv),
            "lr_scheduler_kwargs": {"class": "ConstantLRScheduler", "init_lr": 2.0e-5},
            "optimizer_kwargs": {"class": "AdamW", "betas": [0.9, 0.98], "weight_decay": 0.01},
        }
        if task_type == "classification":
            model_kwargs["num_labels"] = num_labels

        config = EasyDict(
            {
                "model": {"model_py_path": model_py, "kwargs": model_kwargs},
                "dataset": {
                    "dataset_py_path": dataset_py,
                    "train_lmdb": str(self.lmdb_dir / task_name / "train"),
                    "valid_lmdb": str(self.lmdb_dir / task_name / "valid"),
                    "test_lmdb": str(self.lmdb_dir / task_name / "test"),
                    "dataloader_kwargs": {"batch_size": batch_size, "num_workers": 0},
                    "kwargs": {
                        "tokenizer": model_path,
                        "structure_vocab_size": structure_vocab_size,
                    },
                },
                "Trainer": {
                    "max_epochs": max_epochs,
                    "log_every_n_steps": 1,
                    "strategy": {"class": "auto"},
                    "logger": False,
                    "enable_checkpointing": False,
                    "val_check_interval": 1.0,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "num_nodes": 1,
                    "accumulate_grad_batches": 1,
                    "precision": 16 if torch.cuda.is_available() else 32,
                    "num_sanity_val_steps": 0,
                },
            }
        )

        model = my_load_model(config.model)
        data_module = my_load_dataset(config.dataset)
        trainer = load_trainer(config)

        try:
            trainer.fit(model=model, datamodule=data_module)

            if checkpoint_path.exists():
                print("loading best checkpoint from", checkpoint_path)
                model.load_checkpoint(str(checkpoint_path))
            else:
                print("best checkpoint was not found; testing the current model state")

            trainer.test(model=model, datamodule=data_module)
        finally:
            self._close_lmdb_datamodule(data_module)

        print("test predictions:", test_result_csv)
        print("model checkpoint:", checkpoint_path)

        if download:
            if test_result_csv.exists():
                self._download(test_result_csv)
            if checkpoint_path.exists():
                self._download(checkpoint_path)

        return {
            "checkpoint_path": str(checkpoint_path),
            "test_result_csv": str(test_result_csv),
            "task_type": task_type,
        }

    def predict_downstream(
        self,
        task_type: str,
        input_csv: str,
        checkpoint_path: str,
        upload_csv: bool = False,
        use_last_structure_tokens: bool = False,
        structure_zip: str = "",
        upload_structure_zip: bool = False,
        num_labels: int = 2,
        batch_size: int = 1,
        model_path: str = MODEL_PROSST_2048,
        structure_vocab_size: int = 2048,
        output_csv: Optional[str] = None,
        download: bool = True,
    ) -> pd.DataFrame:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be classification or regression.")

        input_csv = self._prepare_input_csv(
            input_csv,
            upload_csv,
            use_last_structure_tokens,
            f"{task_type}_predict",
        )
        structure_base_dir = self.maybe_extract_asset_zip(
            structure_zip,
            upload_structure_zip,
        )
        output_path = Path(output_csv) if output_csv else self.output_dir / f"prosst_{task_type}_predictions.csv"

        df = predict_csv(
            input_csv=input_csv,
            output_csv=str(output_path),
            task_type=task_type,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            num_labels=num_labels,
            batch_size=batch_size,
            cache_dir=str(self.cache_dir),
            structure_vocab_size=structure_vocab_size,
            structure_base_dir=structure_base_dir,
        )

        print("saved predictions:", output_path)
        if download:
            self._download(output_path)
        return df

    def upload_checkpoint_to_hf(
        self,
        repo_id: str,
        checkpoint_path: str,
        task_type: str,
        num_labels: int = 2,
        model_path: str = MODEL_PROSST_2048,
        private: bool = False,
        run_login: bool = True,
        title: str = "ColabProSST model",
        description: str = "A ProSST checkpoint trained with ColabProSST.",
        download_package: bool = False,
    ) -> Path:
        if not repo_id.strip():
            raise ValueError("Set repo_id, for example: username/Model-ProSST-Task")
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be classification or regression.")

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

        if run_login:
            from huggingface_hub import notebook_login

            notebook_login()

        package_root = self.saprothub_dir / "model_to_push" / "prosst"
        package_dir = package_root / repo_id.replace("/", "__")
        shutil.rmtree(package_dir, ignore_errors=True)
        package_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(checkpoint, package_dir / "model.pt")
        metadata = {
            "model_family": "ProSST",
            "base_model": model_path,
            "checkpoint_format": "SaprotHub/ColabProSST torch checkpoint",
            "task_type": task_type,
            "input_format": "amino-acid input_ids + ProSST ss_input_ids",
            "structure_vocab_size": 2048,
            "colab_tool": "ColabProSST",
        }
        if task_type == "classification":
            metadata["num_labels"] = int(num_labels)
        (package_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        readme = f"""---
library_name: pytorch
base_model: {model_path}
tags:
- protein-language-model
- ProSST
- ColabProSST
---

# {title}

{description}

This repository contains a SaprotHub/ColabProSST checkpoint (`model.pt`) and metadata for a ProSST downstream model.

## Input Format

ColabProSST uses amino-acid tokenizer `input_ids` together with ProSST structure token `ss_input_ids`. It does not use SaProt AA+3Di merged tokens.

## Task

- Task type: `{task_type}`
- Base model: `{model_path}`

Use `saprot/scripts/predict_prosst.py` from the ColabProSST branch to run prediction with this checkpoint.
"""
        (package_dir / "README.md").write_text(readme, encoding="utf-8")

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(package_dir),
            commit_message="Upload ColabProSST checkpoint",
        )
        print("uploaded to", f"https://huggingface.co/{repo_id}")

        if download_package:
            archive_path = Path(shutil.make_archive(str(package_dir), "zip", package_dir))
            self._download(archive_path)

        return package_dir
