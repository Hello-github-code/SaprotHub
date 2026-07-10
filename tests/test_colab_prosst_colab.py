import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "colab" / "ColabProSST.ipynb"
UI_PATH = REPO_ROOT / "saprot" / "utils" / "colab_prosst_ui.py"


class ColabProSSTNotebookTest(unittest.TestCase):
    def test_notebook_uses_one_live_interface_cell(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len(notebook["cells"]), 2)
        self.assertEqual(notebook["cells"][0]["cell_type"], "markdown")
        self.assertEqual(notebook["cells"][1]["cell_type"], "code")

        source = "".join(notebook["cells"][1]["source"])
        tree = ast.parse(source)
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("ColabProSSTUI(COLABPROSST_WORKFLOW)", source)
        self.assertIn("COLABPROSST_UI.launch()", source)
        self.assertIn("probe_runtime()", source)
        self.assertNotIn("WORKFLOW", assigned_names)
        self.assertNotIn("os.environ['TRANSFORMERS_CACHE']", source)

    def test_notebook_checks_both_source_checkouts(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][1]["source"])

        self.assertIn("saprot/utils/colab_prosst_ui.py", source)
        self.assertIn("prosst/structure/get_sst_seq.py", source)
        self.assertIn("prosst/structure/static/AE.pt", source)
        self.assertIn("prosst/structure/static/2048.joblib", source)


class ColabProSSTWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas

        from saprot.utils.colab_prosst_workflow import ColabProSSTWorkflow

        cls.pd = pandas
        cls.workflow_class = ColabProSSTWorkflow

    def test_training_learning_rate_reaches_model_config(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "training.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "label": 0,
                        "stage": "train",
                        "structure_tokens": "0 1 2",
                    },
                    {
                        "sequence": "ACE",
                        "label": 1,
                        "stage": "valid",
                        "structure_tokens": "0 1 3",
                    },
                    {
                        "sequence": "ACF",
                        "label": 0,
                        "stage": "test",
                        "structure_tokens": "0 1 4",
                    },
                ]
            ).to_csv(csv_path, index=False)

            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )
            captured = {}

            class DummyModel:
                pass

            class DummyDataModule:
                pass

            class DummyTrainer:
                def fit(self, model, datamodule):
                    captured["fit"] = (model, datamodule)

                def test(self, model, datamodule):
                    captured["test"] = (model, datamodule)

            def load_model(config):
                captured["model_config"] = config
                return DummyModel()

            with patch(
                "saprot.utils.colab_prosst_workflow.construct_prosst_lmdb"
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_model",
                side_effect=load_model,
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_dataset",
                return_value=DummyDataModule(),
            ), patch(
                "saprot.utils.colab_prosst_workflow.load_trainer",
                return_value=DummyTrainer(),
            ):
                workflow.train_downstream(
                    task_type="classification",
                    input_csv=str(csv_path),
                    task_name="learning-rate-test",
                    num_labels=2,
                    max_epochs=1,
                    learning_rate=3.0e-5,
                    load_pretrained=False,
                    download=False,
                )

            scheduler = captured["model_config"].kwargs.lr_scheduler_kwargs
            self.assertEqual(scheduler["class"], "ConstantLRScheduler")
            self.assertEqual(scheduler["init_lr"], 3.0e-5)
            self.assertIn("fit", captured)
            self.assertIn("test", captured)

    def test_classification_category_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "labels.csv"
            self.pd.DataFrame({"label": [0, 1, 2, 3, 4]}).to_csv(
                csv_path, index=False
            )

            with self.assertRaisesRegex(
                ValueError,
                "NUM_LABELS=2, observed_categories=5",
            ):
                self.workflow_class._validate_training_labels(
                    str(csv_path), "classification", 2
                )

    def test_classification_labels_must_start_at_zero(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "labels.csv"
            self.pd.DataFrame({"label": [1, 2]}).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "expected=\[0, 1\], observed=\[1, 2\]",
            ):
                self.workflow_class._validate_training_labels(
                    str(csv_path), "classification", 2
                )

    def test_task_name_cannot_escape_workflow_directories(self):
        for task_name in ["", ".", "..", "../outside", "folder\\outside"]:
            with self.subTest(task_name=task_name):
                with self.assertRaises(ValueError):
                    self.workflow_class._validate_task_name(task_name)

        self.assertEqual(
            self.workflow_class._validate_task_name("My-ProSST_Task.1"),
            "My-ProSST_Task.1",
        )


class ColabProSSTInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas
        import torch

        from saprot.scripts import mutation_zeroshot_prosst, predict_prosst

        cls.pd = pandas
        cls.torch = torch
        cls.mutation = mutation_zeroshot_prosst
        cls.prediction = predict_prosst

    class FakeTokenizer:
        vocab = {"A": 3, "C": 4, "D": 5, "E": 6}

        def get_vocab(self):
            return self.vocab

        def __call__(self, sequences, return_tensors="pt"):
            rows = []
            for sequence in sequences:
                rows.append([1, *[self.vocab[aa] for aa in sequence], 2])
            return {
                "input_ids": ColabProSSTInferenceTest.torch.tensor(rows),
                "attention_mask": ColabProSSTInferenceTest.torch.ones(
                    (len(rows), len(rows[0])), dtype=ColabProSSTInferenceTest.torch.long
                ),
            }

        def batch_encode_plus(
            self,
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=None,
        ):
            encoded = [
                [1, *[self.vocab[aa] for aa in sequence], 2]
                for sequence in sequences
            ]
            target_length = max(len(row) for row in encoded)
            rows = [row + [0] * (target_length - len(row)) for row in encoded]
            masks = [
                [1] * len(row) + [0] * (target_length - len(row))
                for row in encoded
            ]
            return {
                "input_ids": ColabProSSTInferenceTest.torch.tensor(rows),
                "attention_mask": ColabProSSTInferenceTest.torch.tensor(masks),
            }

    class FakeMaskedLM:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, **_kwargs):
            torch = ColabProSSTInferenceTest.torch
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 25))
            logits[:, 1, 3] = 1.0
            logits[:, 1, 4] = 2.0
            logits[:, 3, 3] = 4.0
            logits[:, 3, 5] = 1.0
            return types.SimpleNamespace(logits=logits)

    def test_zero_shot_score_uses_log_probability_differences(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "mutations.csv"
            output_csv = root / "scores.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "mutant": "A1C:D3A",
                        "structure_tokens": "0 1 2",
                    }
                ]
            ).to_csv(input_csv, index=False)

            with patch.object(
                self.mutation.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.mutation.AutoModelForMaskedLM,
                "from_pretrained",
                return_value=self.FakeMaskedLM(),
            ):
                result = self.mutation.score_mutants(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    device="cpu",
                )

            self.assertAlmostEqual(result.loc[0, "score"], 4.0, places=5)
            self.assertTrue(output_csv.exists())

    def test_prediction_writes_class_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "prediction.csv"
            output_csv = root / "predictions.csv"
            checkpoint = root / "model.pt"
            checkpoint.touch()
            self.pd.DataFrame(
                [
                    {"sequence": "ACD", "structure_tokens": "0 1 2"},
                    {"sequence": "ACE", "structure_tokens": "3 4 5"},
                ]
            ).to_csv(input_csv, index=False)

            class FakePredictionModel:
                def forward(inner_self, inputs):
                    batch_size = inputs["input_ids"].shape[0]
                    return self.torch.tensor([[0.0, 2.0], [3.0, 1.0]])[
                        :batch_size
                    ]

            with patch.object(
                self.prediction.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.prediction,
                "_load_model",
                return_value=FakePredictionModel(),
            ):
                result = self.prediction.predict_csv(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    task_type="classification",
                    checkpoint_path=str(checkpoint),
                    num_labels=2,
                    batch_size=2,
                    device="cpu",
                )

            self.assertEqual(result["pred"].tolist(), [1, 0])
            self.assertIn("prob_0", result.columns)
            self.assertIn("prob_1", result.columns)
            self.assertTrue(output_csv.exists())

    def test_prediction_rejects_invalid_batch_size_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 1"):
            self.prediction.predict_csv(
                input_csv="unused.csv",
                output_csv="unused-output.csv",
                task_type="classification",
                checkpoint_path="unused.pt",
                batch_size=0,
            )


@unittest.skipUnless(
    importlib.util.find_spec("ipywidgets") is not None,
    "ipywidgets is installed only in the Colab UI runtime",
)
class ColabProSSTWidgetTest(unittest.TestCase):
    def test_every_interface_page_constructs(self):
        fake_saprot = types.ModuleType("saprot")
        fake_saprot.__path__ = []
        fake_utils = types.ModuleType("saprot.utils")
        fake_utils.__path__ = []
        fake_workflow = types.ModuleType("saprot.utils.colab_prosst_workflow")
        fake_workflow.MODEL_PROSST_2048 = "AI4Protein/ProSST-2048"

        module_name = "_colab_prosst_ui_widget_test"
        spec = importlib.util.spec_from_file_location(module_name, UI_PATH)
        module = importlib.util.module_from_spec(spec)
        replacements = {
            "saprot": fake_saprot,
            "saprot.utils": fake_utils,
            "saprot.utils.colab_prosst_workflow": fake_workflow,
            module_name: module,
        }

        with patch.dict(sys.modules, replacements):
            spec.loader.exec_module(module)

        class DummyWorkflow:
            def maybe_upload_path(self, current_path, upload_enabled):
                return "/tmp/uploaded.file"

        ui = module.ColabProSSTUI(DummyWorkflow())
        rendered = []
        ui.display = lambda *items: rendered.append(items)
        ui.clear_output = lambda **_kwargs: None

        pages = [
            ui._home_page,
            ui._training_page,
            ui._prediction_menu_page,
            ui._property_prediction_page,
            ui._mutation_page,
            ui._structure_page,
            ui._share_page,
        ]
        for page in pages:
            with self.subTest(page=page.__name__):
                rendered.clear()
                page()
                self.assertTrue(rendered)


if __name__ == "__main__":
    unittest.main()
