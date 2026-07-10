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
