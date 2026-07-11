import ast
import base64
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "colab" / "ColabProSST.ipynb"
COLABSAPROT_PATH = REPO_ROOT / "colab" / "SaprotHub_v2.ipynb"
UI_PATH = REPO_ROOT / "saprot" / "utils" / "colab_prosst_ui.py"
README_PATH = REPO_ROOT / "README.md"


class ColabProSSTNotebookTest(unittest.TestCase):
    def test_notebook_uses_one_live_interface_cell(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len(notebook["cells"]), 3)
        self.assertEqual(notebook["cells"][0]["cell_type"], "markdown")
        self.assertEqual(notebook["cells"][1]["cell_type"], "markdown")
        self.assertEqual(notebook["cells"][2]["cell_type"], "code")

        source = "".join(notebook["cells"][2]["source"])
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
        self.assertIn("import ipywidgets; import jupyter_ui_poll", source)
        self.assertIn("Runtime > Manage sessions", source)
        self.assertIn("run_button.png", source)
        self.assertIn("run_button_working.png", source)
        self.assertIn("warnings.filterwarnings('ignore', category=FutureWarning)", source)
        self.assertIn("os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'", source)
        self.assertNotIn("#@param", source)
        self.assertNotIn("DOWNLOAD_CSV_TEMPLATES", source)
        self.assertNotIn("WORKFLOW", assigned_names)
        self.assertNotIn("os.environ['TRANSFORMERS_CACHE']", source)

        introduction = "".join(notebook["cells"][0]["source"])
        self.assertIn("Prepare sequence and structure inputs", introduction)
        self.assertIn("Recommended for a first run", introduction)
        self.assertIn("CSV already contains `structure_tokens`", introduction)
        self.assertIn("CSV does not contain `structure_tokens`", introduction)
        self.assertIn("Reuse latest structure conversion", introduction)

        tutorial = "".join(notebook["cells"][1]["source"])
        self.assertIn("How to start", tutorial)
        self.assertIn("youtube.com/watch?v=nmLtjlCI_7M", tutorial)
        self.assertIn("Switch_Runtime_2.png", tutorial)
        self.assertIn("to run ColabProSST", tutorial)
        self.assertIn("T4 GPU", tutorial)
        self.assertIn("L4 GPU", tutorial)
        self.assertIn("A100 GPU", tutorial)
        self.assertIn("ProSST backbone frozen", tutorial)
        self.assertNotIn("ColabSeprot", tutorial)
        self.assertNotIn("ProTrek", tutorial)

    def test_readme_links_to_the_current_prosst_notebook(self):
        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "colab.research.google.com/github/Hello-github-code/SaprotHub/"
            "blob/prosst/colab/ColabProSST.ipynb",
            readme,
        )
        self.assertIn("Old notebooks saved in Google Drive", readme)

    def test_notebook_checks_both_source_checkouts(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])

        self.assertIn("saprot/utils/colab_prosst_ui.py", source)
        self.assertIn("prosst/structure/get_sst_seq.py", source)
        self.assertIn("prosst/structure/static/AE.pt", source)
        self.assertIn("prosst/structure/static/2048.joblib", source)

    def test_notebook_anchors_and_refreshes_colab_source_checkout(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][2]["source"])

        self.assertIn("Path('/content')", source)
        self.assertIn("os.chdir(ROOT)", source)
        self.assertNotIn("ROOT = Path(os.getcwd())", source)
        self.assertIn("def update_saprothub():", source)
        self.assertIn("'fetch', '--depth', '50'", source)
        self.assertIn("'.SaprotHub-installing'", source)
        self.assertIn("'.ProSST-installing'", source)
        self.assertIn("def project_revision(home=SAPROT_HOME):", source)
        self.assertIn("ColabProSST source:", source)
        self.assertIn("Official ProSST source:", source)
        self.assertIn("module_name.startswith('prosst.')", source)
        self.assertNotIn("load_colabprosst_workflow", source)

    def test_home_menu_matches_colabsaprot_top_level_actions(self):
        source = UI_PATH.read_text(encoding="utf-8")
        home_source = source.split("def _home_page(self):", 1)[1].split(
            "def _training_page(self):", 1
        )[0]
        prediction_source = source.split(
            "def _prediction_menu_page(self):", 1
        )[1].split("def _property_prediction_page(self):", 1)[0]
        guide_source = source.split("def _input_guide(self):", 1)[1].split(
            "def _build_system_widgets(self):", 1
        )[0]

        self.assertIn("I want to train my own model", home_source)
        self.assertIn(
            "I want to use existing models to make prediction", home_source
        )
        self.assertIn("I want to share my model publicly", home_source)
        self.assertIn("self._input_guide()", home_source)
        self.assertIn("Recommended for a first run", guide_source)
        self.assertIn("CSV contains structure_tokens", guide_source)
        self.assertIn("CSV contains structure file paths", guide_source)
        self.assertIn('width="100%"', guide_source)
        self.assertIn("max_width=self.GUIDE_WIDTH", guide_source)
        self.assertNotIn("convert a protein structure", home_source)
        self.assertNotIn("Download CSV templates", home_source)
        self.assertIn("Convert protein structure to ProSST tokens", prediction_source)
        self.assertIn("Download CSV templates", prediction_source)

    def test_shared_interface_copy_is_kept_in_sync_with_colabsaprot(self):
        reference_notebook = json.loads(
            COLABSAPROT_PATH.read_text(encoding="utf-8")
        )
        reference = "".join(reference_notebook["cells"][2]["source"])
        prosst = UI_PATH.read_text(encoding="utf-8")
        shared_copy = [
            "I want to train my own model",
            "I want to use existing models to make prediction",
            "I want to share my model publicly",
            "Please finish the setting of your training task",
            "Task setting:",
            "Name your task:",
            "Task type:",
            "Number of categories:",
            "Model setting:",
            "Base model:",
            "Dataset setting:",
            "Training hyper-parameters:",
            "Batch size:",
            "Epoch:",
            "Learning rate:",
            "Start training",
            "Protein property prediction",
            "Choose the prediction task:",
            "Start prediction",
            "Mutational effect prediction",
            "Go back",
            "Refresh",
            "Stop",
        ]

        for copy in shared_copy:
            with self.subTest(copy=copy):
                self.assertIn(copy, reference)
                self.assertIn(copy, prosst)

        self.assertIn("Convert protein structure to ProSST tokens", prosst)
        self.assertIn("Structure input:", prosst)

    def test_task_pages_require_an_explicit_structure_input_mode(self):
        source = UI_PATH.read_text(encoding="utf-8")

        self.assertEqual(source.count("structure_input = _StructureInput(self)"), 3)
        self.assertIn("CSV contains structure_tokens", source)
        self.assertIn("Reuse latest structure conversion", source)
        self.assertIn("CSV contains structure file paths", source)
        self.assertIn("structure_input.reuse_latest", source)
        self.assertIn("structure_input.structure_zip", source)
        self.assertNotIn(
            'description="Reuse tokens from the latest structure conversion"',
            source,
        )
        structure_page = source.split("def _structure_page(self):", 1)[1].split(
            "def _share_page(self):", 1
        )[0]
        self.assertIn("Use these tokens in your next task", structure_page)
        self.assertIn("Reuse latest structure conversion", structure_page)
        self.assertIn("lasts only for this running Colab session", structure_page)

        training_page = source.split("def _training_page(self):", 1)[1].split(
            "def _prediction_menu_page(self):", 1
        )[0]
        self.assertIn("The training is completed. You can then", training_page)
        self.assertIn("is selected automatically in this session", training_page)
        self.assertIn("self._task_intro(change[\"new\"])", training_page)

    def test_background_tasks_never_clear_the_whole_cell(self):
        source = UI_PATH.read_text(encoding="utf-8")
        start_task_source = source.split("def _start_task(self,", 1)[1].split(
            "def stop_task(self,", 1
        )[0]
        stop_task_source = source.split("def stop_task(self,", 1)[1].split(
            "def _download_templates(self,", 1
        )[0]

        self.assertIn("output.clear_output(wait=True)", start_task_source)
        self.assertNotIn("self.clear_output", start_task_source)
        self.assertIn(
            "self.system_status.clear_output(wait=True)", stop_task_source
        )
        self.assertNotIn("self.clear_output", stop_task_source)

    def test_navigation_uses_page_history_instead_of_a_hardcoded_home(self):
        source = UI_PATH.read_text(encoding="utf-8")
        navigation_source = source.split(
            "def _update_navigation_controls(self):", 1
        )[1].split("def _start_task(self,", 1)[0]

        self.assertIn(
            "self.navigation_history.append(previous_page)", navigation_source
        )
        self.assertIn(
            "previous_page = self.navigation_history.pop()", navigation_source
        )
        self.assertIn("remember=False", navigation_source)
        self.assertNotIn("_navigate(self._home_page)", source)
        self.assertIn("return to the previous", source)


@unittest.skipUnless(shutil.which("git"), "git is required for bootstrap tests")
class ColabProSSTBootstrapTest(unittest.TestCase):
    @staticmethod
    def _load_bootstrap_functions(root, repo_url):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        tree = ast.parse("".join(notebook["cells"][2]["source"]))
        function_names = {
            "checkout_complete",
            "run_command",
            "clone_saprothub",
            "update_saprothub",
            "project_revision",
            "ensure_official_prosst",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        ]
        if {node.name for node in functions} != function_names:
            raise AssertionError("Could not extract every bootstrap function")

        namespace = {
            "Path": Path,
            "ROOT": root,
            "SAPROT_HOME": root / "SaprotHub",
            "SAPROT_REQUIRED": [Path("required.txt")],
            "SAPROTHUB_REPO": repo_url,
            "SAPROTHUB_BRANCH": "prosst",
            "PROSST_HOME": root / "ProSST",
            "PROSST_REQUIRED": [Path("required.txt")],
            "PROSST_REPO": repo_url,
            "shutil": shutil,
            "subprocess": subprocess,
        }
        module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
        return namespace

    @staticmethod
    def _git(*args, cwd=None):
        return subprocess.check_output(
            ["git", *map(str, args)],
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    def test_checkout_clone_update_and_failed_reclone_are_safe(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            remote = root / "remote.git"
            source = root / "source"
            runtime = root / "runtime"
            runtime.mkdir()

            self._git("init", "--bare", remote)
            self._git("init", "-b", "prosst", source)
            self._git("config", "user.email", "test@example.com", cwd=source)
            self._git("config", "user.name", "ColabProSST Test", cwd=source)
            (source / "required.txt").write_text("version 1", encoding="utf-8")
            self._git("add", "required.txt", cwd=source)
            self._git("commit", "-m", "version 1", cwd=source)
            self._git("remote", "add", "origin", remote.as_uri(), cwd=source)
            self._git("push", "-u", "origin", "prosst", cwd=source)
            self._git(
                "--git-dir",
                remote,
                "symbolic-ref",
                "HEAD",
                "refs/heads/prosst",
            )

            bootstrap = self._load_bootstrap_functions(runtime, remote.as_uri())
            bootstrap["clone_saprothub"]()
            checkout = runtime / "SaprotHub"
            self.assertEqual(
                (checkout / "required.txt").read_text(encoding="utf-8"),
                "version 1",
            )
            bootstrap["ensure_official_prosst"]()
            official_checkout = runtime / "ProSST"
            self.assertEqual(
                (official_checkout / "required.txt").read_text(encoding="utf-8"),
                "version 1",
            )

            (source / "required.txt").write_text("version 2", encoding="utf-8")
            self._git("add", "required.txt", cwd=source)
            self._git("commit", "-m", "version 2", cwd=source)
            self._git("push", cwd=source)
            expected_revision = self._git("rev-parse", "--short", "HEAD", cwd=source)

            bootstrap["update_saprothub"]()
            self.assertEqual(
                (checkout / "required.txt").read_text(encoding="utf-8"),
                "version 2",
            )
            self.assertEqual(bootstrap["project_revision"](), expected_revision)

            bootstrap["SAPROTHUB_REPO"] = (root / "missing.git").as_uri()
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap["clone_saprothub"]()
            self.assertEqual(
                (checkout / "required.txt").read_text(encoding="utf-8"),
                "version 2",
            )

            bootstrap["PROSST_REPO"] = (root / "missing.git").as_uri()
            bootstrap["PROSST_REQUIRED"] = [Path("new-required.txt")]
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap["ensure_official_prosst"]()
            self.assertEqual(
                (official_checkout / "required.txt").read_text(encoding="utf-8"),
                "version 1",
            )


class ColabProSSTStructureRuntimeTest(unittest.TestCase):
    def test_threadpool_guard_ignores_only_deleted_library_paths(self):
        import threadpoolctl

        from saprot.data.pdb2prosst import (
            _patch_threadpoolctl_stale_library_scan,
        )

        controller_class = threadpoolctl.ThreadpoolController
        method_name = "_make_controller_from_path"
        marker_name = "_colabprosst_stale_library_guard"
        original = getattr(controller_class, method_name)
        had_marker = hasattr(controller_class, marker_name)
        marker_value = getattr(controller_class, marker_name, None)

        def raise_missing_library(_controller, _filepath):
            raise OSError("cannot open shared object file")

        try:
            setattr(controller_class, method_name, raise_missing_library)
            if hasattr(controller_class, marker_name):
                delattr(controller_class, marker_name)

            _patch_threadpoolctl_stale_library_scan()
            guarded = getattr(controller_class, method_name)
            controller = object.__new__(controller_class)

            missing_path = str(
                Path(tempfile.gettempdir()) / "deleted-colab-numpy-openblas.so"
            )
            self.assertFalse(Path(missing_path).exists())
            self.assertIsNone(guarded(controller, missing_path))

            with tempfile.NamedTemporaryFile() as existing_library:
                with self.assertRaises(OSError):
                    guarded(controller, existing_library.name)
        finally:
            setattr(controller_class, method_name, original)
            if had_marker:
                setattr(controller_class, marker_name, marker_value)
            elif hasattr(controller_class, marker_name):
                delattr(controller_class, marker_name)


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

    def test_uploaded_content_is_saved_with_a_safe_filename(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )

            saved_path = Path(
                workflow.save_uploaded_content("../training.csv", b"a,b\n1,2\n")
            )
            self.assertEqual(saved_path.parent, root / "uploads")
            self.assertEqual(saved_path.name, "training.csv")
            self.assertEqual(saved_path.read_bytes(), b"a,b\n1,2\n")

            windows_path = Path(
                workflow.save_uploaded_content("folder\\valid.csv", b"valid")
            )
            self.assertEqual(windows_path.name, "valid.csv")
            self.assertEqual(windows_path.read_bytes(), b"valid")

            for invalid_name in ["", ".", ".."]:
                with self.subTest(invalid_name=invalid_name):
                    with self.assertRaises(ValueError):
                        workflow.save_uploaded_content(invalid_name, b"")

    def test_reusing_latest_conversion_overrides_other_structure_sources(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "input.csv"
            output_csv = root / "output.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "structure_tokens": "1 1 1",
                        "pdb_path": "old.pdb",
                    }
                ]
            ).to_csv(input_csv, index=False)

            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )
            workflow.last_structure = {
                "sequence": "ACD",
                "structure_tokens": [7, 8, 9],
            }
            workflow.attach_last_structure_tokens(input_csv, output_csv)

            result = self.pd.read_csv(output_csv)
            self.assertEqual(result.loc[0, "structure_tokens"], "7 8 9")
            self.assertEqual(result.loc[0, "pdb_path"], "old.pdb")

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
                r"expected=\[0, 1\], observed=\[1, 2\]",
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

    def test_prediction_model_uses_explicit_optimizer_config(self):
        class FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return None

        cases = [
            ("classification", "ProSSTClassificationModel"),
            ("regression", "ProSSTRegressionModel"),
        ]
        for task_type, constructor_name in cases:
            with self.subTest(task_type=task_type), patch.object(
                self.prediction,
                constructor_name,
                return_value=FakeModel(),
            ) as constructor:
                self.prediction._load_model(
                    task_type=task_type,
                    model_path="AI4Protein/ProSST-2048",
                    checkpoint_path="model.pt",
                    num_labels=2,
                    device=self.torch.device("cpu"),
                )

                optimizer = constructor.call_args.kwargs["optimizer_kwargs"]
                self.assertEqual(optimizer["class"], "AdamW")
                self.assertEqual(optimizer["betas"], [0.9, 0.98])
                self.assertEqual(optimizer["weight_decay"], 0.01)


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
            def __init__(self):
                self.saved_upload = None

            def maybe_upload_path(self, current_path, upload_enabled):
                return "/tmp/uploaded.file"

            def save_uploaded_content(self, filename, content):
                self.saved_upload = (filename, content)
                return f"/tmp/{Path(filename).name}"

        workflow = DummyWorkflow()
        ui = module.ColabProSSTUI(workflow)
        rendered = []
        global_clear_calls = []
        ui.display = lambda *items: rendered.append(items)
        ui.clear_output = lambda **kwargs: global_clear_calls.append(kwargs)

        input_guide = ui._input_guide()
        self.assertEqual(input_guide.layout.width, "100%")
        self.assertEqual(input_guide.layout.max_width, ui.GUIDE_WIDTH)
        self.assertEqual(input_guide.layout.overflow, "visible")
        self.assertIsNone(input_guide.layout.height)
        self.assertIn("CSV contains structure_tokens", input_guide.value)

        structure_input = module._StructureInput(ui)
        self.assertEqual(structure_input.mode.value, structure_input.TOKENS)
        self.assertEqual(structure_input.zip_upload.path.layout.display, "none")
        self.assertIn("Upload only the CSV", structure_input.hint.value)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tokens_csv = root / "tokens.csv"
            tokens_csv.write_text(
                "sequence,structure_tokens\nACD,\"1 2 3\"\n",
                encoding="utf-8",
            )
            sequence_csv = root / "sequence.csv"
            sequence_csv.write_text("sequence\nACD\n", encoding="utf-8")
            paths_csv = root / "paths.csv"
            paths_csv.write_text(
                "sequence,pdb_path\nACD,protein.pdb\n",
                encoding="utf-8",
            )

            structure_input.validate(tokens_csv)
            with self.assertRaisesRegex(ValueError, "no structure_tokens column"):
                structure_input.validate(sequence_csv)

            structure_input.mode.value = structure_input.PATHS
            structure_input.validate(paths_csv)
            with self.assertRaisesRegex(ValueError, "no pdb_path or structure_path"):
                structure_input.validate(sequence_csv)

            structure_input.mode.value = structure_input.REUSE
            with self.assertRaisesRegex(ValueError, "No structure conversion"):
                structure_input.validate(sequence_csv)

        structure_input.mode.value = structure_input.PATHS
        self.assertIsNone(structure_input.zip_upload.path.layout.display)
        structure_input.zip_upload.value = "/tmp/structures.zip"
        self.assertEqual(
            structure_input.structure_zip, "/tmp/structures.zip"
        )

        structure_input.mode.value = structure_input.REUSE
        self.assertTrue(structure_input.reuse_latest)
        self.assertEqual(structure_input.structure_zip, "")
        self.assertIn("No structure has been converted", structure_input.hint.value)

        workflow.last_structure = {"sequence": "ACD"}
        structure_input = module._StructureInput(ui)
        self.assertEqual(structure_input.mode.value, structure_input.REUSE)
        self.assertIn("3 residues", structure_input.hint.value)

        rendered.clear()
        ui._home_page()
        home_items = rendered[-1]
        self.assertEqual(
            [item.description for item in home_items[1:4]],
            [
                "I want to train my own model",
                "I want to use existing models to make prediction",
                "I want to share my model publicly",
            ],
        )
        self.assertIn("Prepare sequence and structure inputs", home_items[4].value)

        ui.navigation_history.clear()
        ui.current_page = ui._home_page
        ui._update_navigation_controls()
        self.assertTrue(ui.back_button.disabled)

        ui._navigate(ui._prediction_menu_page)
        self.assertEqual(ui.current_page, ui._prediction_menu_page)
        self.assertEqual(ui.navigation_history, [ui._home_page])
        self.assertFalse(ui.back_button.disabled)

        ui._navigate(ui._property_prediction_page)
        self.assertEqual(ui.current_page, ui._property_prediction_page)
        self.assertEqual(
            ui.navigation_history,
            [ui._home_page, ui._prediction_menu_page],
        )

        history_before_refresh = list(ui.navigation_history)
        ui._refresh_page()
        self.assertEqual(ui.current_page, ui._property_prediction_page)
        self.assertEqual(ui.navigation_history, history_before_refresh)

        ui._go_back()
        self.assertEqual(ui.current_page, ui._prediction_menu_page)
        self.assertEqual(ui.navigation_history, [ui._home_page])
        ui._go_back()
        self.assertEqual(ui.current_page, ui._home_page)
        self.assertEqual(ui.navigation_history, [])
        self.assertTrue(ui.back_button.disabled)
        global_clear_calls.clear()

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

        task_button = ui._button("Run test task")
        task_output = ui._output()
        task_started = threading.Event()
        task_release = threading.Event()

        def task_action():
            task_started.set()
            task_release.wait(timeout=5)

        ui._start_task(task_button, task_output, task_action)
        self.assertTrue(task_started.wait(timeout=5))
        task_thread = ui.active_thread
        task_release.set()
        task_thread.join(timeout=5)
        self.assertFalse(task_thread.is_alive())
        ui.stop_task(silent=False)
        self.assertEqual(global_clear_calls, [])

        upload_field = module._UploadField(ui, "Training CSV:", "Choose CSV")
        upload_field.inline_upload.value = "<input type='file'>"
        encoded_chunk = base64.b64encode(b"sequence,label\nACD,1\n").decode("ascii")

        class FakeColabOutput:
            def __init__(self):
                self.responses = iter(
                    [
                        {
                            "action": "append",
                            "file": "training.csv",
                            "data": encoded_chunk,
                        },
                        {"action": "complete"},
                    ]
                )

            def eval_js(self, javascript):
                if "_uploadFilesContinue" in javascript:
                    return next(self.responses)
                if "_uploadFiles(" in javascript:
                    return {"action": "starting"}
                return None

        fake_google = types.ModuleType("google")
        fake_google.__path__ = []
        fake_colab = types.ModuleType("google.colab")
        fake_colab.__path__ = []
        fake_colab.output = FakeColabOutput()
        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.colab": fake_colab},
        ):
            uploaded_path = upload_field._upload_inline()

        self.assertEqual(uploaded_path, "/tmp/training.csv")
        self.assertEqual(
            workflow.saved_upload,
            ("training.csv", b"sequence,label\nACD,1\n"),
        )

        class CanceledColabOutput:
            @staticmethod
            def eval_js(javascript):
                if "_uploadFiles(" in javascript:
                    return {"action": "complete"}
                return None

        fake_colab.output = CanceledColabOutput()
        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.colab": fake_colab},
        ):
            self.assertIsNone(upload_field._upload_inline())

        with tempfile.TemporaryDirectory() as temporary_dir:
            class LegacyWorkflow:
                upload_dir = Path(temporary_dir)

                @staticmethod
                def maybe_upload_path(current_path, upload_enabled):
                    return "/tmp/legacy-upload.file"

            legacy_ui = module.ColabProSSTUI(LegacyWorkflow())
            legacy_field = module._UploadField(
                legacy_ui, "Training CSV:", "Choose CSV"
            )
            fake_colab.output = FakeColabOutput()
            with patch.dict(
                sys.modules,
                {"google": fake_google, "google.colab": fake_colab},
            ):
                legacy_path = Path(legacy_field._upload_inline())

            self.assertEqual(legacy_path, Path(temporary_dir) / "training.csv")
            self.assertEqual(
                legacy_path.read_bytes(), b"sequence,label\nACD,1\n"
            )


if __name__ == "__main__":
    unittest.main()
