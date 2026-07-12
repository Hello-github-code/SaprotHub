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
        self.assertIn("A structure-aware ColabPLM powered by ProSST", introduction)
        self.assertIn("Paper-NeurIPS%202024", introduction)
        self.assertIn("huggingface.co/AI4Protein/ProSST-2048", introduction)
        self.assertIn("github.com/ai4protein/ProSST", introduction)
        self.assertIn("theopmc.github.io", introduction)
        self.assertIn("quantized structure tokens", introduction)
        self.assertIn("sequence-structure disentangled attention", introduction)
        self.assertIn("PDB/mmCIF structure quantization", introduction)
        self.assertIn("residue-level classification", introduction)
        self.assertIn("protein-pair classification and regression", introduction)
        self.assertIn(
            "protein-level and residue-level embedding extraction",
            introduction,
        )
        self.assertIn(
            "single-site saturation mutagenesis with heatmaps",
            introduction,
        )
        self.assertIn("ColabSaprot", introduction)
        self.assertIn("ColabSeprot", introduction)
        self.assertIn("ColabESMC", introduction)
        self.assertIn("ColabProtT5", introduction)
        self.assertIn("SaprotHub/OPMC paper", introduction)
        self.assertIn("Google Colab recommends Chrome", introduction)
        self.assertIn("Hello-github-code/SaprotHub/issues", introduction)
        self.assertNotIn("Prepare sequence and structure inputs", introduction)
        self.assertNotIn("Recommended for a first run", introduction)
        self.assertNotIn("Reuse latest structure conversion", introduction)
        for vocab_size in [20, 128, 512, 1024, 2048, 4096]:
            with self.subTest(vocab_size=vocab_size):
                self.assertIn(f"ProSST-{vocab_size}", introduction)

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
        for vocab_size in [20, 128, 512, 1024, 2048, 4096]:
            with self.subTest(vocab_size=vocab_size):
                self.assertIn(
                    f"prosst/structure/static/{vocab_size}.joblib",
                    source,
                )
        self.assertNotIn("prosst/structure/static/64.joblib", source)

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
        self.assertIn("Prediction &gt;", guide_source)
        self.assertIn("CSV already contains", guide_source)
        self.assertIn("CSV does not contain", guide_source)
        self.assertIn("absolute Colab paths", guide_source)
        self.assertIn("structure_vocab_size", guide_source)
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
            "Residue-level Classification",
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

        self.assertEqual(source.count("structure_input = _StructureInput(self)"), 5)
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
        self.assertIn("self._task_intro(selected_task)", training_page)
        self.assertIn("self._training_dataset_help(selected_task)", training_page)
        self.assertIn("self._uses_category_count(selected_task)", training_page)

        mutation_page = source.split("def _mutation_page(self):", 1)[1].split(
            "def _structure_page(self):", 1
        )[0]
        self.assertIn("Zero-shot model note", mutation_page)
        self.assertIn("log P(mutant) - log P(wild type)", mutation_page)
        self.assertIn("Protein property", mutation_page)

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
    def test_official_model_specs_cover_every_quantized_checkpoint(self):
        from saprot.model.prosst.specs import (
            DEFAULT_PROSST_MODEL,
            PROSST_MODEL_SPECS,
            get_prosst_model_spec,
            resolve_structure_vocab_size,
        )

        self.assertEqual(
            [spec.structure_vocab_size for spec in PROSST_MODEL_SPECS],
            [20, 128, 512, 1024, 2048, 4096],
        )
        self.assertEqual(
            [spec.encoded_structure_vocab_size for spec in PROSST_MODEL_SPECS],
            [23, 131, 515, 1027, 2051, 4099],
        )
        self.assertEqual(DEFAULT_PROSST_MODEL.model_path, "AI4Protein/ProSST-2048")
        self.assertEqual(
            get_prosst_model_spec("AI4Protein/ProSST-4096").structure_vocab_size,
            4096,
        )
        self.assertEqual(
            resolve_structure_vocab_size("AI4Protein/ProSST-20"), 20
        )
        with self.assertRaisesRegex(ValueError, "requires structure_vocab_size=20"):
            resolve_structure_vocab_size("AI4Protein/ProSST-20", 2048)
        with self.assertRaisesRegex(ValueError, "requires an explicit"):
            resolve_structure_vocab_size("local/custom-model")

    def test_structure_tokens_enforce_the_selected_model_vocabulary(self):
        from saprot.data.pdb2prosst import (
            encode_structure_tokens,
            get_structure_tokens_from_entry,
        )

        self.assertEqual(encode_structure_tokens([0, 19], 20), [1, 3, 22, 2])
        with self.assertRaisesRegex(ValueError, r"must be in \[0, 19\]"):
            encode_structure_tokens([20], 20)

        entry = {
            "structure_tokens": "0 1 2",
            "structure_vocab_size": 128,
        }
        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            get_structure_tokens_from_entry(entry, structure_vocab_size=20)
        self.assertEqual(
            get_structure_tokens_from_entry(entry, structure_vocab_size=128),
            [0, 1, 2],
        )

    def test_prosst_checkpoint_records_its_model_contract(self):
        import torch

        from saprot.model.prosst.base import ProSSTBaseModel

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model.pt"
            model = object.__new__(ProSSTBaseModel)
            torch.nn.Module.__init__(model)
            model.model = torch.nn.Linear(2, 1)
            model.task = "regression"
            model.config_path = "AI4Protein/ProSST-20"
            model.structure_vocab_size = 20
            model.lora_kwargs = None

            model.save_checkpoint(str(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            self.assertEqual(
                checkpoint["colabprosst"],
                {
                    "base_model": "AI4Protein/ProSST-20",
                    "structure_vocab_size": 20,
                    "task": "regression",
                },
            )

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


class ColabProSSTResidueDataTest(unittest.TestCase):
    def test_residue_label_parser_accepts_documented_formats(self):
        from saprot.data.prosst_labels import parse_residue_labels

        self.assertEqual(parse_residue_labels("0 1 -100 2"), [0, 1, -100, 2])
        self.assertEqual(parse_residue_labels("0,1,2"), [0, 1, 2])
        self.assertEqual(parse_residue_labels("[2, 1, 0]"), [2, 1, 0])
        self.assertEqual(parse_residue_labels([1, 0]), [1, 0])

        with self.assertRaisesRegex(ValueError, "integer category IDs"):
            parse_residue_labels("0 1.5 2")
        with self.assertRaisesRegex(ValueError, "use -100 only"):
            parse_residue_labels("0 -1 2")

    def test_token_classification_csv_builds_aligned_lmdb_samples(self):
        import lmdb
        import pandas as pd

        from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "residue-training.csv"
            pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "residue_labels": "0 1 0",
                        "stage": "train",
                        "structure_tokens": "0 1 2",
                    },
                    {
                        "sequence": "ACE",
                        "residue_labels": "[1, -100, 0]",
                        "stage": "valid",
                        "structure_tokens": "3 4 5",
                    },
                    {
                        "sequence": "ACF",
                        "residue_labels": "1,1,0",
                        "stage": "test",
                        "structure_tokens": "6 7 8",
                    },
                ]
            ).to_csv(csv_path, index=False)

            construct_prosst_lmdb(
                str(csv_path),
                str(root / "LMDB"),
                "residue-task",
                "token_classification",
                structure_vocab_size=20,
            )

            expected = {
                "train": [0, 1, 0],
                "valid": [1, -100, 0],
                "test": [1, 1, 0],
            }
            for stage, expected_labels in expected.items():
                env = lmdb.open(
                    str(root / "LMDB" / "residue-task" / stage),
                    readonly=True,
                    lock=False,
                )
                try:
                    with env.begin() as transaction:
                        sample = json.loads(transaction.get(b"0").decode())
                    self.assertEqual(sample["label"], expected_labels)
                    self.assertEqual(sample["structure_vocab_size"], 20)
                finally:
                    env.close()

    def test_token_classification_csv_rejects_misaligned_labels(self):
        import pandas as pd

        from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "bad-residue-training.csv"
            pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "residue_labels": "0 1",
                        "stage": "train",
                        "structure_tokens": "0 1 2",
                    }
                ]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "residue_labels length must match sequence length",
            ):
                construct_prosst_lmdb(
                    str(csv_path),
                    str(root / "LMDB"),
                    "bad-residue-task",
                    "token_classification",
                    structure_vocab_size=20,
                )

    def test_token_classification_dataset_aligns_special_and_padding_tokens(self):
        import torch

        from saprot.dataset.prosst.prosst_token_classification_dataset import (
            ProSSTTokenClassificationDataset,
        )

        class FakeTokenizer:
            vocab = {"A": 3, "C": 4, "D": 5}

            def batch_encode_plus(self, sequences, **_kwargs):
                encoded = [
                    [1, *[self.vocab[residue] for residue in sequence], 2]
                    for sequence in sequences
                ]
                target_length = max(len(row) for row in encoded)
                input_ids = [
                    row + [0] * (target_length - len(row)) for row in encoded
                ]
                attention_mask = [
                    [1] * len(row) + [0] * (target_length - len(row))
                    for row in encoded
                ]
                return {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                }

        dataset = object.__new__(ProSSTTokenClassificationDataset)
        dataset.tokenizer = FakeTokenizer()
        dataset.structure_vocab_size = 20
        dataset.max_length = 10

        inputs, labels = dataset.collate_fn(
            [
                ("ACD", [0, 1, 2], [0, 1, -100]),
                ("AC", [3, 4], [1, 0]),
            ]
        )

        self.assertEqual(inputs["inputs"]["input_ids"].shape, (2, 5))
        self.assertEqual(
            inputs["inputs"]["ss_input_ids"].tolist(),
            [[1, 3, 4, 5, 2], [1, 6, 7, 2, 0]],
        )
        self.assertEqual(
            labels["labels"].tolist(),
            [[-100, 0, 1, -100, -100], [-100, 1, 0, -100, -100]],
        )


class ColabProSSTResidueModelTest(unittest.TestCase):
    def test_token_checkpoint_metadata_records_number_of_categories(self):
        from saprot.model.prosst.prosst_token_classification_model import (
            ProSSTTokenClassificationModel,
        )

        model = object.__new__(ProSSTTokenClassificationModel)
        object.__setattr__(model, "task", "token_classification")
        object.__setattr__(model, "config_path", "AI4Protein/ProSST-128")
        object.__setattr__(model, "structure_vocab_size", 128)
        object.__setattr__(model, "num_labels", 3)

        self.assertEqual(
            model._checkpoint_metadata()["colabprosst"],
            {
                "base_model": "AI4Protein/ProSST-128",
                "structure_vocab_size": 128,
                "task": "token_classification",
                "num_labels": 3,
            },
        )

    def test_token_classification_loss_ignores_non_residue_positions(self):
        import torch

        from saprot.model.prosst.prosst_token_classification_model import (
            ProSSTTokenClassificationModel,
        )

        class CaptureMetric:
            def update(self, logits, targets):
                self.logits = logits
                self.targets = targets

        model = object.__new__(ProSSTTokenClassificationModel)
        torch.nn.Module.__init__(model)
        model.num_labels = 3
        metric = CaptureMetric()
        model.metrics = {"valid": {"valid_acc": metric}}

        logits = torch.tensor(
            [
                [
                    [9.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [0.0, 1.0, 4.0],
                    [0.0, 9.0, 0.0],
                ]
            ]
        )
        labels = {"labels": torch.tensor([[-100, 0, 2, -100]])}

        loss = model.loss_func("valid", logits, labels)
        expected = torch.nn.functional.cross_entropy(
            logits[0, 1:3],
            torch.tensor([0, 2]),
        )

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(metric.targets.tolist(), [0, 2])
        self.assertEqual(metric.logits.shape, (2, 3))

    def test_token_classification_test_csv_uses_one_based_residue_indices(self):
        import pandas as pd
        import torch

        from saprot.model.prosst.prosst_token_classification_model import (
            ProSSTTokenClassificationModel,
        )

        class DummyMetric:
            @staticmethod
            def compute():
                return torch.tensor(1.0)

            @staticmethod
            def reset():
                return None

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "residue-test.csv"
            model = object.__new__(ProSSTTokenClassificationModel)
            torch.nn.Module.__init__(model)
            model.num_labels = 2
            model.test_result_path = str(output_path)
            model.test_outputs = [torch.tensor(0.25)]
            model.test_token_outputs = [
                (
                    torch.tensor([1, 3]),
                    torch.tensor([[3.0, 1.0], [0.0, 2.0]]),
                    torch.tensor([0, 1]),
                )
            ]
            model.metrics = {"test": {"test_acc": DummyMetric()}}
            model.output_test_metrics = lambda _log_dict: None
            model.log_info = lambda _log_dict: None

            model.on_test_epoch_end()

            result = pd.read_csv(output_path)
            self.assertEqual(result["sample_index"].tolist(), [0, 0])
            self.assertEqual(result["residue_index"].tolist(), [1, 3])
            self.assertEqual(result["pred"].tolist(), [0, 1])
            self.assertEqual(result["target"].tolist(), [0, 1])


class ColabProSSTPairDataTest(unittest.TestCase):
    class FakeTokenizer:
        vocab = {"A": 3, "C": 4, "D": 5, "E": 6}

        def batch_encode_plus(self, sequences, **_kwargs):
            import torch

            encoded = [
                [1, *[self.vocab[residue] for residue in sequence], 2]
                for sequence in sequences
            ]
            target_length = max(len(row) for row in encoded)
            return {
                "input_ids": torch.tensor(
                    [
                        row + [0] * (target_length - len(row))
                        for row in encoded
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1] * len(row) + [0] * (target_length - len(row))
                        for row in encoded
                    ]
                ),
            }

    def test_pair_csv_builds_two_independently_aligned_structures(self):
        import lmdb
        import pandas as pd

        from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "pair-training.csv"
            pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "label": 1,
                        "stage": stage,
                        "structure_tokens_1": "0 1 2",
                        "structure_tokens_2": "3 4",
                        "structure_vocab_size": 20,
                    }
                    for stage in ["train", "valid", "test"]
                ]
            ).to_csv(csv_path, index=False)

            construct_prosst_lmdb(
                str(csv_path),
                str(root / "LMDB"),
                "pair-task",
                "pair_classification",
                structure_vocab_size=20,
            )

            env = lmdb.open(
                str(root / "LMDB" / "pair-task" / "train"),
                readonly=True,
                lock=False,
            )
            try:
                with env.begin() as transaction:
                    sample = json.loads(transaction.get(b"0").decode())
            finally:
                env.close()

            self.assertEqual(sample["seq_1"], "ACD")
            self.assertEqual(sample["seq_2"], "AC")
            self.assertEqual(sample["structure_tokens_1"], "0 1 2")
            self.assertEqual(sample["structure_tokens_2"], "3 4")
            self.assertEqual(sample["structure_vocab_size"], 20)
            self.assertEqual(sample["label"], 1)

    def test_pair_csv_requires_structure_for_each_partner(self):
        import pandas as pd

        from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "bad-pair.csv"
            pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "label": 1,
                        "stage": "train",
                        "structure_tokens_1": "0 1 2",
                    }
                ]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "Pair protein 2"):
                construct_prosst_lmdb(
                    str(csv_path),
                    str(root / "LMDB"),
                    "bad-pair-task",
                    "pair_classification",
                    structure_vocab_size=20,
                )

    def test_pair_classification_rejects_fractional_category_ids(self):
        import pandas as pd

        from saprot.utils.construct_prosst_lmdb import construct_prosst_lmdb

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "fractional-pair.csv"
            pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "label": 1.5,
                        "stage": "train",
                        "structure_tokens_1": "0 1 2",
                        "structure_tokens_2": "3 4",
                    }
                ]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "integer category ID"):
                construct_prosst_lmdb(
                    str(csv_path),
                    str(root / "LMDB"),
                    "fractional-pair-task",
                    "pair_classification",
                    structure_vocab_size=20,
                )

    def test_pair_datasets_collate_both_prosst_inputs_and_label_types(self):
        import torch

        from saprot.dataset.prosst.prosst_pair_classification_dataset import (
            ProSSTPairClassificationDataset,
        )
        from saprot.dataset.prosst.prosst_pair_regression_dataset import (
            ProSSTPairRegressionDataset,
        )

        batch = [
            ("ACD", [0, 1, 2], "AC", [3, 4], 1),
            ("AC", [5, 6], "ACE", [7, 8, 9], 0),
        ]
        classification = object.__new__(ProSSTPairClassificationDataset)
        classification.tokenizer = self.FakeTokenizer()
        classification.structure_vocab_size = 20
        classification.max_length = 10

        inputs, labels = classification.collate_fn(batch)
        self.assertEqual(set(inputs), {"inputs_1", "inputs_2"})
        self.assertEqual(inputs["inputs_1"]["input_ids"].shape, (2, 5))
        self.assertEqual(inputs["inputs_2"]["input_ids"].shape, (2, 5))
        self.assertEqual(
            inputs["inputs_1"]["ss_input_ids"].tolist(),
            [[1, 3, 4, 5, 2], [1, 8, 9, 2, 0]],
        )
        self.assertEqual(labels["labels"].dtype, torch.long)
        self.assertEqual(labels["labels"].tolist(), [1, 0])

        regression = object.__new__(ProSSTPairRegressionDataset)
        regression.tokenizer = self.FakeTokenizer()
        regression.structure_vocab_size = 20
        regression.max_length = 10
        _inputs, regression_labels = regression.collate_fn(batch)
        self.assertEqual(regression_labels["labels"].dtype, torch.float)


class ColabProSSTPairModelTest(unittest.TestCase):
    def test_pair_forward_uses_both_pooled_representations(self):
        import torch

        from saprot.model.prosst.pair_base import ProSSTPairBaseModel

        model = object.__new__(ProSSTPairBaseModel)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.classifier = torch.nn.Linear(4, 1, bias=False)
        model.model.classifier.weight.data.copy_(
            torch.tensor([[1.0, 10.0, 100.0, 1000.0]])
        )
        model.get_pooled_representations = lambda inputs: inputs["pooled"]

        output = model.forward(
            {"pooled": torch.tensor([[1.0, 2.0]])},
            {"pooled": torch.tensor([[3.0, 4.0]])},
        )

        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(output.item(), 4321.0)

    def test_pair_classification_metadata_records_task_and_categories(self):
        from saprot.model.prosst.prosst_pair_classification_model import (
            ProSSTPairClassificationModel,
        )

        model = object.__new__(ProSSTPairClassificationModel)
        object.__setattr__(model, "task", "pair_classification")
        object.__setattr__(model, "config_path", "AI4Protein/ProSST-512")
        object.__setattr__(model, "structure_vocab_size", 512)
        object.__setattr__(model, "num_labels", 4)

        self.assertEqual(
            model._checkpoint_metadata()["colabprosst"],
            {
                "base_model": "AI4Protein/ProSST-512",
                "structure_vocab_size": 512,
                "task": "pair_classification",
                "num_labels": 4,
            },
        )

    def test_pair_losses_use_expected_label_types_and_shapes(self):
        import torch

        from saprot.model.prosst.prosst_pair_classification_model import (
            ProSSTPairClassificationModel,
        )
        from saprot.model.prosst.prosst_pair_regression_model import (
            ProSSTPairRegressionModel,
        )

        class CaptureMetric:
            def set_dtype(self, _dtype):
                return self

            def update(self, predictions, targets):
                self.predictions = predictions
                self.targets = targets

        classification = object.__new__(ProSSTPairClassificationModel)
        torch.nn.Module.__init__(classification)
        classification.metrics = {
            "valid": {"valid_acc": CaptureMetric()},
        }
        logits = torch.tensor([[1.0, 3.0], [4.0, 0.0]])
        category_ids = torch.tensor([1, 0], dtype=torch.long)
        classification_loss = classification.loss_func(
            "valid",
            logits,
            {"labels": category_ids},
        )
        self.assertTrue(
            torch.allclose(
                classification_loss,
                torch.nn.functional.cross_entropy(logits, category_ids),
            )
        )

        regression = object.__new__(ProSSTPairRegressionModel)
        torch.nn.Module.__init__(regression)
        metric = CaptureMetric()
        regression.metrics = {"valid": {"valid_loss": metric}}
        predictions = torch.tensor([1.5, -0.5])
        targets = torch.tensor([1.0, 0.0])
        regression_loss = regression.loss_func(
            "valid",
            predictions,
            {"labels": targets},
        )
        self.assertTrue(
            torch.allclose(
                regression_loss,
                torch.nn.functional.mse_loss(predictions, targets),
            )
        )
        self.assertEqual(metric.targets.shape, (2,))


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
                    model_path="AI4Protein/ProSST-20",
                    load_pretrained=False,
                    download=False,
                )

            scheduler = captured["model_config"].kwargs.lr_scheduler_kwargs
            self.assertEqual(scheduler["class"], "ConstantLRScheduler")
            self.assertEqual(scheduler["init_lr"], 3.0e-5)
            self.assertEqual(
                captured["model_config"].kwargs.structure_vocab_size,
                20,
            )
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

    def test_residue_training_selects_token_model_and_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "residue-training.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "residue_labels": "0 1 0",
                        "stage": "train",
                        "structure_tokens": "0 1 2",
                    },
                    {
                        "sequence": "ACE",
                        "residue_labels": "1 0 1",
                        "stage": "valid",
                        "structure_tokens": "0 1 3",
                    },
                    {
                        "sequence": "ACF",
                        "residue_labels": "0 1 1",
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
                    pass

                def test(self, model, datamodule):
                    pass

            def load_model(config):
                captured["model"] = config
                return DummyModel()

            def load_dataset(config):
                captured["dataset"] = config
                return DummyDataModule()

            with patch(
                "saprot.utils.colab_prosst_workflow.construct_prosst_lmdb"
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_model",
                side_effect=load_model,
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_dataset",
                side_effect=load_dataset,
            ), patch(
                "saprot.utils.colab_prosst_workflow.load_trainer",
                return_value=DummyTrainer(),
            ):
                result = workflow.train_downstream(
                    task_type="token_classification",
                    input_csv=str(csv_path),
                    task_name="residue-training-test",
                    num_labels=2,
                    max_epochs=1,
                    load_pretrained=False,
                    download=False,
                )

            self.assertEqual(
                captured["model"].model_py_path,
                "prosst/prosst_token_classification_model",
            )
            self.assertEqual(captured["model"].kwargs.num_labels, 2)
            self.assertEqual(
                captured["dataset"].dataset_py_path,
                "prosst/prosst_token_classification_dataset",
            )
            self.assertEqual(result["task_type"], "token_classification")

    def test_pair_training_selects_pair_models_and_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "pair-training.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "label": label,
                        "stage": stage,
                        "structure_tokens_1": "0 1 2",
                        "structure_tokens_2": "3 4",
                    }
                    for label, stage in [(0, "train"), (1, "valid"), (0, "test")]
                ]
            ).to_csv(csv_path, index=False)

            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )
            captured = []

            class DummyModel:
                pass

            class DummyDataModule:
                pass

            class DummyTrainer:
                def fit(self, model, datamodule):
                    pass

                def test(self, model, datamodule):
                    pass

            def load_model(config):
                captured[-1]["model"] = config
                return DummyModel()

            def load_dataset(config):
                captured[-1]["dataset"] = config
                return DummyDataModule()

            with patch(
                "saprot.utils.colab_prosst_workflow.construct_prosst_lmdb"
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_model",
                side_effect=load_model,
            ), patch(
                "saprot.utils.colab_prosst_workflow.my_load_dataset",
                side_effect=load_dataset,
            ), patch(
                "saprot.utils.colab_prosst_workflow.load_trainer",
                return_value=DummyTrainer(),
            ):
                for task_type in ["pair_classification", "pair_regression"]:
                    captured.append({"task_type": task_type})
                    workflow.train_downstream(
                        task_type=task_type,
                        input_csv=str(csv_path),
                        task_name=f"{task_type}-test",
                        num_labels=2,
                        max_epochs=1,
                        load_pretrained=False,
                        download=False,
                    )

            expected = {
                "pair_classification": (
                    "prosst/prosst_pair_classification_model",
                    "prosst/prosst_pair_classification_dataset",
                ),
                "pair_regression": (
                    "prosst/prosst_pair_regression_model",
                    "prosst/prosst_pair_regression_dataset",
                ),
            }
            for entry in captured:
                model_path, dataset_path = expected[entry["task_type"]]
                self.assertEqual(entry["model"].model_py_path, model_path)
                self.assertEqual(entry["dataset"].dataset_py_path, dataset_path)
            self.assertEqual(captured[0]["model"].kwargs.num_labels, 2)
            self.assertNotIn("num_labels", captured[1]["model"].kwargs)

    def test_pair_tasks_reject_single_structure_reuse(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )

            for method_name, kwargs in [
                (
                    "train_downstream",
                    {
                        "task_type": "pair_classification",
                        "input_csv": "unused.csv",
                    },
                ),
                (
                    "predict_downstream",
                    {
                        "task_type": "pair_regression",
                        "input_csv": "unused.csv",
                        "checkpoint_path": "unused.pt",
                    },
                ),
            ]:
                with self.subTest(method=method_name), self.assertRaisesRegex(
                    ValueError,
                    "cannot reuse one latest structure conversion",
                ):
                    getattr(workflow, method_name)(
                        use_last_structure_tokens=True,
                        **kwargs,
                    )

    def test_pair_label_validation_matches_scalar_task_semantics(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            classification_csv = root / "pair-classification.csv"
            regression_csv = root / "pair-regression.csv"
            self.pd.DataFrame({"label": [0, 1, 0]}).to_csv(
                classification_csv,
                index=False,
            )
            self.pd.DataFrame({"label": [0.25, -1.5, 2.0]}).to_csv(
                regression_csv,
                index=False,
            )

            self.workflow_class._validate_training_labels(
                str(classification_csv),
                "pair_classification",
                2,
            )
            self.workflow_class._validate_training_labels(
                str(regression_csv),
                "pair_regression",
                2,
            )

            with self.assertRaisesRegex(ValueError, "integer category IDs"):
                self.workflow_class._validate_training_labels(
                    str(regression_csv),
                    "pair_classification",
                    2,
                )

    def test_embedding_workflow_packages_tensor_and_index_outputs(self):
        import zipfile

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "embedding-input.csv"
            self.pd.DataFrame(
                [{"sequence": "ACD", "structure_tokens": "0 1 2"}]
            ).to_csv(input_csv, index=False)
            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )
            captured = {}

            def extract(**kwargs):
                captured.update(kwargs)
                Path(kwargs["output_pt"]).write_bytes(b"embedding tensor")
                self.pd.DataFrame(
                    [{"embedding_index": 0, "sequence": "ACD"}]
                ).to_csv(kwargs["output_index_csv"], index=False)
                return {
                    "embedding_level": kwargs["level"],
                    "protein_embeddings": "placeholder",
                }

            with patch(
                "saprot.utils.colab_prosst_workflow.extract_prosst_embeddings",
                side_effect=extract,
            ):
                result = workflow.extract_embeddings(
                    input_csv=str(input_csv),
                    model_path="AI4Protein/ProSST-20",
                    level="both",
                    batch_size=4,
                    download=False,
                )

            self.assertEqual(captured["structure_vocab_size"], 20)
            self.assertEqual(captured["batch_size"], 4)
            self.assertEqual(captured["level"], "both")
            archive_path = Path(result["archive_path"])
            self.assertTrue(archive_path.exists())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "prosst_both_embeddings.pt",
                        "prosst_both_embeddings_index.csv",
                    ],
                )

    def test_saturation_workflow_packages_tables_and_heatmap(self):
        import zipfile

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "saturation-input.csv"
            self.pd.DataFrame(
                [{"sequence": "ACD", "structure_tokens": "0 1 2"}]
            ).to_csv(input_csv, index=False)
            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )
            captured = {}

            def score(**kwargs):
                captured.update(kwargs)
                Path(kwargs["output_csv"]).write_text(
                    "mutation,score\nA1C,1.0\n",
                    encoding="utf-8",
                )
                Path(kwargs["output_matrix_csv"]).write_text(
                    "mutant,A1\nC,1.0\n",
                    encoding="utf-8",
                )
                Path(kwargs["output_heatmap_png"]).write_bytes(b"png")
                return {"score_table": self.pd.DataFrame()}

            with patch(
                "saprot.utils.colab_prosst_workflow."
                "score_saturation_mutagenesis",
                side_effect=score,
            ):
                result = workflow.run_saturation_mutagenesis(
                    input_csv=str(input_csv),
                    model_path="AI4Protein/ProSST-20",
                    download=False,
                )

            self.assertEqual(captured["structure_vocab_size"], 20)
            archive_path = Path(result["archive_path"])
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "prosst_saturation_heatmap.png",
                        "prosst_saturation_matrix.csv",
                        "prosst_saturation_scores.csv",
                    ],
                )

    def test_csv_templates_follow_the_selected_model_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            workflow = self.workflow_class(
                output_dir=str(root / "output"),
                upload_dir=str(root / "uploads"),
                asset_dir=str(root / "assets"),
                cache_dir=str(root / "cache"),
                saprothub_dir=str(root / "SaprotHub"),
            )

            workflow.create_csv_templates(
                template_dir=str(root / "templates"),
                structure_vocab_size=20,
            )

            token_template = self.pd.read_csv(
                root / "templates" / "prosst_classification_template.csv"
            )
            residue_template = self.pd.read_csv(
                root / "templates" / "prosst_token_classification_template.csv"
            )
            path_template = self.pd.read_csv(
                root / "templates" / "prosst_classification_pdb_template.csv"
            )
            pair_template = self.pd.read_csv(
                root / "templates" / "prosst_pair_classification_template.csv"
            )
            pair_path_template = self.pd.read_csv(
                root
                / "templates"
                / "prosst_pair_regression_pdb_template.csv"
            )
            pair_prediction = self.pd.read_csv(
                root / "templates" / "prosst_pair_prediction_template.csv"
            )
            embedding_template = self.pd.read_csv(
                root / "templates" / "prosst_embedding_template.csv"
            )
            saturation_template = self.pd.read_csv(
                root / "templates" / "prosst_saturation_template.csv"
            )
            self.assertEqual(
                token_template["structure_vocab_size"].unique().tolist(),
                [20],
            )
            self.assertEqual(
                residue_template["structure_vocab_size"].unique().tolist(),
                [20],
            )
            self.assertEqual(
                residue_template.loc[0, "residue_labels"],
                "0 1 0",
            )
            self.assertNotIn("structure_vocab_size", path_template.columns)
            self.assertEqual(
                pair_template["structure_vocab_size"].unique().tolist(),
                [20],
            )
            self.assertTrue(
                {
                    "sequence_1",
                    "sequence_2",
                    "structure_tokens_1",
                    "structure_tokens_2",
                }.issubset(pair_template.columns)
            )
            self.assertTrue(
                {"pdb_path_1", "pdb_path_2"}.issubset(
                    pair_path_template.columns
                )
            )
            self.assertNotIn("label", pair_prediction.columns)
            self.assertEqual(
                embedding_template.columns.tolist(),
                token_template[[
                    "sequence",
                    "structure_tokens",
                    "structure_vocab_size",
                ]].columns.tolist(),
            )
            self.assertEqual(
                embedding_template["structure_vocab_size"].unique().tolist(),
                [20],
            )
            self.assertEqual(len(saturation_template), 1)
            self.assertEqual(
                saturation_template.columns.tolist(),
                ["sequence", "structure_tokens", "structure_vocab_size"],
            )

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
            self.assertEqual(result.loc[0, "structure_vocab_size"], 2048)
            self.assertEqual(result.loc[0, "pdb_path"], "old.pdb")

    def test_reusing_latest_conversion_rejects_a_different_model_family(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "input.csv"
            output_csv = root / "output.csv"
            self.pd.DataFrame([{"sequence": "ACD"}]).to_csv(
                input_csv,
                index=False,
            )

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
                "structure_vocab_size": 128,
            }

            with self.assertRaisesRegex(
                ValueError,
                "tokens use structure_vocab_size=128",
            ):
                workflow.attach_last_structure_tokens(
                    str(input_csv),
                    str(output_csv),
                    structure_vocab_size=2048,
                )

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

    def test_zero_shot_uses_the_selected_models_structure_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "mutations.csv"
            output_csv = root / "scores.csv"
            self.pd.DataFrame(
                [
                    {
                        "sequence": "ACD",
                        "mutant": "A1C",
                        "structure_tokens": "4095 1 2",
                        "structure_vocab_size": 4096,
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
                    model_path="AI4Protein/ProSST-4096",
                    device="cpu",
                )

            self.assertAlmostEqual(result.loc[0, "score"], 1.0, places=5)

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

    def test_pair_prediction_prepares_both_sequence_structure_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "pair-prediction.csv"
            output_csv = root / "pair-predictions.csv"
            checkpoint = root / "model.pt"
            checkpoint.touch()
            self.pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "structure_tokens_1": "0 1 2",
                        "structure_tokens_2": "3 4",
                    },
                    {
                        "sequence_1": "AC",
                        "sequence_2": "ACE",
                        "structure_tokens_1": "5 6",
                        "structure_tokens_2": "7 8 9",
                    },
                ]
            ).to_csv(input_csv, index=False)

            captured = {}

            class FakePairPredictionModel:
                def forward(inner_self, inputs_1, inputs_2):
                    captured["inputs_1"] = inputs_1
                    captured["inputs_2"] = inputs_2
                    return self.torch.tensor([[0.0, 2.0], [3.0, 1.0]])

            with patch.object(
                self.prediction.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.prediction,
                "_load_model",
                return_value=FakePairPredictionModel(),
            ):
                result = self.prediction.predict_csv(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    task_type="pair_classification",
                    checkpoint_path=str(checkpoint),
                    num_labels=2,
                    batch_size=2,
                    model_path="AI4Protein/ProSST-20",
                    device="cpu",
                )

            self.assertEqual(result["pred"].tolist(), [1, 0])
            self.assertEqual(
                captured["inputs_1"]["ss_input_ids"].tolist(),
                [[1, 3, 4, 5, 2], [1, 8, 9, 2, 0]],
            )
            self.assertEqual(
                captured["inputs_2"]["ss_input_ids"].tolist(),
                [[1, 6, 7, 2, 0], [1, 10, 11, 12, 2]],
            )
            self.assertTrue(output_csv.exists())

    def test_pair_prediction_requires_the_second_structure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "incomplete-pair.csv"
            output_csv = root / "pair-predictions.csv"
            checkpoint = root / "model.pt"
            checkpoint.touch()
            self.pd.DataFrame(
                [
                    {
                        "sequence_1": "ACD",
                        "sequence_2": "AC",
                        "structure_tokens_1": "0 1 2",
                    }
                ]
            ).to_csv(input_csv, index=False)

            with self.assertRaisesRegex(ValueError, "pair protein 2"):
                self.prediction.predict_csv(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    task_type="pair_regression",
                    checkpoint_path=str(checkpoint),
                    model_path="AI4Protein/ProSST-20",
                    device="cpu",
                )

    def test_residue_prediction_preserves_each_sequences_token_count(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "prediction.csv"
            output_csv = root / "predictions.csv"
            checkpoint = root / "model.pt"
            checkpoint.touch()
            self.pd.DataFrame(
                [
                    {"sequence": "ACD", "structure_tokens": "0 1 2"},
                    {"sequence": "AC", "structure_tokens": "3 4"},
                ]
            ).to_csv(input_csv, index=False)

            class FakeTokenPredictionModel:
                def forward(inner_self, inputs):
                    input_ids = inputs["input_ids"]
                    logits = self.torch.zeros(
                        (input_ids.shape[0], input_ids.shape[1], 2)
                    )
                    logits[:, :, 0] = 1.0
                    logits[:, 2, 1] = 3.0
                    return logits

            with patch.object(
                self.prediction.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.prediction,
                "_load_model",
                return_value=FakeTokenPredictionModel(),
            ):
                result = self.prediction.predict_csv(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    task_type="token_classification",
                    checkpoint_path=str(checkpoint),
                    num_labels=2,
                    batch_size=1,
                    device="cpu",
                )

            self.assertEqual(result["prediction_length"].tolist(), [3, 2])
            self.assertEqual(
                result["predicted_labels"].tolist(),
                ["0 1 0", "0 1"],
            )
            self.assertEqual(len(result.loc[0, "confidence"].split()), 3)
            self.assertEqual(len(result.loc[1, "prob_0"].split()), 2)
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

    def test_prediction_rejects_model_and_vocabulary_mismatch(self):
        with self.assertRaisesRegex(ValueError, "requires structure_vocab_size=20"):
            self.prediction.predict_csv(
                input_csv="unused.csv",
                output_csv="unused-output.csv",
                task_type="classification",
                checkpoint_path="unused.pt",
                model_path="AI4Protein/ProSST-20",
                structure_vocab_size=2048,
            )

    def test_prediction_rejects_checkpoint_from_another_family_model(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model.pt"
            self.torch.save(
                {
                    "model": {},
                    "colabprosst": {
                        "task": "classification",
                        "base_model": "AI4Protein/ProSST-2048",
                        "structure_vocab_size": 2048,
                    },
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.prediction.validate_checkpoint_compatibility(
                    str(checkpoint_path),
                    "classification",
                    "AI4Protein/ProSST-4096",
                    4096,
                )

    def test_prediction_rejects_checkpoint_with_another_category_count(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model.pt"
            self.torch.save(
                {
                    "model": {},
                    "colabprosst": {
                        "task": "token_classification",
                        "base_model": "AI4Protein/ProSST-128",
                        "structure_vocab_size": 128,
                        "num_labels": 3,
                    },
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "num_labels=3"):
                self.prediction.validate_checkpoint_compatibility(
                    str(checkpoint_path),
                    "token_classification",
                    "AI4Protein/ProSST-128",
                    128,
                    num_labels=2,
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
            ("token_classification", "ProSSTTokenClassificationModel"),
            ("pair_classification", "ProSSTPairClassificationModel"),
            ("pair_regression", "ProSSTPairRegressionModel"),
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
                    structure_vocab_size=2048,
                    device=self.torch.device("cpu"),
                )

                optimizer = constructor.call_args.kwargs["optimizer_kwargs"]
                self.assertEqual(optimizer["class"], "AdamW")
                self.assertEqual(optimizer["betas"], [0.9, 0.98])
                self.assertEqual(optimizer["weight_decay"], 0.01)
                self.assertEqual(
                    constructor.call_args.kwargs["structure_vocab_size"],
                    2048,
                )


class ColabProSSTEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas
        import torch

        from saprot.scripts import extract_prosst_embeddings

        cls.embedding = extract_prosst_embeddings
        cls.pd = pandas
        cls.torch = torch

    class FakeTokenizer:
        vocab = {"A": 3, "C": 4, "D": 5}

        def batch_encode_plus(
            self,
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=None,
        ):
            torch = ColabProSSTEmbeddingTest.torch
            encoded = [
                [1, *[self.vocab[residue] for residue in sequence], 2]
                for sequence in sequences
            ]
            target_length = max(len(row) for row in encoded)
            return {
                "input_ids": torch.tensor(
                    [
                        row + [0] * (target_length - len(row))
                        for row in encoded
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1] * len(row) + [0] * (target_length - len(row))
                        for row in encoded
                    ]
                ),
            }

    class FakeModel:
        def __init__(self):
            self.last_ss_input_ids = None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(
            self,
            input_ids,
            attention_mask,
            ss_input_ids,
            output_hidden_states,
            return_dict,
        ):
            self.last_ss_input_ids = ss_input_ids.detach().cpu()
            hidden = ColabProSSTEmbeddingTest.torch.stack(
                [input_ids.float(), ss_input_ids.float()],
                dim=-1,
            )
            return types.SimpleNamespace(
                hidden_states=(self.torch_zeros(hidden), hidden)
            )

        @staticmethod
        def torch_zeros(hidden):
            return ColabProSSTEmbeddingTest.torch.zeros_like(hidden)

    def test_extracts_aligned_protein_and_residue_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "embeddings.csv"
            output_pt = root / "embeddings.pt"
            self.pd.DataFrame(
                [
                    {"sequence": "ACD", "structure_tokens": "0 1 2"},
                    {"sequence": "AC", "structure_tokens": "3 4"},
                ]
            ).to_csv(input_csv, index=False)
            fake_model = self.FakeModel()

            with patch.object(
                self.embedding.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.embedding.AutoModelForMaskedLM,
                "from_pretrained",
                return_value=fake_model,
            ):
                result = self.embedding.extract_embeddings(
                    input_csv=str(input_csv),
                    output_pt=str(output_pt),
                    model_path="AI4Protein/ProSST-20",
                    level="both",
                    batch_size=2,
                    device="cpu",
                )

            saved = self.torch.load(output_pt, map_location="cpu")
            self.assertEqual(saved["format_version"], 1)
            self.assertEqual(saved["embedding_level"], "both")
            self.assertEqual(saved["layer_index"], 1)
            self.assertEqual(saved["hidden_size"], 2)
            self.assertEqual(saved["dtype"], "float32")
            self.assertEqual(saved["sequence_lengths"].tolist(), [3, 2])
            self.assertEqual(
                [tuple(value.shape) for value in saved["residue_embeddings"]],
                [(3, 2), (2, 2)],
            )
            self.assertTrue(
                self.torch.allclose(
                    saved["residue_embeddings"][0],
                    self.torch.tensor(
                        [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
                    ),
                )
            )
            self.assertTrue(
                self.torch.allclose(
                    saved["protein_embeddings"],
                    self.torch.tensor([[4.0, 4.0], [3.5, 6.5]]),
                )
            )
            self.assertEqual(
                fake_model.last_ss_input_ids.tolist(),
                [[1, 3, 4, 5, 2], [1, 6, 7, 2, 0]],
            )
            index = self.pd.read_csv(result["output_index_csv"])
            self.assertEqual(index["embedding_index"].tolist(), [0, 1])
            self.assertEqual(index["sequence_length"].tolist(), [3, 2])

    def test_embedding_extraction_never_silently_truncates(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "too-long.csv"
            self.pd.DataFrame(
                [{"sequence": "ACD", "structure_tokens": "0 1 2"}]
            ).to_csv(input_csv, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "Embeddings are not silently truncated",
            ):
                self.embedding.extract_embeddings(
                    input_csv=str(input_csv),
                    output_pt=str(root / "unused.pt"),
                    model_path="AI4Protein/ProSST-20",
                    max_length=2,
                    device="cpu",
                )


class ColabProSSTSaturationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas
        import torch

        from saprot.scripts import saturation_mutagenesis_prosst

        cls.pd = pandas
        cls.torch = torch
        cls.saturation = saturation_mutagenesis_prosst

    class FakeTokenizer:
        def __init__(self):
            self.vocab = {
                amino_acid: index + 3
                for index, amino_acid in enumerate(
                    ColabProSSTSaturationTest.saturation.CANONICAL_AMINO_ACIDS
                )
            }

        def get_vocab(self):
            return self.vocab

        def __call__(self, sequences, return_tensors="pt"):
            rows = [
                [1, *[self.vocab[residue] for residue in sequence], 2]
                for sequence in sequences
            ]
            return {
                "input_ids": ColabProSSTSaturationTest.torch.tensor(rows),
                "attention_mask": ColabProSSTSaturationTest.torch.ones(
                    (len(rows), len(rows[0])),
                    dtype=ColabProSSTSaturationTest.torch.long,
                ),
            }

    class FakeModel:
        def __init__(self):
            self.ss_input_ids = None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(
            self,
            input_ids,
            attention_mask,
            ss_input_ids,
            return_dict,
        ):
            self.ss_input_ids = ss_input_ids.detach().cpu()
            logits = ColabProSSTSaturationTest.torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 25)
            )
            for token_id in range(3, 23):
                logits[:, :, token_id] = float(token_id)
            return types.SimpleNamespace(logits=logits)

    def test_scores_every_position_and_writes_centered_heatmap(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "saturation.csv"
            output_csv = root / "scores.csv"
            matrix_csv = root / "matrix.csv"
            heatmap_png = root / "heatmap.png"
            self.pd.DataFrame(
                [{"sequence": "AC", "structure_tokens": "0 1"}]
            ).to_csv(input_csv, index=False)
            fake_model = self.FakeModel()

            with patch.object(
                self.saturation.AutoTokenizer,
                "from_pretrained",
                return_value=self.FakeTokenizer(),
            ), patch.object(
                self.saturation.AutoModelForMaskedLM,
                "from_pretrained",
                return_value=fake_model,
            ):
                result = self.saturation.score_saturation_mutagenesis(
                    input_csv=str(input_csv),
                    output_csv=str(output_csv),
                    output_matrix_csv=str(matrix_csv),
                    output_heatmap_png=str(heatmap_png),
                    model_path="AI4Protein/ProSST-20",
                    device="cpu",
                )

            scores = result["score_table"].set_index("mutation")["score"]
            self.assertEqual(len(scores), 40)
            self.assertAlmostEqual(scores["A1A"], 0.0, places=6)
            self.assertAlmostEqual(scores["A1C"], 1.0, places=6)
            self.assertAlmostEqual(scores["C2A"], -1.0, places=6)
            self.assertEqual(result["score_matrix"].shape, (20, 2))
            self.assertEqual(
                result["matrix_table"].columns.tolist(),
                ["mutant", "A1", "C2"],
            )
            self.assertEqual(
                fake_model.ss_input_ids.tolist(),
                [[1, 3, 4, 2]],
            )
            self.assertTrue(output_csv.exists())
            self.assertTrue(matrix_csv.exists())
            self.assertGreater(heatmap_png.stat().st_size, 0)

    def test_requires_exactly_one_canonical_protein(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_csv = root / "multiple.csv"
            output_args = {
                "output_csv": str(root / "scores.csv"),
                "output_matrix_csv": str(root / "matrix.csv"),
                "output_heatmap_png": str(root / "heatmap.png"),
                "model_path": "AI4Protein/ProSST-20",
                "device": "cpu",
            }
            self.pd.DataFrame(
                [
                    {"sequence": "AC", "structure_tokens": "0 1"},
                    {"sequence": "AD", "structure_tokens": "0 2"},
                ]
            ).to_csv(input_csv, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one protein row"):
                self.saturation.score_saturation_mutagenesis(
                    input_csv=str(input_csv),
                    **output_args,
                )

            self.pd.DataFrame(
                [{"sequence": "AX", "structure_tokens": "0 1"}]
            ).to_csv(input_csv, index=False)
            with self.assertRaisesRegex(ValueError, "canonical amino acids"):
                self.saturation.score_saturation_mutagenesis(
                    input_csv=str(input_csv),
                    **output_args,
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
        fake_model = types.ModuleType("saprot.model")
        fake_model.__path__ = []
        fake_prosst = types.ModuleType("saprot.model.prosst")
        fake_prosst.__path__ = []
        fake_specs = types.ModuleType("saprot.model.prosst.specs")
        fake_specs.PROSST_MODEL_SPECS = tuple(
            types.SimpleNamespace(
                model_path=f"AI4Protein/ProSST-{vocab_size}",
                structure_vocab_size=vocab_size,
                display_name=f"Official ProSST ({vocab_size})",
            )
            for vocab_size in [20, 128, 512, 1024, 2048, 4096]
        )
        fake_specs.DEFAULT_PROSST_MODEL = fake_specs.PROSST_MODEL_SPECS[4]
        fake_specs.get_prosst_model_spec = lambda model_path: next(
            spec
            for spec in fake_specs.PROSST_MODEL_SPECS
            if spec.model_path == model_path
        )

        module_name = "_colab_prosst_ui_widget_test"
        spec = importlib.util.spec_from_file_location(module_name, UI_PATH)
        module = importlib.util.module_from_spec(spec)
        replacements = {
            "saprot": fake_saprot,
            "saprot.utils": fake_utils,
            "saprot.model": fake_model,
            "saprot.model.prosst": fake_prosst,
            "saprot.model.prosst.specs": fake_specs,
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
        self.assertIn("CSV already contains", input_guide.value)
        self.assertIn("structure_tokens", input_guide.value)
        self.assertIn("absolute Colab paths", input_guide.value)
        self.assertIn("Protein-pair tasks", input_guide.value)
        self.assertIn("cannot be reused for a pair", input_guide.value)

        model_dropdown = ui._model_dropdown()
        self.assertFalse(model_dropdown.disabled)
        self.assertEqual(
            [value for _label, value in model_dropdown.options],
            [spec.model_path for spec in fake_specs.PROSST_MODEL_SPECS],
        )
        self.assertEqual(
            model_dropdown.value,
            "AI4Protein/ProSST-2048",
        )

        task_dropdown = ui._task_dropdown()
        self.assertIn(
            ("Residue-level Classification", "token_classification"),
            task_dropdown.options,
        )
        self.assertIn(
            ("Protein-pair Classification", "pair_classification"),
            task_dropdown.options,
        )
        self.assertIn(
            ("Protein-pair Regression", "pair_regression"),
            task_dropdown.options,
        )
        self.assertIn(
            "one category for each amino-acid residue",
            ui._task_intro("token_classification"),
        )
        self.assertIn(
            "residue_labels",
            ui._training_dataset_help("token_classification"),
        )
        self.assertIn(
            "aligned one-to-one with the input residues",
            ui._prediction_output_help("token_classification"),
        )
        self.assertIn(
            "whether they interact",
            ui._task_intro("pair_classification"),
        )
        self.assertIn(
            "sequence_1",
            ui._training_dataset_help("pair_regression"),
        )
        self.assertTrue(ui._uses_category_count("pair_classification"))
        self.assertFalse(ui._uses_category_count("pair_regression"))

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
            pair_tokens_csv = root / "pair-tokens.csv"
            pair_tokens_csv.write_text(
                "sequence_1,sequence_2,structure_tokens_1,structure_tokens_2\n"
                "ACD,AC,\"1 2 3\",\"4 5\"\n",
                encoding="utf-8",
            )
            incomplete_pair_csv = root / "incomplete-pair.csv"
            incomplete_pair_csv.write_text(
                "sequence_1,sequence_2,structure_tokens_1\n"
                "ACD,AC,\"1 2 3\"\n",
                encoding="utf-8",
            )
            pair_paths_csv = root / "pair-paths.csv"
            pair_paths_csv.write_text(
                "sequence_1,sequence_2,pdb_path_1,structure_path_2\n"
                "ACD,AC,protein_1.pdb,protein_2.cif\n",
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

            structure_input.set_pair_mode(True)
            self.assertEqual(structure_input.mode.value, structure_input.TOKENS)
            self.assertNotIn(
                structure_input.REUSE,
                [value for _label, value in structure_input.mode.options],
            )
            self.assertIn("structure_tokens_1", structure_input.hint.value)
            structure_input.validate(pair_tokens_csv)
            with self.assertRaisesRegex(ValueError, "structure_tokens_2"):
                structure_input.validate(incomplete_pair_csv)

            structure_input.mode.value = structure_input.PATHS
            structure_input.validate(pair_paths_csv)
            with self.assertRaisesRegex(
                ValueError,
                r"protein\(s\): \[1, 2\]",
            ):
                structure_input.validate(paths_csv)

            structure_input.set_pair_mode(False)
            self.assertIn(
                structure_input.REUSE,
                [value for _label, value in structure_input.mode.options],
            )

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
            ui._embedding_page,
            ui._saturation_page,
            ui._mutation_page,
            ui._structure_page,
            ui._share_page,
        ]
        for page in pages:
            with self.subTest(page=page.__name__):
                rendered.clear()
                page()
                self.assertTrue(rendered)

        rendered.clear()
        ui._prediction_menu_page()
        prediction_menu_items = rendered[-1]
        self.assertTrue(
            any(
                getattr(item, "description", "")
                == "Extract protein embeddings"
                for item in prediction_menu_items
            )
        )
        self.assertTrue(
            any(
                getattr(item, "description", "")
                == "Single-site saturation mutagenesis"
                for item in prediction_menu_items
            )
        )

        rendered.clear()
        ui._embedding_page()
        embedding_items = rendered[-1]
        embedding_level = next(
            item
            for item in embedding_items
            if getattr(item, "description", "") == "Embedding level:"
        )
        self.assertEqual(
            list(embedding_level.options),
            [
                ("Protein-level", "protein"),
                ("Residue-level", "residue"),
                ("Both", "both"),
            ],
        )
        self.assertTrue(
            any(
                "[L, D]" in str(getattr(item, "value", ""))
                and "CLS, EOS, and padding" in str(getattr(item, "value", ""))
                for item in embedding_items
            )
        )

        rendered.clear()
        ui._saturation_page()
        saturation_items = rendered[-1]
        self.assertTrue(
            any(
                "exactly one protein row"
                in str(getattr(item, "value", ""))
                and "20 x L" in str(getattr(item, "value", ""))
                for item in saturation_items
            )
        )
        self.assertTrue(
            any(
                getattr(item, "description", "")
                == "Download saturation ZIP"
                for item in saturation_items
            )
        )

        rendered.clear()
        ui._training_page()
        training_items = rendered[-1]
        training_task = next(
            item
            for item in training_items
            if getattr(item, "description", "") == "Task type:"
        )
        category_count = next(
            item
            for item in training_items
            if getattr(item, "description", "") == "Number of categories:"
        )
        training_csv = next(
            item
            for item in training_items
            if getattr(item, "description", "") == "Training CSV:"
        )
        training_structure = next(
            item
            for item in training_items
            if getattr(item, "description", "") == "Structure input:"
        )
        training_task.value = "token_classification"
        self.assertIsNone(category_count.layout.display)
        self.assertIn("residue_labels", training_csv.placeholder)
        self.assertTrue(
            any(
                "one integer category per residue"
                in str(getattr(item, "value", ""))
                for item in training_items
            )
        )
        training_task.value = "pair_classification"
        self.assertIsNone(category_count.layout.display)
        self.assertIn("sequence_1", training_csv.placeholder)
        self.assertNotIn(
            "reuse",
            [value for _label, value in training_structure.options],
        )
        training_task.value = "pair_regression"
        self.assertEqual(category_count.layout.display, "none")

        rendered.clear()
        ui.latest_task_type = "token_classification"
        ui.latest_num_labels = 3
        ui._property_prediction_page()
        prediction_items = rendered[-1]
        prediction_task = next(
            item
            for item in prediction_items
            if getattr(item, "description", "") == "Task type:"
        )
        prediction_categories = next(
            item
            for item in prediction_items
            if getattr(item, "description", "") == "Number of categories:"
        )
        prediction_csv = next(
            item
            for item in prediction_items
            if getattr(item, "description", "") == "Prediction CSV:"
        )
        prediction_structure = next(
            item
            for item in prediction_items
            if getattr(item, "description", "") == "Structure input:"
        )
        self.assertEqual(prediction_task.value, "token_classification")
        self.assertEqual(prediction_categories.value, 3)
        self.assertTrue(
            any(
                "aligned one-to-one with the input residues"
                in str(getattr(item, "value", ""))
                for item in prediction_items
            )
        )
        prediction_task.value = "regression"
        self.assertEqual(prediction_categories.layout.display, "none")
        prediction_task.value = "token_classification"
        self.assertIsNone(prediction_categories.layout.display)
        prediction_task.value = "pair_classification"
        self.assertIsNone(prediction_categories.layout.display)
        self.assertIn("sequence_1", prediction_csv.placeholder)
        self.assertNotIn(
            "reuse",
            [value for _label, value in prediction_structure.options],
        )
        prediction_task.value = "pair_regression"
        self.assertEqual(prediction_categories.layout.display, "none")

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

        with patch.object(
            module.pkgutil,
            "get_data",
            return_value=b"window.colabUploadTest = true;",
        ):
            styled_upload_field = module._UploadField(
                ui,
                "Training CSV:",
                "Choose CSV",
            )
        upload_html = styled_upload_field.inline_upload.value
        self.assertIn(">Choose file</label>", upload_html)
        self.assertIn(">No file selected</span>", upload_html)
        self.assertIn("clip-path: inset(50%)", upload_html)
        self.assertIn("type=\"file\"", upload_html)

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
