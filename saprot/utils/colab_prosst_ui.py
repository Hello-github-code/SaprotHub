import base64
import collections
import csv
import ctypes
import pkgutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from saprot.utils.colab_prosst_workflow import MODEL_PROSST_2048


class _UploadField:
    def __init__(self, ui, description, placeholder):
        self.ui = ui
        widgets = ui.widgets
        upload_id = uuid.uuid4().hex
        self.input_id = f"colabprosst-files-{upload_id}"
        self.output_id = f"colabprosst-result-{upload_id}"
        self.path = widgets.Text(
            value="",
            placeholder=placeholder,
            description=description,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.upload_button = widgets.Button(
            description="Upload your file",
            button_style="info",
            layout=widgets.Layout(width=ui.WIDTH, height=ui.HEIGHT),
        )
        self.inline_upload = widgets.HTML(
            value=self._inline_upload_html(),
            layout=widgets.Layout(width=ui.WIDTH, display="none"),
        )
        self.status = widgets.HTML(layout=widgets.Layout(width=ui.WIDTH))
        self.upload_button.on_click(self._upload)
        self.items = [
            self.path,
            self.upload_button,
            self.inline_upload,
            self.status,
        ]

    def _inline_upload_html(self):
        try:
            files_js = pkgutil.get_data(
                "google.colab.files", "resources/files.js"
            )
        except (ImportError, ModuleNotFoundError):
            return ""
        if files_js is None:
            return ""

        return """
            <input type="file" id="{input_id}" name="files[]" disabled
                   style="border:none" />
            <output id="{output_id}"></output>
            <script>{files_js}</script>
        """.format(
            input_id=self.input_id,
            output_id=self.output_id,
            files_js=files_js.decode("utf-8"),
        )

    def _upload_inline(self):
        from google.colab import output

        output.eval_js(
            'document.getElementById("{input_id}").value = ""'.format(
                input_id=self.input_id
            )
        )
        result = output.eval_js(
            'google.colab._files._uploadFiles("{input_id}", "{output_id}")'.format(
                input_id=self.input_id,
                output_id=self.output_id,
            )
        )
        uploaded_files = collections.defaultdict(bytes)
        while result["action"] != "complete":
            result = output.eval_js(
                'google.colab._files._uploadFilesContinue("{output_id}")'.format(
                    output_id=self.output_id
                )
            )
            if result["action"] == "append":
                uploaded_files[result["file"]] += base64.b64decode(result["data"])

        if not uploaded_files:
            return None
        filename, content = next(iter(uploaded_files.items()))
        save_uploaded_content = getattr(
            self.ui.workflow, "save_uploaded_content", None
        )
        if callable(save_uploaded_content):
            return save_uploaded_content(filename, content)

        safe_name = Path(str(filename).replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Uploaded file must have a valid filename.")
        save_path = Path(self.ui.workflow.upload_dir) / safe_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(bytes(content))
        return str(save_path)

    @property
    def value(self):
        return self.path.value.strip()

    @value.setter
    def value(self, path):
        self.path.value = str(path or "")

    def set_visible(self, visible):
        display = None if visible else "none"
        self.path.layout.display = display
        self.upload_button.layout.display = display
        self.status.layout.display = display
        if not visible:
            self.inline_upload.layout.display = "none"

    def _upload(self, _button):
        self.upload_button.disabled = True
        self.status.value = "Choose one file below, or cancel the upload."
        try:
            if self.inline_upload.value:
                self.inline_upload.layout.display = None
                uploaded_path = self._upload_inline()
            else:
                uploaded_path = self.ui.workflow.maybe_upload_path("", True)
            if uploaded_path is None:
                self.status.value = "Upload canceled."
                return
            self.value = uploaded_path
            self.status.value = f"Uploaded: {Path(uploaded_path).name}"
        except Exception as exc:
            self.status.value = f"<font color='red'>{type(exc).__name__}: {exc}</font>"
        finally:
            self.inline_upload.layout.display = "none"
            self.upload_button.disabled = False


class _StructureInput:
    TOKENS = "tokens"
    REUSE = "reuse"
    PATHS = "paths"

    def __init__(self, ui):
        self.ui = ui
        widgets = ui.widgets
        default_mode = (
            self.REUSE
            if getattr(ui.workflow, "last_structure", None) is not None
            else self.TOKENS
        )
        self.mode = widgets.RadioButtons(
            options=[
                ("CSV contains structure_tokens", self.TOKENS),
                ("Reuse latest structure conversion", self.REUSE),
                ("CSV contains structure file paths", self.PATHS),
            ],
            value=default_mode,
            description="Structure input:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%", max_width=ui.GUIDE_WIDTH),
        )
        self.hint = widgets.HTML(
            layout=widgets.Layout(
                width="100%",
                max_width=ui.GUIDE_WIDTH,
                overflow="visible",
            )
        )
        self.zip_upload = _UploadField(
            ui,
            "Structure ZIP:",
            "ZIP containing PDB/mmCIF files referenced by the CSV",
        )
        self.items = [self.mode, self.hint, *self.zip_upload.items]
        self.mode.observe(self._update, names="value")
        self._update({"new": self.mode.value})

    @property
    def reuse_latest(self):
        return self.mode.value == self.REUSE

    @property
    def structure_zip(self):
        if self.mode.value != self.PATHS:
            return ""
        return self.zip_upload.value

    def validate(self, csv_path):
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = {column.strip().lower() for column in next(reader)}
            except StopIteration as exc:
                raise ValueError("The uploaded CSV is empty.") from exc

        if self.mode.value == self.TOKENS and "structure_tokens" not in columns:
            raise ValueError(
                "You selected `CSV contains structure_tokens`, but the uploaded "
                "CSV has no structure_tokens column. Choose another structure "
                "input mode or add that column."
            )
        if self.mode.value == self.PATHS and not columns.intersection(
            {"pdb_path", "structure_path"}
        ):
            raise ValueError(
                "You selected `CSV contains structure file paths`, but the "
                "uploaded CSV has no pdb_path or structure_path column."
            )
        if self.mode.value == self.REUSE and getattr(
            self.ui.workflow, "last_structure", None
        ) is None:
            raise ValueError(
                "No structure conversion is available in this Colab session. "
                "Run `Convert protein structure to ProSST tokens` first."
            )

    def _update(self, change):
        mode = change["new"]
        self.zip_upload.set_visible(mode == self.PATHS)

        if mode == self.TOKENS:
            self.hint.value = (
                "Every CSV row must contain <code>structure_tokens</code>. "
                "Upload only the CSV; a Structure ZIP is not needed."
            )
        elif mode == self.REUSE:
            last_structure = getattr(self.ui.workflow, "last_structure", None)
            if last_structure is None:
                availability = (
                    "<br><font color='red'>No structure has been converted in "
                    "this session yet.</font>"
                )
            else:
                sequence_length = len(last_structure.get("sequence", ""))
                availability = (
                    f"<br>Latest conversion available: {sequence_length} residues."
                )
            self.hint.value = (
                "First run <b>Convert protein structure to ProSST tokens</b>. "
                "Then upload a CSV whose every row has the same sequence. "
                "Do not upload a Structure ZIP."
                + availability
            )
        else:
            self.hint.value = (
                "Every CSV row must contain <code>pdb_path</code> or "
                "<code>structure_path</code>. Upload a Structure ZIP when those "
                "values are filenames or relative paths. Existing absolute Colab "
                "paths do not require a ZIP."
            )


class ColabProSSTUI:
    """ColabSaprot-style interactive interface backed by ColabProSSTWorkflow."""

    WIDTH = "500px"
    HEIGHT = "30px"
    GUIDE_WIDTH = "720px"

    def __init__(self, workflow):
        try:
            import ipywidgets
            from IPython.display import clear_output, display
        except Exception as exc:
            raise RuntimeError(
                "ColabProSSTUI requires ipywidgets and an IPython notebook runtime."
            ) from exc

        self.widgets = ipywidgets
        self.display = display
        self.clear_output = clear_output
        self.workflow = workflow
        self.current_page = None
        self.active_thread = None
        self.latest_checkpoint = ""
        self._polling = False
        self._build_system_widgets()

    def _html(self, value, **layout_kwargs):
        return self.widgets.HTML(
            value=value,
            layout=self.widgets.Layout(**layout_kwargs),
        )

    def _heading(self, text, level=2):
        return self._html(f"<h{level}>{text}</h{level}>")

    def _separator(self):
        return self._html("<h3>---------------------------------------------------------------------------</h3>")

    def _button(self, description, width=None, style=""):
        return self.widgets.Button(
            description=description,
            button_style=style,
            layout=self.widgets.Layout(
                width=width or self.WIDTH,
                height=self.HEIGHT,
            ),
        )

    def _model_dropdown(self):
        return self.widgets.Dropdown(
            options=[("Official ProSST (2048)", MODEL_PROSST_2048)],
            value=MODEL_PROSST_2048,
            description="Base model:",
            disabled=True,
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    def _task_dropdown(self):
        return self.widgets.Dropdown(
            options=[
                ("Protein-level Classification", "classification"),
                ("Protein-level Regression", "regression"),
            ],
            value="classification",
            description="Task type:",
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    @staticmethod
    def _task_intro(task_type):
        if task_type == "classification":
            return (
                "<font color='red'>What is <b>Protein-level Classification:</b> "
                "Given a protein, you have some categories and you want to "
                "predict which category the protein belongs to.</font>"
            )
        return (
            "<font color='red'>What is <b>Protein-level Regression:</b> Given a "
            "protein, you want to predict a score about its property such as "
            "stability or enzyme activity.</font>"
        )

    def _num_labels(self):
        return self.widgets.BoundedIntText(
            value=2,
            min=2,
            max=100000000,
            step=1,
            description="Number of categories:",
            style={"description_width": "initial"},
            layout=self.widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )

    def _output(self):
        return self.widgets.Output(
            layout=self.widgets.Layout(width="100%", border="0")
        )

    def _input_guide(self):
        return self._html(
            "<h3>Prepare sequence and structure inputs</h3>"
            "<p>Every protein needs an amino-acid sequence and matching ProSST "
            "structure tokens. Choose one input method:</p>"
            "<ol>"
            "<li><b>Recommended for a first run:</b> convert one PDB/mmCIF "
            "structure, return to the task, and select <b>Reuse latest structure "
            "conversion</b>. Every CSV row must contain that same sequence.</li>"
            "<li><b>CSV contains structure_tokens:</b> upload only the CSV. This "
            "is the best option for repeated use and future Colab sessions.</li>"
            "<li><b>CSV contains structure file paths:</b> include "
            "<code>pdb_path</code> or <code>structure_path</code>, then upload a "
            "ZIP when those paths are relative filenames.</li>"
            "</ol>",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
        )

    def _build_system_widgets(self):
        self.back_button = self._button("Go back", width="120px", style="success")
        self.refresh_button = self._button("Refresh", width="120px", style="success")
        self.stop_button = self._button("Stop", width="120px", style="danger")
        self.system_status = self.widgets.Output(
            layout=self.widgets.Layout(width=self.WIDTH)
        )

        self.back_button.on_click(lambda _button: self._navigate(self._home_page))
        self.refresh_button.on_click(lambda _button: self._navigate(self.current_page))
        self.stop_button.on_click(lambda _button: self.stop_task())

        self.system_widgets = [
            self._html(
                "<b><font color='red'>Note: At any time you can use the buttons "
                "below to stop and restart.</font></b>"
            ),
            self.widgets.HBox(
                [self.back_button, self.refresh_button, self.stop_button]
            ),
            self._html(
                "<b>Go back:</b> stop the running task and return to the first "
                "interface.<br><b>Refresh:</b> stop the running task and reset the "
                "current interface.<br><b>Stop:</b> stop the running task."
            ),
            self.system_status,
        ]

    def _navigate(self, page):
        if page is None:
            return
        self.stop_task(silent=True)
        self.current_page = page
        self.clear_output(wait=True)
        page()
        self.display(*self.system_widgets)

    def _start_task(self, button, output, action):
        if self.active_thread is not None and self.active_thread.is_alive():
            with output:
                print("A task is already running. Stop it before starting another one.")
            return

        def runner():
            button.disabled = True
            output.clear_output(wait=True)
            with output:
                try:
                    action()
                except SystemExit:
                    print("Task interrupted by user.")
                except Exception as exc:
                    print(f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                finally:
                    button.disabled = False
                    if self.active_thread is threading.current_thread():
                        self.active_thread = None

        self.active_thread = threading.Thread(target=runner, daemon=True)
        self.active_thread.start()

    def stop_task(self, silent=False):
        thread = self.active_thread
        if thread is None or not thread.is_alive():
            self.active_thread = None
            if not silent:
                self.system_status.clear_output(wait=True)
                with self.system_status:
                    print("No running task to be stopped.")
            return

        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident), ctypes.py_object(SystemExit)
        )
        if result == 0:
            raise RuntimeError("The running task thread no longer exists.")
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), None
            )
            raise RuntimeError("Could not stop the running task safely.")
        thread.join(timeout=2)
        if thread.is_alive():
            self.active_thread = thread
            if not silent:
                self.system_status.clear_output(wait=True)
                with self.system_status:
                    print(
                        "Stop requested. The task is finishing its current native "
                        "operation; wait before starting another task."
                    )
            return
        self.active_thread = None
        if not silent:
            self.system_status.clear_output(wait=True)
            with self.system_status:
                print("Task interrupted by user.")

    def _download_templates(self, button):
        output = self.system_status

        def action():
            self.workflow.create_csv_templates(download=True)

        self._start_task(button, output, action)

    def _home_page(self):
        self.current_page = self._home_page
        title = self._heading(
            "Please choose what you want to do with ColabProSST"
        )
        input_guide = self._input_guide()
        train_button = self._button(
            "I want to train my own model", width="400px"
        )
        predict_button = self._button(
            "I want to use existing models to make prediction", width="400px"
        )
        share_button = self._button(
            "I want to share my model publicly", width="400px"
        )

        train_button.on_click(lambda _button: self._navigate(self._training_page))
        predict_button.on_click(
            lambda _button: self._navigate(self._prediction_menu_page)
        )
        share_button.on_click(lambda _button: self._navigate(self._share_page))

        self.display(
            title,
            train_button,
            predict_button,
            share_button,
            input_guide,
        )

    def _training_page(self):
        self.current_page = self._training_page
        widgets = self.widgets
        task_name = widgets.Text(
            value="ProSSTUserTask",
            description="Name your task:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        task_type = self._task_dropdown()
        num_labels = self._num_labels()
        task_intro = self._html(
            self._task_intro("classification"), width=self.WIDTH
        )
        model = self._model_dropdown()
        csv_input = _UploadField(
            self,
            "Training CSV:",
            "Path to a CSV with sequence, label, stage, and structure input",
        )
        structure_input = _StructureInput(self)
        template_button = self._button(
            "Download CSV templates", width="220px"
        )
        batch_size = widgets.Dropdown(
            options=[1, 2, 4, 8, 16, 32],
            value=1,
            description="Batch size:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        epochs = widgets.BoundedIntText(
            value=2,
            min=1,
            max=100,
            description="Epoch:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        learning_rate = widgets.FloatText(
            value=2.0e-5,
            description="Learning rate:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        advanced_button = self._button("Show advanced settings")
        freeze_backbone = widgets.Checkbox(
            value=True,
            description="Freeze ProSST backbone",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        gradient_checkpointing = widgets.Checkbox(
            value=True,
            description="Use gradient checkpointing",
            style={"description_width": "initial"},
            layout=widgets.Layout(display="none"),
        )
        download = widgets.Checkbox(
            value=True,
            description="Download checkpoint and test predictions",
            style={"description_width": "initial"},
        )
        start_button = self._button("Start training", style="info")
        output = self._output()
        finish_hint = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def update_task(change):
            num_labels.layout.display = None if change["new"] == "classification" else "none"
            task_intro.value = self._task_intro(change["new"])

        def toggle_advanced(_button):
            show = freeze_backbone.layout.display == "none"
            mode = None if show else "none"
            freeze_backbone.layout.display = mode
            gradient_checkpointing.layout.display = mode
            advanced_button.description = (
                "Hide advanced settings" if show else "Show advanced settings"
            )

        def train():
            if not csv_input.value:
                raise ValueError("Upload a training CSV or enter its path.")
            clean_task_name = task_name.value.strip()
            if not clean_task_name or Path(clean_task_name).name != clean_task_name:
                raise ValueError(
                    "Task name must be a non-empty file-name-safe value."
                )
            structure_input.validate(csv_input.value)
            print("Start training...")
            result = self.workflow.train_downstream(
                task_type=task_type.value,
                input_csv=csv_input.value,
                use_last_structure_tokens=structure_input.reuse_latest,
                structure_zip=structure_input.structure_zip,
                task_name=clean_task_name,
                num_labels=num_labels.value,
                max_epochs=epochs.value,
                batch_size=batch_size.value,
                learning_rate=learning_rate.value,
                model_path=model.value,
                freeze_backbone=freeze_backbone.value,
                gradient_checkpointing=gradient_checkpointing.value,
                download=download.value,
            )
            self.latest_checkpoint = result["checkpoint_path"]
            print("Training finished.")
            print("Model checkpoint:", result["checkpoint_path"])
            print("Test predictions:", result["test_result_csv"])
            finish_hint.value = (
                "<h3>The training is completed. You can then:</h3>"
                "<ul>"
                "<li><b>Train again:</b> click <b>Refresh</b>, adjust the "
                "settings, and start a new task.</li>"
                "<li><b>Use this model for prediction:</b> click <b>Go back</b>, "
                "choose <b>I want to use existing models to make prediction</b>, "
                "then choose <b>Protein property prediction</b>. The checkpoint "
                "is selected automatically in this session.</li>"
                "<li><b>Share this model:</b> click <b>Go back</b> and choose "
                "<b>I want to share my model publicly</b>.</li>"
                "</ul>"
            )
            finish_hint.layout.display = None

        task_type.observe(update_task, names="value")
        template_button.on_click(self._download_templates)
        advanced_button.on_click(toggle_advanced)
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, train)
        )

        self.display(
            self._heading("Please finish the setting of your training task"),
            self._heading("Task setting:", level=3),
            task_name,
            task_type,
            num_labels,
            task_intro,
            self._heading("Model setting:", level=3),
            model,
            self._heading("Dataset setting:", level=3),
            self._html(
                "The CSV must contain <code>sequence</code>, <code>label</code>, "
                "and <code>stage</code>. Then choose one structure input method."
            ),
            *csv_input.items,
            *structure_input.items,
            template_button,
            self._heading("Training hyper-parameters:", level=3),
            batch_size,
            epochs,
            learning_rate,
            advanced_button,
            freeze_backbone,
            gradient_checkpointing,
            download,
            self._separator(),
            start_button,
            output,
            finish_hint,
        )

    def _prediction_menu_page(self):
        self.current_page = self._prediction_menu_page
        property_button = self._button(
            "Protein property prediction", style="info"
        )
        mutation_button = self._button(
            "Mutational effect prediction", style="info"
        )
        structure_button = self._button(
            "Convert protein structure to ProSST tokens", style="info"
        )
        template_button = self._button("Download CSV templates")
        property_button.on_click(
            lambda _button: self._navigate(self._property_prediction_page)
        )
        mutation_button.on_click(
            lambda _button: self._navigate(self._mutation_page)
        )
        structure_button.on_click(
            lambda _button: self._navigate(self._structure_page)
        )
        template_button.on_click(self._download_templates)
        self.display(
            self._heading(
                "ColabProSST supports multiple prediction tasks, which one "
                "would you like to choose?"
            ),
            self._separator(),
            property_button,
            self._html(
                "Use a trained ProSST checkpoint for protein-level classification "
                "or regression."
            ),
            self._separator(),
            mutation_button,
            self._html(
                "Use official ProSST masked-language-model scores for zero-shot "
                "single-site or multi-site mutation effects."
            ),
            self._separator(),
            structure_button,
            self._html(
                "Convert a PDB or mmCIF structure into the structure tokens "
                "required by ProSST."
            ),
            self._separator(),
            template_button,
        )

    def _property_prediction_page(self):
        self.current_page = self._property_prediction_page
        widgets = self.widgets
        task_type = self._task_dropdown()
        num_labels = self._num_labels()
        task_intro = self._html(
            self._task_intro("classification"), width=self.WIDTH
        )
        model = self._model_dropdown()
        checkpoint = _UploadField(
            self,
            "Model checkpoint:",
            "Path to a ColabProSST .pt checkpoint",
        )
        checkpoint.value = self.latest_checkpoint
        csv_input = _UploadField(
            self,
            "Prediction CSV:",
            "Path to a CSV with sequence and structure input",
        )
        structure_input = _StructureInput(self)
        batch_size = widgets.Dropdown(
            options=[1, 2, 4, 8, 16, 32],
            value=1,
            description="Batch size:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        download = widgets.Checkbox(
            value=True,
            description="Download prediction CSV",
            style={"description_width": "initial"},
        )
        start_button = self._button("Start prediction", style="info")
        output = self._output()

        def update_task(change):
            num_labels.layout.display = None if change["new"] == "classification" else "none"
            task_intro.value = self._task_intro(change["new"])

        def predict():
            if not checkpoint.value:
                raise ValueError("Upload a model checkpoint or enter its path.")
            if not csv_input.value:
                raise ValueError("Upload a prediction CSV or enter its path.")
            structure_input.validate(csv_input.value)
            print("Start prediction...")
            result = self.workflow.predict_downstream(
                task_type=task_type.value,
                input_csv=csv_input.value,
                checkpoint_path=checkpoint.value,
                use_last_structure_tokens=structure_input.reuse_latest,
                structure_zip=structure_input.structure_zip,
                num_labels=num_labels.value,
                batch_size=batch_size.value,
                model_path=model.value,
                download=download.value,
            )
            self.display(result.head())

        task_type.observe(update_task, names="value")
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, predict)
        )
        self.display(
            self._heading("Protein property prediction"),
            self._heading("Choose the prediction task:", level=3),
            task_type,
            num_labels,
            task_intro,
            self._heading("Choose the model for prediction:", level=3),
            model,
            *checkpoint.items,
            self._heading("Input proteins:", level=3),
            *csv_input.items,
            *structure_input.items,
            batch_size,
            download,
            self._separator(),
            start_button,
            output,
        )

    def _mutation_page(self):
        self.current_page = self._mutation_page
        widgets = self.widgets
        model = self._model_dropdown()
        csv_input = _UploadField(
            self,
            "Mutation CSV:",
            "Path to a CSV with sequence, mutant, and structure input",
        )
        structure_input = _StructureInput(self)
        download = widgets.Checkbox(
            value=True,
            description="Download mutation score CSV",
            style={"description_width": "initial"},
        )
        start_button = self._button("Start prediction", style="info")
        output = self._output()

        def predict():
            if not csv_input.value:
                raise ValueError("Upload a mutation CSV or enter its path.")
            structure_input.validate(csv_input.value)
            print("Start mutational effect prediction...")
            result = self.workflow.run_zero_shot(
                input_csv=csv_input.value,
                use_last_structure_tokens=structure_input.reuse_latest,
                structure_zip=structure_input.structure_zip,
                model_path=model.value,
                download=download.value,
            )
            self.display(result.head())

        start_button.on_click(
            lambda _button: self._start_task(start_button, output, predict)
        )
        self.display(
            self._heading("Mutational effect prediction"),
            self._heading("Model setting:", level=3),
            model,
            self._heading("Mutation data:", level=3),
            self._html(
                "The CSV must contain <code>sequence</code> and "
                "<code>mutant</code>. Then choose one structure input method."
            ),
            *csv_input.items,
            *structure_input.items,
            download,
            self._separator(),
            start_button,
            output,
        )

    def _structure_page(self):
        self.current_page = self._structure_page
        widgets = self.widgets
        structure = _UploadField(
            self,
            "Structure file:",
            "Path to one PDB or mmCIF file",
        )
        chain = widgets.Text(
            value="",
            placeholder="Leave empty to use all chains",
            description="Chain:",
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        vocab = widgets.Dropdown(
            options=[2048],
            value=2048,
            description="Structure vocab:",
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        download = widgets.Checkbox(
            value=True,
            description="Download structure token CSV",
            style={"description_width": "initial"},
        )
        start_button = self._button("Convert structure", style="info")
        output = self._output()
        next_steps = self._html(
            "",
            width="100%",
            max_width=self.GUIDE_WIDTH,
            overflow="visible",
            display="none",
        )

        def convert():
            if not structure.value:
                raise ValueError("Upload a PDB/mmCIF file or enter its path.")
            print("Converting structure to ProSST tokens...")
            result = self.workflow.convert_structure(
                structure_path=structure.value,
                chain_id=chain.value,
                structure_vocab_size=vocab.value,
                download=download.value,
            )
            self.display(result)
            next_steps.value = (
                "<h3>Use these tokens in your next task</h3>"
                "<ol>"
                "<li>Click <b>Go back</b> below and open training or the "
                "prediction task you need.</li>"
                "<li>Upload a CSV containing the same amino-acid sequence.</li>"
                "<li>Select <b>Reuse latest structure conversion</b>. Do not "
                "upload a Structure ZIP.</li>"
                "</ol>"
                "The shortcut lasts only for this running Colab session. For a "
                "future session, keep the downloaded conversion CSV and place "
                "its <code>structure_tokens</code> in your task CSV."
            )
            next_steps.layout.display = None

        start_button.on_click(
            lambda _button: self._start_task(start_button, output, convert)
        )
        self.display(
            self._heading("Convert protein structure to ProSST tokens"),
            self._heading("Structure setting:", level=3),
            *structure.items,
            chain,
            vocab,
            download,
            self._separator(),
            start_button,
            output,
            next_steps,
        )

    def _share_page(self):
        self.current_page = self._share_page
        widgets = self.widgets
        repo_id = widgets.Text(
            value="",
            placeholder="username/Model-ProSST-Task",
            description="Repository ID:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        checkpoint = _UploadField(
            self,
            "Model checkpoint:",
            "Path to a trained ColabProSST .pt checkpoint",
        )
        checkpoint.value = self.latest_checkpoint
        task_type = self._task_dropdown()
        num_labels = self._num_labels()
        private = widgets.Checkbox(
            value=False,
            description="Create a private Hugging Face repository",
            style={"description_width": "initial"},
        )
        login = widgets.Checkbox(
            value=True,
            description="Open Hugging Face login",
            style={"description_width": "initial"},
        )
        title = widgets.Text(
            value="ColabProSST model",
            description="Model title:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height=self.HEIGHT),
        )
        description = widgets.Textarea(
            value="A ProSST checkpoint trained with ColabProSST.",
            description="Description:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=self.WIDTH, height="90px"),
        )
        start_button = self._button("Upload model", style="info")
        output = self._output()

        def update_task(change):
            num_labels.layout.display = None if change["new"] == "classification" else "none"

        def upload():
            if not checkpoint.value:
                raise ValueError("Upload a model checkpoint or enter its path.")
            print("Uploading model to Hugging Face...")
            package = self.workflow.upload_checkpoint_to_hf(
                repo_id=repo_id.value,
                checkpoint_path=checkpoint.value,
                task_type=task_type.value,
                num_labels=num_labels.value,
                private=private.value,
                run_login=login.value,
                title=title.value,
                description=description.value,
            )
            print("Local model package:", package)

        task_type.observe(update_task, names="value")
        start_button.on_click(
            lambda _button: self._start_task(start_button, output, upload)
        )
        self.display(
            self._heading("Share your ColabProSST model"),
            repo_id,
            *checkpoint.items,
            task_type,
            num_labels,
            private,
            login,
            title,
            description,
            self._separator(),
            start_button,
            output,
        )

    def launch(self, poll=True):
        """Display the first page and optionally keep Colab widget events alive."""
        self.current_page = self._home_page
        self._home_page()
        self.display(*self.system_widgets)

        if not poll:
            return self

        try:
            from jupyter_ui_poll import ui_events
        except Exception as exc:
            raise RuntimeError(
                "jupyter_ui_poll is required for the live ColabProSST interface."
            ) from exc

        self._polling = True
        try:
            with ui_events() as poll_events:
                while self._polling:
                    poll_events(10)
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._polling = False
            self.stop_task(silent=True)
            self.clear_output(wait=True)
            self.display(
                self._heading(
                    "The program is interrupted. Click the run-button to restart."
                )
            )


def launch_colabprosst(workflow, poll=True):
    return ColabProSSTUI(workflow).launch(poll=poll)
