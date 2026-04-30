from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

fake_pipelines = types.ModuleType("pipelines")
fake_pipelines.PipelineDescriptor = object
fake_pipelines.ProcessResult = object
fake_pipelines.load_pipeline_catalog = lambda: ([], [])
sys.modules.setdefault("pipelines", fake_pipelines)
sys.modules.setdefault("pipelines.core", types.ModuleType("pipelines.core"))

fake_pipeline_errors = types.ModuleType("pipelines.core.errors")
fake_pipeline_errors.format_pipeline_exception = lambda exc, _pipeline: str(exc)
sys.modules.setdefault("pipelines.core.errors", fake_pipeline_errors)

fake_pipeline_utils = types.ModuleType("pipelines.core.utils")
fake_pipeline_utils.append_result_group = lambda *args, **kwargs: None
fake_pipeline_utils.initialize_output_h5 = lambda *args, **kwargs: None
sys.modules.setdefault("pipelines.core.utils", fake_pipeline_utils)

from eye_flow import ProcessApp  # noqa: E402

for _module_name in (
    "pipelines",
    "pipelines.core",
    "pipelines.core.errors",
    "pipelines.core.utils",
):
    _module = sys.modules.get(_module_name)
    if _module is not None and getattr(_module, "__file__", None) is None:
        sys.modules.pop(_module_name, None)


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class BatchRunTests(unittest.TestCase):
    def _make_fake_app(
        self,
        *,
        holo_path: Path,
        hd_path: Path | None,
        dv_path: Path | None,
        base_output_dir: Path,
        zip_should_fail: bool,
    ):
        logs: list[str] = []

        def _run_pipelines_to_output(
            *,
            output_h5_path,
            pipelines,
            holodoppler_h5,
            doppler_vision_h5,
        ):
            output_h5_path.parent.mkdir(parents=True, exist_ok=True)
            output_h5_path.write_text("result", encoding="utf-8")
            return output_h5_path

        def _zip_output_dir(folder, target_path=None, progress_callback=None):
            if zip_should_fail:
                raise RuntimeError("zip failed")
            if progress_callback is not None:
                progress_callback(1, 1, Path("subject_eyeflow.h5"))
            assert target_path is not None
            target_path.write_text("archive", encoding="utf-8")
            return target_path

        pipeline = SimpleNamespace(name="Demo", available=True, input_slot="both")
        app = SimpleNamespace(
            batch_output_var=_Var(str(base_output_dir)),
            batch_zip_var=_Var(True),
            batch_zip_name_var=_Var("outputs.zip"),
            _progress_primary_style="MinimalPrimary.Horizontal.TProgressbar",
            _progress_final_style="MinimalFinal.Horizontal.TProgressbar",
            _progress_total_units=1.0,
            pipeline_rows=[pipeline],
            pipeline_visibility={"Demo": True},
            pipeline_registry={"Demo": pipeline},
            _selected_holo_paths=lambda: [holo_path],
            _validate_selected_inputs=lambda *_args: [
                SimpleNamespace(
                    holo_path=holo_path,
                    data_dir=holo_path.parent / holo_path.stem,
                    hd_h5=hd_path,
                    dv_h5=dv_path,
                )
            ],
            _replace_existing_output_dir_if_needed=lambda *args, **kwargs: False,
            _reset_batch_output=lambda *args, **kwargs: None,
            _log_batch=logs.append,
            _start_progress=lambda total_units, **kwargs: setattr(
                app, "_progress_total_units", total_units
            ),
            _set_progress_units=lambda *args, **kwargs: None,
            _set_minimal_status=lambda *args, **kwargs: None,
            _reset_progress=lambda: None,
            _advance_progress=lambda units=1.0: None,
            _minimal_output_filename_for_run=lambda: None,
            _default_work_h5_name=lambda: "subject_eyeflow.h5",
            _next_available_output_path=lambda path: path,
            _run_pipelines_to_output=_run_pipelines_to_output,
            _zip_output_dir=_zip_output_dir,
            update=lambda: None,
            logs=logs,
        )
        return app
    @mock.patch("eye_flow.messagebox.showwarning")
    @mock.patch("eye_flow.messagebox.showerror")
    def test_run_batch_removes_temp_output_dir_after_successful_zip(
        self,
        _showerror,
        _showwarning,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_path = tmp_path / "subject.holo"
            holo_path.write_text("dummy", encoding="utf-8")
            hd_path = tmp_path / "subject_holodoppler.h5"
            hd_path.write_text("dummy", encoding="utf-8")
            dv_path = tmp_path / "subject_doppler_vision.h5"
            dv_path.write_text("dummy", encoding="utf-8")
            base_output_dir = tmp_path / "outputs"
            base_output_dir.mkdir()

            app = self._make_fake_app(
                holo_path=holo_path,
                hd_path=hd_path,
                dv_path=dv_path,
                base_output_dir=base_output_dir,
                zip_should_fail=False,
            )

            ProcessApp.run_batch(app)

            self.assertTrue((base_output_dir / "outputs.zip").exists())
            self.assertEqual(
                [base_output_dir / "outputs.zip"],
                sorted(base_output_dir.iterdir()),
            )
            self.assertTrue(
                any("Completed. ZIP archive:" in line for line in app.logs),
            )

    @mock.patch("eye_flow.messagebox.showwarning")
    @mock.patch("eye_flow.messagebox.showerror")
    def test_run_batch_keeps_work_dir_when_zip_creation_fails(
        self,
        showerror,
        _showwarning,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_path = tmp_path / "subject.holo"
            holo_path.write_text("dummy", encoding="utf-8")
            hd_path = tmp_path / "subject_holodoppler.h5"
            hd_path.write_text("dummy", encoding="utf-8")
            dv_path = tmp_path / "subject_doppler_vision.h5"
            dv_path.write_text("dummy", encoding="utf-8")
            base_output_dir = tmp_path / "outputs"
            base_output_dir.mkdir()

            app = self._make_fake_app(
                holo_path=holo_path,
                hd_path=hd_path,
                dv_path=dv_path,
                base_output_dir=base_output_dir,
                zip_should_fail=True,
            )

            ProcessApp.run_batch(app)

            work_dirs = [path for path in base_output_dir.iterdir() if path.is_dir()]
            self.assertEqual(1, len(work_dirs))
            self.assertTrue((work_dirs[0] / "subject_eyeflow.h5").exists())
            self.assertFalse((base_output_dir / "outputs.zip").exists())
            self.assertTrue(any(str(work_dirs[0]) in line for line in app.logs))
            self.assertEqual("Zip failed", showerror.call_args.args[0])

class InputHandlingTests(unittest.TestCase):
    def _make_bare_app(self):
        app = ProcessApp.__new__(ProcessApp)
        app.batch_holo_input_var = _Var("")
        app.batch_output_var = _Var("")
        app.batch_zip_var = _Var(False)
        app.batch_zip_name_var = _Var("outputs.zip")
        app._selected_holo_input_paths = []
        app._synchronizing_holo_input_var = False
        app._reset_progress = lambda: None
        app._set_minimal_status = lambda _text: None
        logs: list[str] = []
        app._log_batch = logs.append
        return app, logs

    def test_handle_dropped_paths_assigns_holo_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holo_path = tmp_path / "subject.holo"
            holo_path.write_text("dummy", encoding="utf-8")

            app, logs = self._make_bare_app()

            accepted = ProcessApp._handle_dropped_paths(app, [holo_path])

            self.assertTrue(accepted)
            self.assertEqual(str(holo_path), app.batch_holo_input_var.get())
            self.assertEqual(
                str(tmp_path / "subject" / "subject_EF"),
                app.batch_output_var.get(),
            )
            self.assertEqual("subject_eyeflow.zip", app.batch_zip_name_var.get())
            self.assertEqual(1, len(logs))
            self.assertIn("Drag and drop HOLO", logs[0])

    @mock.patch("eye_flow.messagebox.showwarning")
    def test_validate_selected_inputs_requires_holo_selection(
        self,
        showwarning,
    ) -> None:
        app = ProcessApp.__new__(ProcessApp)

        resolved = ProcessApp._validate_selected_inputs(app, [])

        self.assertIsNone(resolved)
        self.assertEqual("Missing input", showwarning.call_args.args[0])

    def test_minimal_output_filename_uses_current_inputs(self) -> None:
        app = ProcessApp.__new__(ProcessApp)
        app.ui_mode = "minimal"
        app.batch_zip_var = _Var(False)
        app._selected_holo_paths = lambda: [Path("subject.holo")]
        app._default_work_h5_name = lambda: "subject_eyeflow.h5"

        output_name = ProcessApp._minimal_output_filename_for_run(app)

        self.assertEqual("subject_eyeflow.h5", output_name)


class PipelineOrderTests(unittest.TestCase):
    def test_move_pipeline_to_index_updates_order_and_refreshes_library(self) -> None:
        app = ProcessApp.__new__(ProcessApp)
        app.pipeline_rows = [
            SimpleNamespace(name="A"),
            SimpleNamespace(name="B"),
            SimpleNamespace(name="C"),
        ]
        app._persist_pipeline_order = mock.Mock()
        app._populate_pipeline_library = mock.Mock()
        app._pipeline_index = lambda name: next(
            (idx for idx, row in enumerate(app.pipeline_rows) if row.name == name),
            None,
        )

        moved = ProcessApp._move_pipeline_to_index(app, "B", 0)

        self.assertTrue(moved)
        self.assertEqual(["B", "A", "C"], [row.name for row in app.pipeline_rows])
        app._persist_pipeline_order.assert_called_once_with()
        app._populate_pipeline_library.assert_called_once_with(app.pipeline_rows)

    def test_move_pipeline_to_index_supports_downward_insert_position(self) -> None:
        app = ProcessApp.__new__(ProcessApp)
        app.pipeline_rows = [
            SimpleNamespace(name="A"),
            SimpleNamespace(name="B"),
            SimpleNamespace(name="C"),
            SimpleNamespace(name="D"),
        ]
        app._persist_pipeline_order = mock.Mock()
        app._populate_pipeline_library = mock.Mock()

        moved = ProcessApp._move_pipeline_to_index(app, "B", 3)

        self.assertTrue(moved)
        self.assertEqual(["A", "C", "B", "D"], [row.name for row in app.pipeline_rows])
        app._persist_pipeline_order.assert_called_once_with()
        app._populate_pipeline_library.assert_called_once_with(app.pipeline_rows)

    def test_set_pipeline_visibility_moves_enabled_pipeline_to_top(self) -> None:
        app = ProcessApp.__new__(ProcessApp)
        pipeline = SimpleNamespace(name="B", available=True)
        app.pipeline_catalog = {"B": pipeline}
        app.pipeline_visibility = {"A": True, "B": False, "C": False}
        app._persist_pipeline_visibility = mock.Mock()
        app._move_pipeline_to_top = mock.Mock(return_value=True)
        app._update_pipeline_library_summary = mock.Mock()

        ProcessApp._set_pipeline_visibility(app, "B", True)

        self.assertTrue(app.pipeline_visibility["B"])
        app._persist_pipeline_visibility.assert_called_once_with()
        app._move_pipeline_to_top.assert_called_once_with("B")
        app._update_pipeline_library_summary.assert_not_called()

    def test_finish_pipeline_drag_reorders_to_drop_index(self) -> None:
        app = ProcessApp.__new__(ProcessApp)
        app._dragging_pipeline_name = "C"
        app._dragging_pipeline_active = True
        app._pipeline_drop_index = mock.Mock(return_value=0)
        app._move_pipeline_to_index = mock.Mock()
        app._hide_pipeline_drop_indicator = mock.Mock()
        widget = mock.Mock()

        result = ProcessApp._finish_pipeline_drag(
            app,
            SimpleNamespace(widget=widget, y_root=120),
        )

        widget.grab_release.assert_called_once_with()
        app._hide_pipeline_drop_indicator.assert_called_once_with()
        app._pipeline_drop_index.assert_called_once_with(120)
        app._move_pipeline_to_index.assert_called_once_with("C", 0)
        self.assertEqual("break", result)


class MouseWheelBindingTests(unittest.TestCase):
    def test_mousewheel_scroll_units_handles_delta_and_button_events(self) -> None:
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=120))
        )
        self.assertEqual(
            1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=-120))
        )
        self.assertEqual(
            -2, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=240))
        )
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=1))
        )
        self.assertEqual(
            -1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=0, num=4))
        )
        self.assertEqual(
            1, ProcessApp._mousewheel_scroll_units(SimpleNamespace(delta=0, num=5))
        )

    def test_bind_vertical_mousewheel_registers_handlers_that_scroll_canvas(self):
        app = ProcessApp.__new__(ProcessApp)
        widget = mock.Mock()
        canvas = mock.Mock()

        ProcessApp._bind_vertical_mousewheel(app, widget, canvas)

        self.assertEqual(
            ["<MouseWheel>", "<Button-4>", "<Button-5>"],
            [call.args[0] for call in widget.bind.call_args_list],
        )
        self.assertTrue(
            all(call.kwargs.get("add") == "+" for call in widget.bind.call_args_list)
        )

        mousewheel_handler = widget.bind.call_args_list[0].args[1]
        result = mousewheel_handler(SimpleNamespace(delta=-120))

        canvas.yview_scroll.assert_called_once_with(1, "units")
        self.assertEqual("break", result)


if __name__ == "__main__":
    unittest.main()
