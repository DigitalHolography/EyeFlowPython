from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import tkinter.font as tkfont
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import h5py

from app_settings import (
    LAST_BATCH_LOG_FILENAME,
    AppSettingsStore,
    normalize_pipeline_order,
    normalize_pipeline_visibility,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:  # optional dependency
    DND_FILES = None
    TkinterDnD = None

try:
    import sv_ttk
except ImportError:  #  optional dependency
    sv_ttk = None

from pipelines import PipelineDescriptor, load_pipeline_catalog
from pipelines.core.errors import format_pipeline_exception
from utils.io import append_result_group, initialize_output_h5

_BaseAppTk = TkinterDnD.Tk if TkinterDnD is not None else tk.Tk


class _Tooltip:
    """Lightweight tooltip that shows on hover."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str | Callable[[], str],
        bg: str = "#333333",
        fg: str = "#f7f7f7",
    ) -> None:
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _resolved_text(self) -> str:
        if callable(self.text):
            try:
                return str(self.text())
            except Exception:
                return ""
        return str(self.text)

    def _show(self, _event=None) -> None:
        text = self._resolved_text().strip()
        if self.tipwindow or not text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=text,
            justify="left",
            background=self.bg,
            foreground=self.fg,
            relief="solid",
            borderwidth=1,
            wraplength=360,
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


def _normalize_h5_lookup_path(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


class _MergedAttrs(Mapping[str, object]):
    def __init__(self, *sources: h5py.File | None) -> None:
        self._sources = [source for source in sources if source is not None]

    def __getitem__(self, key: str) -> object:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for source in self._sources:
            for key in source.attrs.keys():
                if key not in seen:
                    seen.add(key)
                    yield str(key)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def get(self, key: str, default=None):
        for source in self._sources:
            if key in source.attrs:
                return source.attrs[key]
        return default


class _EyeFlowView:
    def __init__(self, work_h5: h5py.File) -> None:
        self.work_h5 = work_h5

    def _pipeline_group_names(self) -> list[str]:
        eye_flow_group = self.work_h5.get("EyeFlow")
        if not isinstance(eye_flow_group, h5py.Group):
            return []
        return list(eye_flow_group.keys())

    def get(self, key: str, default=None):
        normalized_key = _normalize_h5_lookup_path(key)
        if not normalized_key:
            return default

        explicit = self.work_h5.get(f"EyeFlow/{normalized_key}")
        if explicit is not None:
            return explicit

        for pipeline_group_name in reversed(self._pipeline_group_names()):
            candidate = self.work_h5.get(
                f"EyeFlow/{pipeline_group_name}/{normalized_key}"
            )
            if candidate is not None:
                return candidate
        return default

    def __getitem__(self, key: str):
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.get(key) is not None


@dataclass(frozen=True)
class _ResolvedBatchInputs:
    holo_path: Path
    relative_holo_path: Path
    data_dir: Path
    hd_dir: Path
    dv_dir: Path
    hd_h5: Path
    dv_h5: Path


class _PipelineInputView:
    def __init__(
        self,
        *,
        work_h5: h5py.File,
        holodoppler_h5: h5py.File | None = None,
        doppler_vision_h5: h5py.File | None = None,
        preferred_input: str = "both",
    ) -> None:
        self.work_h5 = work_h5
        self.hd_h5 = holodoppler_h5
        self.dv_h5 = doppler_vision_h5
        self.work = work_h5
        self.hd = holodoppler_h5
        self.dv = doppler_vision_h5
        self.ef = _EyeFlowView(work_h5)
        self.preferred_input = preferred_input
        self.attrs = _MergedAttrs(
            self.work_h5,
            self._preferred_raw_source(),
            self._secondary_raw_source(),
        )

    def _preferred_raw_source(self) -> h5py.File | None:
        if self.preferred_input == "dv":
            return self.dv_h5 or self.hd_h5
        return self.hd_h5 or self.dv_h5

    def _secondary_raw_source(self) -> h5py.File | None:
        preferred = self._preferred_raw_source()
        if preferred is self.hd_h5:
            return self.dv_h5
        if preferred is self.dv_h5:
            return self.hd_h5
        return None

    @property
    def filename(self) -> str:
        primary = self._preferred_raw_source()
        if primary is not None and primary.filename is not None:
            return str(primary.filename)
        if self.work_h5.filename is not None:
            return str(self.work_h5.filename)
        return ""

    def _lookup_in_source(self, source: h5py.File | None, key: str):
        if source is None:
            return None

        direct = source.get(key)
        if direct is not None:
            return direct

        eye_flow_group = source.get("EyeFlow")
        if not isinstance(eye_flow_group, h5py.Group):
            return None

        for pipeline_group_name in reversed(list(eye_flow_group.keys())):
            candidate = source.get(f"EyeFlow/{pipeline_group_name}/{key}")
            if candidate is not None:
                return candidate
        return None

    def get(self, key: str, default=None):
        normalized_key = _normalize_h5_lookup_path(key)
        if not normalized_key:
            return default

        for source in (
            self.work_h5,
            self._preferred_raw_source(),
            self._secondary_raw_source(),
        ):
            found = self._lookup_in_source(source, normalized_key)
            if found is not None:
                return found
        return default

    def __getitem__(self, key: str):
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.get(key) is not None


class ProcessApp(_BaseAppTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EyeFlow")
        self.settings_store = AppSettingsStore()
        self._settings_warning_shown = False
        self._ensure_default_settings()
        self.ui_mode = self.settings_store.load_ui_mode()
        self.pipeline_registry: dict[str, PipelineDescriptor] = {}
        self.pipeline_catalog: dict[str, PipelineDescriptor] = {}
        self.pipeline_rows: list[PipelineDescriptor] = []
        self.pipeline_visibility: dict[str, bool] = {}
        self.pipeline_visibility_vars: dict[str, tk.BooleanVar] = {}
        self.pipeline_row_widgets: dict[str, tk.Widget] = {}
        self._dragging_pipeline_name: str | None = None
        self._dragging_pipeline_active = False
        self._drag_start_root_y = 0
        self._pipeline_drop_indicator: tk.Frame | None = None
        self.batch_holo_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self._selected_holo_input_paths: list[Path] = []
        self._synchronizing_holo_input_var = False
        self.minimal_status_var = tk.StringVar(value="Ready.")
        self.pipeline_library_summary_var = tk.StringVar(value="")
        self.minimal_holo_input_path_var = tk.StringVar(value="No input selected")
        self.holo_hd_status_var = tk.StringVar(value="HD waiting")
        self.holo_dv_status_var = tk.StringVar(value="DV waiting")
        self.minimal_output_path_var = tk.StringVar(value=str(Path.cwd()))
        self.minimal_output_name_var = tk.StringVar(value="Output name: -")
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self._last_saved_batch_log_path: Path | None = None
        self._progress_primary_style = "MinimalPrimary.Horizontal.TProgressbar"
        self._progress_final_style = "MinimalFinal.Horizontal.TProgressbar"
        self._window_icon_image: tk.PhotoImage | None = None
        self._minimal_logo_image: tk.PhotoImage | None = None
        self._minimal_title_font: tkfont.Font | None = None
        self._trim_h5source = tk.BooleanVar(
            value=self.settings_store.load_trim_h5source()
        )

        self._set_initial_window_size()
        self._apply_theme()
        self._set_window_icon()
        self._build_ui()
        self._install_drop_targets()
        self.batch_holo_input_var.trace_add("write", self._on_holo_input_changed)
        self.batch_output_var.trace_add("write", self._on_batch_paths_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._register_pipelines()
        self._reset_batch_output()
        self._update_minimal_path_labels()
        self._apply_ui_mode(self.ui_mode, persist=False)

    def _set_initial_window_size(self) -> None:
        width, height, min_width, min_height = self._window_size_for_mode(self.ui_mode)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(width, screen_width)
        height = min(height, screen_height)
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min_width, min_height)

    def _apply_theme(self) -> None:
        """
        Apply the Sun Valley ttk theme when available; otherwise fall back to a simple dark palette.
        """
        style = ttk.Style(self)
        self._style = style
        if sv_ttk:
            try:
                sv_ttk.set_theme("dark")
            except Exception:
                pass

        # Fallback palette aligned with Sun Valley dark.
        fallback_bg = "#0f1116"
        fallback_surface = "#1b1f27"
        fallback_fg = "#e8eef5"
        fallback_muted = "#9aa6b5"
        fallback_accent = "#4f9dff"

        # Derive colors from the active theme when possible to keep consistency.
        bg = style.lookup("TFrame", "background") or fallback_bg
        fg = style.lookup("TLabel", "foreground") or fallback_fg
        surface = (
            style.lookup("TEntry", "fieldbackground")
            or style.lookup("TEntry", "background")
            or fallback_surface
        )
        muted = (
            style.lookup("TLabel", "foreground", state=("disabled",)) or fallback_muted
        )
        accent = (
            style.lookup("TButton", "bordercolor")
            or style.lookup("TNotebook", "foreground")
            or fallback_accent
        )

        self.configure(bg=bg)
        # set texts colors when created.
        self._text_bg = surface
        self._text_fg = fg
        self._muted_fg = muted
        self._bg_color = bg
        self._surface_color = surface
        self._accent_color = accent
        self._success_color = "#3fb37f"
        self._error_color = "#ff6b6b"
        self._configure_progress_styles()

    def _configure_progress_styles(self) -> None:
        progress_colors = {
            self._progress_primary_style: self._accent_color,
            self._progress_final_style: "#3fb37f",
        }
        for style_name, color in progress_colors.items():
            try:
                self._style.configure(
                    style_name,
                    troughcolor=self._surface_color,
                    background=color,
                    bordercolor=color,
                    lightcolor=color,
                    darkcolor=color,
                )
            except tk.TclError:
                self._style.configure(style_name, background=color)

    def _build_ui(self) -> None:
        self._build_menu()

        container = ttk.Frame(self, padding=0)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = ttk.Frame(container, padding=0)
        self.advanced_view = ttk.Frame(container, padding=0)

        self._build_minimal_view(self.minimal_view)
        self._build_advanced_view(self.advanced_view)

    def _build_menu(self) -> None:
        self.ui_mode_var = tk.StringVar(value=self.ui_mode)
        menu_bar = tk.Menu(self)
        view_menu = tk.Menu(menu_bar, tearoff=False)
        view_menu.add_radiobutton(
            label="Minimal UI",
            value="minimal",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        view_menu.add_radiobutton(
            label="Advanced UI",
            value="advanced",
            variable=self.ui_mode_var,
            command=lambda: self._apply_ui_mode(self.ui_mode_var.get()),
        )
        menu_bar.add_cascade(label="View", menu=view_menu)
        self.configure(menu=menu_bar)

    def _build_minimal_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.grid_anchor("n")

        minimal_wraplength = 480

        content = ttk.Frame(parent, padding=(24, 24, 24, 24))
        content.grid(row=0, column=0, pady=(8, 12))
        content.columnconfigure(0, minsize=minimal_wraplength)
        self.minimal_content = content

        self.minimal_title_label = ttk.Label(
            content,
            text="EyeFlow",
            font=self._get_minimal_title_font(),
        )
        self.minimal_title_label.grid(row=0, column=0, pady=(0, 10))

        minimal_logo = self._load_scaled_logo_image(max_width=360, max_height=144)
        if minimal_logo is not None:
            self._minimal_logo_image = minimal_logo
            self.minimal_logo_label = ttk.Label(content, image=self._minimal_logo_image)
            self.minimal_logo_label.grid(row=1, column=0, pady=(0, 18))

        self.minimal_holo_browse_button = ttk.Button(
            content,
            text="Select .holo file(s)",
            command=self.choose_holo_file,
        )
        self.minimal_holo_browse_button.grid(row=2, column=0, pady=(0, 10))
        self.minimal_holo_input_path_label = tk.Label(
            content,
            textvariable=self.minimal_holo_input_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=minimal_wraplength,
        )
        self.minimal_holo_input_path_label.grid(
            row=3,
            column=0,
            pady=(0, 4),
            sticky="ew",
        )
        self.minimal_holo_status_frame = ttk.Frame(content)
        self.minimal_holo_status_frame.grid(row=4, column=0, pady=(0, 18))
        self.minimal_holo_hd_status_label = tk.Label(
            self.minimal_holo_status_frame,
            textvariable=self.holo_hd_status_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
        )
        self.minimal_holo_hd_status_label.pack(side="left")
        self.minimal_holo_status_separator_label = tk.Label(
            self.minimal_holo_status_frame,
            text=" | ",
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
        )
        self.minimal_holo_status_separator_label.pack(side="left")
        self.minimal_holo_dv_status_label = tk.Label(
            self.minimal_holo_status_frame,
            textvariable=self.holo_dv_status_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
        )
        self.minimal_holo_dv_status_label.pack(side="left")

        self.minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=self.choose_batch_output,
        )
        self.minimal_output_button.grid(row=5, column=0, pady=(0, 10))
        self.minimal_output_path_label = tk.Label(
            content,
            textvariable=self.minimal_output_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=minimal_wraplength,
        )
        self.minimal_output_path_label.grid(row=6, column=0, pady=(0, 6), sticky="ew")
        self.minimal_output_name_label = tk.Label(
            content,
            textvariable=self.minimal_output_name_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=minimal_wraplength,
        )
        self.minimal_output_name_label.grid(
            row=7,
            column=0,
            pady=(0, 18),
            sticky="ew",
        )

        self.minimal_run_button = ttk.Button(
            content, text="Run", command=self.run_batch
        )
        self.minimal_run_button.grid(row=8, column=0, pady=(0, 18))

        self.minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.batch_progress_var,
            length=340,
            style=self._progress_primary_style,
        )
        self.minimal_progress.grid(row=9, column=0, sticky="ew")
        self.minimal_status_label = tk.Label(
            content,
            textvariable=self.minimal_status_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=minimal_wraplength,
        )
        self.minimal_status_label.grid(row=10, column=0, pady=(8, 0), sticky="ew")

    def _get_minimal_title_font(self) -> tkfont.Font:
        if self._minimal_title_font is None:
            title_font = tkfont.nametofont("TkDefaultFont").copy()
            base_size = int(title_font.cget("size")) or 10
            title_font.configure(size=base_size * 2)
            self._minimal_title_font = title_font
        return self._minimal_title_font

    def _build_advanced_view(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.batch_tab = ttk.Frame(self.notebook, padding=10)
        self.pipeline_library_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.batch_tab, text="Run")
        self.notebook.add(self.pipeline_library_tab, text="Pipeline Library")

        self._build_batch_tab(self.batch_tab)
        self._build_pipeline_library_tab(self.pipeline_library_tab)

    def _install_drop_targets(self) -> None:
        if DND_FILES is None:
            return
        for widget in (
            self,
            self.minimal_view,
            self.advanced_view,
            self.batch_tab,
            self.pipeline_library_tab,
        ):
            self._register_drop_target(widget)

        slot_widgets = {
            "holo": (
                getattr(self, "minimal_holo_browse_button", None),
                getattr(self, "minimal_holo_input_path_label", None),
                getattr(self, "batch_holo_input_entry", None),
                getattr(self, "batch_holo_browse_button", None),
            ),
        }
        for slot, widgets in slot_widgets.items():
            for widget in widgets:
                if widget is not None:
                    self._register_drop_target(widget, slot)

    def _register_drop_target(
        self,
        widget: tk.Misc,
        slot_hint: str | None = None,
    ) -> None:
        if DND_FILES is None:
            return
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind(
                "<<Drop>>",
                lambda event, target_slot=slot_hint: self._on_input_drop(
                    event, slot_hint=target_slot
                ),
            )
        except (AttributeError, tk.TclError):
            pass

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)
        parent.rowconfigure(5, weight=1)

        input_label = ttk.Label(parent, text="Input (.holo)")
        input_label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        _Tooltip(
            input_label,
            self._reference_holo_tooltip_text,
            bg=self._surface_color,
            fg=self._text_fg,
        )
        self.batch_holo_input_entry = ttk.Entry(
            parent, textvariable=self.batch_holo_input_var
        )
        self.batch_holo_input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.batch_holo_browse_button = ttk.Button(
            parent, text="Select files", command=self.choose_holo_file
        )
        self.batch_holo_browse_button.grid(
            row=0,
            column=2,
            sticky="w",
        )
        self.batch_holo_status_frame = ttk.Frame(parent)
        self.batch_holo_status_frame.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(2, 0),
        )
        self.batch_holo_hd_status_label = tk.Label(
            self.batch_holo_status_frame,
            textvariable=self.holo_hd_status_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="left",
            anchor="w",
        )
        self.batch_holo_hd_status_label.pack(side="left")
        self.batch_holo_status_separator_label = tk.Label(
            self.batch_holo_status_frame,
            text=" | ",
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="left",
            anchor="w",
        )
        self.batch_holo_status_separator_label.pack(side="left")
        self.batch_holo_dv_status_label = tk.Label(
            self.batch_holo_status_frame,
            textvariable=self.holo_dv_status_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="left",
            anchor="w",
        )
        self.batch_holo_dv_status_label.pack(side="left")

        ttk.Label(parent, text="Output").grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=(8, 0),
        )
        batch_output_entry = ttk.Entry(parent, textvariable=self.batch_output_var)
        batch_output_entry.grid(row=2, column=1, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(parent, text="Browse", command=self.choose_batch_output).grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )

        controls = ttk.Frame(parent)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 4))

        run_btn = ttk.Button(controls, text="Run", command=self.run_batch)
        run_btn.grid(row=0, column=0, sticky="w")

        ttk.Label(parent, text="BatchLog").grid(
            row=4, column=0, sticky="nw", pady=(8, 2)
        )
        batch_output_frame = ttk.Frame(parent)
        batch_output_frame.grid(row=5, column=0, columnspan=3, sticky="nsew")
        batch_output_frame.columnconfigure(0, weight=1)
        batch_output_frame.rowconfigure(0, weight=1)
        self.batch_output = tk.Text(
            batch_output_frame,
            height=14,
            state="disabled",
            bg=self._text_bg,
            fg=self._text_fg,
            insertbackground=self._text_fg,
        )
        batch_output_scroll = ttk.Scrollbar(
            batch_output_frame, orient="vertical", command=self.batch_output.yview
        )
        self.batch_output.configure(yscrollcommand=batch_output_scroll.set)
        self.batch_output.grid(row=0, column=0, sticky="nsew")
        batch_output_scroll.grid(row=0, column=1, sticky="ns")

    def _resource_roots(self) -> list[Path]:
        roots: list[Path] = []
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            roots.append(Path(frozen_root))
        roots.append(Path(__file__).resolve().parents[1])
        roots.append(Path.cwd())
        return roots

    def _resolve_logo_path(self) -> Path | None:
        for root in self._resource_roots():
            candidate = root / "EyeFlow_logo.png"
            if candidate.is_file():
                return candidate
        return None

    def _load_logo_image(self) -> tk.PhotoImage | None:
        logo_path = self._resolve_logo_path()
        if logo_path is None:
            return None
        try:
            return tk.PhotoImage(file=str(logo_path))
        except tk.TclError:
            return None

    def _load_scaled_logo_image(
        self,
        *,
        max_width: int,
        max_height: int,
    ) -> tk.PhotoImage | None:
        image = self._load_logo_image()
        if image is None:
            return None

        scale_x = max(1, (image.width() + max_width - 1) // max_width)
        scale_y = max(1, (image.height() + max_height - 1) // max_height)
        scale = max(scale_x, scale_y)
        if scale > 1:
            image = image.subsample(scale, scale)
        return image

    def _set_window_icon(self) -> None:
        image = self._load_logo_image()
        if image is None:
            return
        self._window_icon_image = image
        try:
            self.iconphoto(True, self._window_icon_image)
        except tk.TclError:
            pass

    def _ensure_default_settings(self) -> None:
        try:
            self.settings_store.initialize_from_defaults()
        except OSError as exc:
            self._show_settings_warning(
                "Settings not initialized",
                f"Could not create default settings file:\n{exc}",
            )

    def _show_settings_warning(self, title: str, details: str) -> None:
        if self._settings_warning_shown:
            return
        self._settings_warning_shown = True
        messagebox.showwarning(title, details)

    def _persist_ui_mode(self) -> None:
        try:
            self.settings_store.save_ui_mode(self.ui_mode)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save UI mode preference:\n{exc}",
            )

    def _persist_trim_h5source(self) -> None:
        try:
            self.settings_store.save_trim_h5source(self._trim_h5source.get())
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save trim preference:\n{exc}",
            )

    def _window_size_for_mode(self, mode: str) -> tuple[int, int, int, int]:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        if mode == "advanced":
            width = min(900, max(760, screen_width - 240), screen_width)
            height = min(520, max(520, screen_height - 240), screen_height)
            min_width = min(620, width)
            min_height = min(420, height)
        else:
            width = max(560, min(660, screen_width - 260))
            height = max(460, min(500, screen_height - 260))
            min_width = min(500, width)
            min_height = min(460, height)
        return width, height, min_width, min_height

    def _ensure_window_size_for_mode(
        self,
        mode: str,
        *,
        force_target_size: bool = False,
    ) -> None:
        target_width, target_height, min_width, min_height = self._window_size_for_mode(
            mode
        )
        if mode == "minimal":
            self.minimal_content.update_idletasks()
            min_width = max(min_width, self.minimal_content.winfo_reqwidth())
            min_height = max(min_height, self.minimal_content.winfo_reqheight())
        self.minsize(min_width, min_height)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        current_width = max(self.winfo_width(), 1)
        current_height = max(self.winfo_height(), 1)
        if (
            not force_target_size
            and current_width >= min_width
            and current_height >= min_height
        ):
            return

        if force_target_size:
            if mode == "minimal":
                try:
                    if self.state() != "normal":
                        self.state("normal")
                except tk.TclError:
                    pass
                self.minimal_content.update_idletasks()
                width = min(min_width, screen_width)
                height = min(min_height, screen_height)
            else:
                width = min(target_width, screen_width)
                height = min(target_height, screen_height)
        else:
            width = min(max(current_width, min_width), screen_width)
            height = min(max(current_height, min_height), screen_height)
        x = max(min(self.winfo_x(), screen_width - width), 0)
        y = max(min(self.winfo_y(), screen_height - height), 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _apply_ui_mode(self, mode: str, *, persist: bool = True) -> None:
        normalized_mode = "advanced" if mode == "advanced" else "minimal"
        self.ui_mode = normalized_mode
        self.ui_mode_var.set(normalized_mode)

        self.minimal_view.pack_forget()
        self.advanced_view.pack_forget()
        if normalized_mode == "advanced":
            self.advanced_view.pack(fill="both", expand=True)
        else:
            self.minimal_view.pack(fill="both", expand=True)

        self.update_idletasks()

        self._ensure_window_size_for_mode(
            normalized_mode,
            force_target_size=(normalized_mode == "minimal"),
        )
        if persist:
            self._persist_ui_mode()

    def _on_close(self) -> None:
        self._persist_ui_mode()
        self._persist_trim_h5source()
        self.destroy()

    def _on_holo_input_changed(self, *_args) -> None:
        if not self._synchronizing_holo_input_var:
            self._selected_holo_input_paths = []
        self._update_minimal_path_labels()

    def _on_batch_paths_changed(self, *_args) -> None:
        self._update_minimal_path_labels()

    @staticmethod
    def _input_slot_label(slot: str) -> str:
        return {
            "holo": "HOLO",
            "hd": "HD",
            "dv": "DV",
            "both": "HD + DV",
            "work": "Work",
        }.get(slot, slot.upper())

    def _path_from_var(self, raw_value: str) -> Path | None:
        value = raw_value.strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _set_batch_holo_input_var(self, value: str) -> None:
        self._synchronizing_holo_input_var = True
        try:
            self.batch_holo_input_var.set(value)
        finally:
            self._synchronizing_holo_input_var = False

    def _selected_holo_paths(self) -> list[Path]:
        if self._selected_holo_input_paths:
            return list(self._selected_holo_input_paths)
        single_path = self._path_from_var(self.batch_holo_input_var.get() or "")
        return [single_path] if single_path is not None else []

    def _selected_holo_path(self) -> Path | None:
        selected_paths = self._selected_holo_paths()
        if len(selected_paths) != 1:
            return None
        return selected_paths[0]

    def _assign_holo_input_path(self, input_path: Path) -> None:
        self._assign_holo_input_paths([input_path])

    def _assign_holo_input_paths(self, input_paths: Sequence[Path]) -> None:
        normalized_paths: list[Path] = []
        seen_paths: set[str] = set()
        for input_path in input_paths:
            normalized_path = input_path.expanduser()
            if not normalized_path.is_absolute():
                normalized_path = Path.cwd() / normalized_path
            normalized_key = str(normalized_path).lower()
            if normalized_key in seen_paths:
                continue
            seen_paths.add(normalized_key)
            normalized_paths.append(normalized_path)

        self._selected_holo_input_paths = normalized_paths
        if not normalized_paths:
            self._set_batch_holo_input_var("")
            self._apply_input_defaults(None)
            return

        display_value = (
            str(normalized_paths[0])
            if len(normalized_paths) == 1
            else f"{len(normalized_paths)} .holo files selected"
        )
        self._set_batch_holo_input_var(display_value)
        self._apply_input_defaults(normalized_paths)

    @staticmethod
    def _list_h5_candidates(search_dir: Path) -> list[Path]:
        if not search_dir.is_dir():
            return []
        return sorted(
            (
                candidate
                for candidate in search_dir.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}
            ),
            key=lambda path: path.name.lower(),
        )

    def _resolve_h5_in_directory(
        self,
        search_dir: Path,
        *,
        slot: str,
        preferred_name: str,
    ) -> Path:
        candidates = self._list_h5_candidates(search_dir)
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a {self._input_slot_label(slot)} .h5/.hdf5 file in:\n"
                f"{search_dir}"
            )

        preferred_name_lower = preferred_name.lower()
        for candidate in candidates:
            if candidate.name.lower() == preferred_name_lower:
                return candidate

        if len(candidates) == 1:
            return candidates[0]

        candidate_list = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Found multiple {self._input_slot_label(slot)} .h5/.hdf5 files in:\n"
            f"{search_dir}\n\n"
            f"Expected a single file there or a file named:\n{preferred_name}\n\n"
            f"Candidates:\n{candidate_list}"
        )

    def _resolve_inputs_from_holo(
        self,
        holo_path: Path,
        *,
        require_holo_file: bool = True,
        relative_holo_path: Path | None = None,
    ) -> _ResolvedBatchInputs:
        expanded_holo = holo_path.expanduser()
        if not expanded_holo.is_absolute():
            expanded_holo = Path.cwd() / expanded_holo

        if expanded_holo.suffix.lower() != ".holo":
            raise ValueError(
                f"{self._input_slot_label('holo')} input must be a .holo file:\n"
                f"{expanded_holo}"
            )
        if require_holo_file:
            if not expanded_holo.exists():
                raise FileNotFoundError(
                    f"{self._input_slot_label('holo')} input does not exist:\n{expanded_holo}"
                )
            if not expanded_holo.is_file():
                raise ValueError(
                    f"{self._input_slot_label('holo')} input must be a .holo file:\n"
                    f"{expanded_holo}"
                )

        data_dir = expanded_holo.parent / expanded_holo.stem
        if not data_dir.is_dir():
            raise FileNotFoundError(
                "Could not find the data folder matching the selected .holo file:\n"
                f"{data_dir}"
            )

        hd_dir = data_dir / f"{expanded_holo.stem}_HD"
        dv_dir = data_dir / f"{expanded_holo.stem}_DV"
        hd_raw_dir = hd_dir / "raw"
        dv_h5_dir = dv_dir / "h5"
        hd_h5: Path | None = None
        dv_h5: Path | None = None
        missing_items: list[str] = []

        if not hd_dir.is_dir():
            missing_items.append(f"HD folder missing:\n{hd_dir}")
        elif not hd_raw_dir.is_dir():
            missing_items.append(f"HD raw folder missing:\n{hd_raw_dir}")
        else:
            try:
                hd_h5 = self._resolve_h5_in_directory(
                    hd_raw_dir,
                    slot="hd",
                    preferred_name=f"{hd_dir.name}_output.h5",
                )
            except FileNotFoundError as exc:
                missing_items.append(str(exc))

        if not dv_dir.is_dir():
            missing_items.append(f"DV folder missing:\n{dv_dir}")
        elif not dv_h5_dir.is_dir():
            missing_items.append(f"DV h5 folder missing:\n{dv_h5_dir}")
        else:
            try:
                dv_h5 = self._resolve_h5_in_directory(
                    dv_h5_dir,
                    slot="dv",
                    preferred_name=f"{dv_dir.name}.h5",
                )
            except FileNotFoundError as exc:
                missing_items.append(str(exc))

        if missing_items or hd_h5 is None or dv_h5 is None:
            raise FileNotFoundError(
                "Missing required input data for the selected .holo file:\n\n"
                + "\n\n".join(missing_items)
            )

        return _ResolvedBatchInputs(
            holo_path=expanded_holo,
            relative_holo_path=relative_holo_path or Path(expanded_holo.name),
            data_dir=data_dir,
            hd_dir=hd_dir,
            dv_dir=dv_dir,
            hd_h5=hd_h5,
            dv_h5=dv_h5,
        )

    def _batch_input_root(self, input_paths: Sequence[Path]) -> Path:
        if not input_paths:
            return Path.cwd()
        if len(input_paths) == 1:
            return input_paths[0].parent
        try:
            common_path = os.path.commonpath([str(path.parent) for path in input_paths])
        except ValueError:
            return Path.cwd()
        return Path(common_path)

    def _relative_holo_path_for_batch(self, holo_path: Path, batch_root: Path) -> Path:
        try:
            return holo_path.relative_to(batch_root)
        except ValueError:
            anchor = Path(holo_path.anchor)
            drive_token = holo_path.drive.rstrip(":\\/") or "root"
            tail = holo_path.relative_to(anchor) if anchor != holo_path else Path()
            return Path(drive_token) / tail

    def _resolve_selected_inputs(
        self,
        input_paths: Sequence[Path],
    ) -> list[_ResolvedBatchInputs]:
        normalized_inputs: list[Path] = []
        for input_path in input_paths:
            expanded_input = input_path.expanduser()
            if not expanded_input.is_absolute():
                expanded_input = Path.cwd() / expanded_input
            normalized_inputs.append(expanded_input)

        if not normalized_inputs:
            raise ValueError("Select one or more .holo files.")

        if len(normalized_inputs) == 1:
            return [self._resolve_inputs_from_holo(normalized_inputs[0])]

        batch_root = self._batch_input_root(normalized_inputs)
        resolved_inputs: list[_ResolvedBatchInputs] = []
        errors: list[str] = []

        for input_path in normalized_inputs:
            try:
                resolved_inputs.append(
                    self._resolve_inputs_from_holo(
                        input_path,
                        relative_holo_path=self._relative_holo_path_for_batch(
                            input_path,
                            batch_root,
                        ),
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{input_path}:\n{exc}")

        if errors:
            raise FileNotFoundError(
                "Missing required input data for one or more selected .holo files:\n\n"
                + "\n\n".join(errors)
            )

        return resolved_inputs

    def _handle_dropped_paths(
        self,
        dropped_paths: Sequence[Path],
        *,
        slot_hint: str | None = None,
    ) -> bool:
        del slot_hint
        valid_paths: list[Path] = []
        for dropped_path in dropped_paths:
            cleaned = str(dropped_path).strip().strip("{}")
            if not cleaned:
                continue
            candidate = Path(cleaned).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".holo":
                valid_paths.append(candidate)

        if not valid_paths:
            return False

        self._assign_holo_input_paths(valid_paths)
        if len(valid_paths) == 1:
            self._log_batch(f"[INPUT] Drag and drop HOLO -> {valid_paths[0]}")
        else:
            self._log_batch(
                f"[INPUT] Drag and drop HOLO batch -> {len(valid_paths)} files"
            )
        return True

    def _on_input_drop(self, event, *, slot_hint: str | None = None) -> None:
        raw_data = getattr(event, "data", "")
        try:
            dropped_values = self.tk.splitlist(raw_data)
        except tk.TclError:
            dropped_values = (raw_data,)

        dropped_paths = [Path(value) for value in dropped_values if value]
        if self._handle_dropped_paths(dropped_paths, slot_hint=slot_hint):
            return

        messagebox.showwarning(
            "Unsupported drop",
            "Drop one or more .holo files into the input area.",
        )

    def _normalized_input_token(self, input_path: Path) -> str:
        token = re.sub(r"[^A-Za-z0-9]+", "_", input_path.stem).strip("_")
        return token or input_path.stem or "output"

    def _default_output_stem(self) -> str:
        selected_paths = self._selected_holo_paths()
        if not selected_paths:
            base_name = "output"
        elif len(selected_paths) == 1:
            base_name = self._normalized_input_token(selected_paths[0])
        else:
            base_name = "batch"
        return f"{base_name}_eyeflow"

    def _default_work_h5_name_for_input(self, input_path: Path | None) -> str:
        base_name = (
            self._normalized_input_token(input_path)
            if input_path is not None
            else "output"
        )
        return f"{base_name}_eyeflow.h5"

    def _default_work_h5_name(self) -> str:
        return self._default_work_h5_name_for_input(self._selected_holo_path())

    def _default_archive_name(self) -> str:
        return f"{self._default_output_stem()}.zip"

    def _default_output_artifact_name(self) -> str:
        selected_inputs = self._selected_holo_paths()
        if len(selected_inputs) > 1:
            return "one *_eyeflow.h5 in each *_EF folder"
        if self.batch_zip_var.get():
            return self._default_archive_name()
        return self._default_work_h5_name()

    def _reference_holo_tooltip_text(self) -> str:
        return "Pick one or more reference .holo files."

    def _set_holo_status_parts(
        self,
        *,
        hd_text: str,
        hd_color: str,
        dv_text: str,
        dv_color: str,
    ) -> None:
        self.holo_hd_status_var.set(hd_text)
        self.holo_dv_status_var.set(dv_text)
        for label_name, color in (
            ("minimal_holo_hd_status_label", hd_color),
            ("minimal_holo_dv_status_label", dv_color),
            ("batch_holo_hd_status_label", hd_color),
            ("batch_holo_dv_status_label", dv_color),
        ):
            label = getattr(self, label_name, None)
            if label is not None:
                label.configure(fg=color)

    def _probe_holo_data_status(
        self,
        holo_path: Path,
        *,
        require_holo_file: bool,
    ) -> tuple[bool, bool]:
        normalized_holo = holo_path.expanduser()
        if not normalized_holo.is_absolute():
            normalized_holo = Path.cwd() / normalized_holo

        if (
            normalized_holo.suffix.lower() != ".holo"
            or (
                require_holo_file
                and (
                    not normalized_holo.exists()
                    or not normalized_holo.is_file()
                )
            )
        ):
            return False, False

        data_dir = normalized_holo.parent / normalized_holo.stem
        hd_dir = data_dir / f"{normalized_holo.stem}_HD"
        dv_dir = data_dir / f"{normalized_holo.stem}_DV"
        hd_raw_dir = hd_dir / "raw"
        dv_h5_dir = dv_dir / "h5"

        hd_found = False
        dv_found = False

        if hd_raw_dir.is_dir():
            try:
                self._resolve_h5_in_directory(
                    hd_raw_dir,
                    slot="hd",
                    preferred_name=f"{hd_dir.name}_output.h5",
                )
            except FileNotFoundError:
                pass
            else:
                hd_found = True

        if dv_h5_dir.is_dir():
            try:
                self._resolve_h5_in_directory(
                    dv_h5_dir,
                    slot="dv",
                    preferred_name=f"{dv_dir.name}.h5",
                )
            except FileNotFoundError:
                pass
            else:
                dv_found = True

        return hd_found, dv_found

    def _update_minimal_found_statuses(self, holo_paths: Sequence[Path]) -> None:
        if not holo_paths:
            self._set_holo_status_parts(
                hd_text="HD waiting",
                hd_color=self._muted_fg,
                dv_text="DV waiting",
                dv_color=self._muted_fg,
            )
            return

        if len(holo_paths) == 1:
            normalized_holo = holo_paths[0].expanduser()
            if not normalized_holo.is_absolute():
                normalized_holo = Path.cwd() / normalized_holo

            if (
                not normalized_holo.exists()
                or not normalized_holo.is_file()
                or normalized_holo.suffix.lower() != ".holo"
            ):
                self._set_holo_status_parts(
                    hd_text="HD unavailable",
                    hd_color=self._error_color,
                    dv_text="DV unavailable",
                    dv_color=self._error_color,
                )
                return

            hd_found, dv_found = self._probe_holo_data_status(
                normalized_holo,
                require_holo_file=True,
            )

            self._set_holo_status_parts(
                hd_text="HD found" if hd_found else "HD not found",
                hd_color=self._success_color if hd_found else self._error_color,
                dv_text="DV found" if dv_found else "DV not found",
                dv_color=self._success_color if dv_found else self._error_color,
            )
            return

        normalized_paths: list[Path] = []
        for holo_path in holo_paths:
            normalized_holo = holo_path.expanduser()
            if not normalized_holo.is_absolute():
                normalized_holo = Path.cwd() / normalized_holo
            if normalized_holo.suffix.lower() != ".holo":
                continue
            normalized_paths.append(normalized_holo)

        if not normalized_paths:
            self._set_holo_status_parts(
                hd_text="HD unavailable",
                hd_color=self._error_color,
                dv_text="DV unavailable",
                dv_color=self._error_color,
            )
            return

        total_entries = len(normalized_paths)
        hd_found_count = 0
        dv_found_count = 0
        for normalized_holo in normalized_paths:
            hd_found, dv_found = self._probe_holo_data_status(
                normalized_holo,
                require_holo_file=True,
            )
            hd_found_count += int(hd_found)
            dv_found_count += int(dv_found)

        self._set_holo_status_parts(
            hd_text=f"HD {hd_found_count}/{total_entries} found",
            hd_color=(
                self._success_color
                if hd_found_count == total_entries
                else self._error_color
            ),
            dv_text=f"DV {dv_found_count}/{total_entries} found",
            dv_color=(
                self._success_color
                if dv_found_count == total_entries
                else self._error_color
            ),
        )

    def _next_available_output_path(self, output_path: Path) -> Path:
        if not output_path.exists():
            return output_path
        suffix = output_path.suffix
        stem = output_path.stem
        parent = output_path.parent
        idx = 1
        candidate = parent / f"{stem}_{idx}{suffix}"
        while candidate.exists():
            idx += 1
            candidate = parent / f"{stem}_{idx}{suffix}"
        return candidate

    def _update_minimal_path_labels(self) -> None:
        holo_paths = self._selected_holo_paths()
        if not holo_paths:
            self.minimal_holo_input_path_var.set("No input selected")
        elif len(holo_paths) == 1:
            self.minimal_holo_input_path_var.set(str(holo_paths[0]))
        else:
            self.minimal_holo_input_path_var.set(
                f"{len(holo_paths)} .holo files selected"
            )
        self._update_minimal_found_statuses(holo_paths)
        if not holo_paths:
            self.minimal_output_name_var.set("Output name: -")
        elif len(holo_paths) > 1:
            self.minimal_output_name_var.set(
                "Output: one *_eyeflow.h5 in each *_EF folder"
            )
        else:
            self.minimal_output_name_var.set(
                f"Output name: {self._default_output_artifact_name()}"
            )

        output_value = (self.batch_output_var.get() or "").strip()
        self.minimal_output_path_var.set(output_value or "No output folder selected")

    def _set_minimal_status(self, text: str) -> None:
        self.minimal_status_var.set(text)
        self.update_idletasks()

    def _batch_log_path(self) -> Path:
        return self.settings_store.path.with_name(LAST_BATCH_LOG_FILENAME)

    def _persist_batch_log_snapshot(self) -> None:
        log_path = self._batch_log_path()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                self.batch_output.get("1.0", "end-1c"),
                encoding="utf-8",
            )
        except OSError:
            self._last_saved_batch_log_path = None
            return
        self._last_saved_batch_log_path = log_path

    def _set_progress_style(self, style_name: str) -> None:
        if hasattr(self, "minimal_progress"):
            self.minimal_progress.configure(style=style_name)

    def _reset_progress(self) -> None:
        self._progress_total_units = 1.0
        self._progress_completed_units = 0.0
        self._set_progress_style(self._progress_primary_style)
        self.batch_progress_var.set(0.0)
        self.update_idletasks()

    def _start_progress(
        self,
        total_units: float,
        *,
        style_name: str | None = None,
        status_text: str | None = None,
    ) -> None:
        self._progress_total_units = max(float(total_units), 1.0)
        self._progress_completed_units = 0.0
        self._set_progress_style(style_name or self._progress_primary_style)
        self.batch_progress_var.set(0.0)
        if status_text is not None:
            self._set_minimal_status(status_text)
        self.update_idletasks()

    def _set_progress_units(self, completed_units: float) -> None:
        clamped_units = min(
            max(float(completed_units), 0.0),
            max(self._progress_total_units, 1.0),
        )
        self._progress_completed_units = clamped_units
        self.batch_progress_var.set(
            (clamped_units / max(self._progress_total_units, 1.0)) * 100.0
        )
        self.update_idletasks()

    def _advance_progress(self, units: float = 1.0) -> None:
        self._set_progress_units(self._progress_completed_units + units)

    def _default_output_dir_for_input(self, input_path: Path) -> Path:
        output_dir = input_path.parent if input_path.is_file() else input_path
        if input_path.is_file() and input_path.suffix.lower() == ".holo":
            output_dir = input_path.parent / input_path.stem / f"{input_path.stem}_EF"
        return output_dir

    def _default_output_dir_for_inputs(self, input_paths: Sequence[Path]) -> Path:
        if not input_paths:
            return Path.cwd()
        if len(input_paths) == 1:
            return self._default_output_dir_for_input(input_paths[0])
        return self._batch_input_root(input_paths)

    def _batch_output_dir_for_resolved_input(
        self,
        base_output_dir: Path,
        resolved_input: _ResolvedBatchInputs,
    ) -> Path:
        stem = resolved_input.holo_path.stem
        return (
            base_output_dir
            / resolved_input.relative_holo_path.parent
            / stem
            / f"{stem}_EF"
        )

    def _replace_existing_output_dir_if_needed(
        self,
        base_output_dir: Path,
        *,
        holo_path: Path,
        force_replace: bool = False,
    ) -> bool:
        if not force_replace:
            expected_output_dir = self._default_output_dir_for_input(holo_path)
            if base_output_dir.resolve() != expected_output_dir.resolve():
                return False
        if not base_output_dir.exists():
            return False
        if base_output_dir.is_dir():
            shutil.rmtree(base_output_dir)
        else:
            base_output_dir.unlink()
        return True

    def _apply_input_defaults(self, input_path: Path | Sequence[Path] | None) -> None:
        if input_path is None:
            output_dir = Path.cwd()
        elif isinstance(input_path, Path):
            output_dir = self._default_output_dir_for_input(input_path)
        else:
            output_dir = self._default_output_dir_for_inputs(input_path)
        self.batch_output_var.set(str(output_dir))
        self.batch_zip_name_var.set(self._default_archive_name())
        self._reset_progress()
        self._set_minimal_status("Ready.")

    def _minimal_output_filename_for_run(self) -> str | None:
        if self.ui_mode != "minimal":
            return None
        if len(self._selected_holo_paths()) != 1:
            return None
        if self.batch_zip_var.get():
            return None
        return self._default_work_h5_name()

    def _build_pipeline_library_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text="Select the pipelines to run. "
            "This preference is saved between app launches.",
        ).grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        controls.columnconfigure(4, weight=1)
        ttk.Button(
            controls,
            text="Select all",
            command=self.select_all_pipelines,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Deselect all",
            command=self.deselect_all_pipelines,
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Reload pipelines",
            command=self.refresh_pipeline_catalog,
        ).grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Button(
            controls,
            text="Open folder",
            command=self.open_pipeline_folder,
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Label(controls, textvariable=self.pipeline_library_summary_var).grid(
            row=0, column=4, sticky="e"
        )

        library_container = ttk.Frame(parent)
        library_container.grid(row=2, column=0, sticky="nsew")
        library_container.columnconfigure(0, weight=1)
        library_container.rowconfigure(0, weight=1)

        self.pipeline_library_canvas = tk.Canvas(
            library_container, highlightthickness=0, bg=self._bg_color
        )
        self.pipeline_library_canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_container,
            orient="vertical",
            command=self.pipeline_library_canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        self.pipeline_library_canvas.configure(yscrollcommand=library_scroll.set)
        self.pipeline_library_inner = ttk.Frame(self.pipeline_library_canvas)
        self.pipeline_library_window = self.pipeline_library_canvas.create_window(
            (0, 0), window=self.pipeline_library_inner, anchor="nw"
        )
        self.pipeline_library_inner.bind(
            "<Configure>",
            lambda _evt: self.pipeline_library_canvas.configure(
                scrollregion=self.pipeline_library_canvas.bbox("all")
            ),
        )
        self.pipeline_library_canvas.bind(
            "<Configure>",
            lambda evt: self.pipeline_library_canvas.itemconfigure(
                self.pipeline_library_window, width=evt.width
            ),
        )
        self._bind_vertical_mousewheel(
            self.pipeline_library_canvas, self.pipeline_library_canvas
        )
        self._bind_vertical_mousewheel(
            self.pipeline_library_inner, self.pipeline_library_canvas
        )
        self._bind_vertical_mousewheel(library_scroll, self.pipeline_library_canvas)

    def _bind_vertical_mousewheel(self, widget: tk.Misc, canvas: tk.Canvas) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(
                sequence,
                lambda event, target_canvas=canvas: self._on_vertical_mousewheel(
                    event, target_canvas
                ),
                add="+",
            )

    @staticmethod
    def _mousewheel_scroll_units(event: tk.Event) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            steps = max(1, abs(delta) // 120) if abs(delta) >= 120 else 1
            return -steps if delta > 0 else steps

        button = getattr(event, "num", None)
        if button == 4:
            return -1
        if button == 5:
            return 1
        return 0

    def _on_vertical_mousewheel(self, event: tk.Event, canvas: tk.Canvas) -> str | None:
        scroll_units = self._mousewheel_scroll_units(event)
        if not scroll_units:
            return None
        canvas.yview_scroll(scroll_units, "units")
        return "break"

    def _register_pipelines(self) -> None:
        available, missing = load_pipeline_catalog()
        rows = sorted(
            [*available, *missing], key=lambda pipeline: pipeline.name.lower()
        )
        self.pipeline_registry = {p.name: p for p in available}
        self.pipeline_catalog = {p.name: p for p in rows}
        rows = self._sync_pipeline_order(rows)
        self.pipeline_rows = rows
        self._sync_pipeline_visibility(rows)
        self._populate_pipeline_library(rows)
        self._install_drop_targets()

    def _descriptor_tooltip_text(self, descriptor) -> str:
        parts: list[str] = []
        description = getattr(descriptor, "description", "")
        if description:
            parts.append(description)
        required_pipelines = getattr(descriptor, "required_pipelines", [])
        if required_pipelines:
            parts.append(f"Requires pipelines: {', '.join(required_pipelines)}")
        missing_pipelines = getattr(descriptor, "missing_pipelines", [])
        if missing_pipelines:
            parts.append(
                "Unavailable until these pipelines are available: "
                f"{', '.join(missing_pipelines)}"
            )
        missing_deps = getattr(descriptor, "missing_deps", []) or getattr(
            descriptor, "requires", []
        )
        if missing_deps:
            parts.append(f"Install: {', '.join(missing_deps)}")
        return "\n".join(parts)

    def _pipeline_status_text(self, pipeline: PipelineDescriptor) -> str:
        if pipeline.available:
            return "Available"
        if pipeline.missing_deps:
            return f"Missing deps: {', '.join(pipeline.missing_deps)}"
        return "Unavailable"

    def _populate_pipeline_library(self, rows: list[PipelineDescriptor]) -> None:
        for child in self.pipeline_library_inner.winfo_children():
            child.destroy()
        self.pipeline_visibility_vars = {}
        self.pipeline_row_widgets = {}
        self.pipeline_library_inner.columnconfigure(0, weight=1)

        selected_header = ttk.Label(self.pipeline_library_inner, text="Selected")
        selected_header.grid(row=0, column=0, sticky="w", pady=(0, 6))
        order_header = ttk.Label(self.pipeline_library_inner, text="Pipeline")
        order_header.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6))
        status_header = ttk.Label(self.pipeline_library_inner, text="Status")
        status_header.grid(row=0, column=2, sticky="w", padx=(12, 0), pady=(0, 6))
        self._bind_vertical_mousewheel(selected_header, self.pipeline_library_canvas)
        self._bind_vertical_mousewheel(order_header, self.pipeline_library_canvas)
        self._bind_vertical_mousewheel(status_header, self.pipeline_library_canvas)

        for idx, pipeline in enumerate(rows, start=1):
            is_available = getattr(pipeline, "available", True)
            var = tk.BooleanVar(
                value=self.pipeline_visibility.get(pipeline.name, False)
                and is_available
            )
            row_frame = ttk.Frame(self.pipeline_library_inner)
            row_frame.grid(
                row=idx,
                column=0,
                columnspan=3,
                sticky="ew",
                pady=(0, 6),
            )
            row_frame.columnconfigure(1, weight=1)

            check = ttk.Checkbutton(
                row_frame,
                text="",
                variable=var,
                state="normal" if is_available else "disabled",
                command=lambda name=pipeline.name, visible_var=var: (
                    self._set_pipeline_visibility(name, visible_var.get())
                ),
            )
            check.grid(row=0, column=0, sticky="w")
            name_label = ttk.Label(
                row_frame,
                text=pipeline.name,
                cursor="fleur" if is_available else "",
            )
            name_label.grid(row=0, column=1, sticky="w", padx=(12, 0))
            self._bind_vertical_mousewheel(check, self.pipeline_library_canvas)
            self._bind_vertical_mousewheel(name_label, self.pipeline_library_canvas)
            self._bind_vertical_mousewheel(row_frame, self.pipeline_library_canvas)
            self.pipeline_row_widgets[pipeline.name] = row_frame

            if is_available:
                for widget in (row_frame, name_label):
                    self._bind_pipeline_drag(widget, pipeline.name)

            status_text = self._pipeline_status_text(pipeline)
            status = ttk.Label(row_frame, text=status_text)
            status.grid(row=0, column=2, sticky="w", padx=(12, 0))
            self._bind_vertical_mousewheel(status, self.pipeline_library_canvas)
            if is_available:
                self._bind_pipeline_drag(status, pipeline.name)

            tip_text = self._descriptor_tooltip_text(pipeline)
            if tip_text:
                _Tooltip(check, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(name_label, tip_text, bg=self._surface_color, fg=self._text_fg)
                _Tooltip(status, tip_text, bg=self._surface_color, fg=self._text_fg)

            self.pipeline_visibility_vars[pipeline.name] = var

        if self._pipeline_drop_indicator is None or not self._pipeline_drop_indicator.winfo_exists():
            self._pipeline_drop_indicator = tk.Frame(
                self.pipeline_library_inner,
                bg=self._accent_color,
                height=2,
                highlightthickness=0,
                bd=0,
            )
        self._hide_pipeline_drop_indicator()
        self._update_pipeline_library_summary()

    def _bind_pipeline_drag(self, widget: tk.Widget, name: str) -> None:
        widget.bind(
            "<ButtonPress-1>",
            lambda event, pipeline_name=name: self._start_pipeline_drag(
                event,
                pipeline_name,
            ),
            add="+",
        )
        widget.bind(
            "<B1-Motion>",
            self._on_pipeline_drag_motion,
            add="+",
        )
        widget.bind(
            "<ButtonRelease-1>",
            self._finish_pipeline_drag,
            add="+",
        )

    def _sync_pipeline_order(
        self,
        rows: list[PipelineDescriptor],
    ) -> list[PipelineDescriptor]:
        ordered_names, changed = normalize_pipeline_order(
            (pipeline.name for pipeline in rows),
            self.settings_store.load_pipeline_order(),
        )
        rows_by_name = {pipeline.name: pipeline for pipeline in rows}
        ordered_rows = [
            rows_by_name[name] for name in ordered_names if name in rows_by_name
        ]
        if changed:
            self._persist_pipeline_order(ordered_rows)
        return ordered_rows

    def _persist_pipeline_order(
        self,
        rows: Sequence[PipelineDescriptor] | None = None,
    ) -> None:
        order = [pipeline.name for pipeline in (rows or self.pipeline_rows)]
        try:
            self.settings_store.save_pipeline_order(order)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save pipeline order preference:\n{exc}",
            )

    def _pipeline_index(self, name: str) -> int | None:
        return next(
            (idx for idx, pipeline in enumerate(self.pipeline_rows) if pipeline.name == name),
            None,
        )

    def _move_pipeline_to_index(
        self,
        name: str,
        target_index: int,
        *,
        persist: bool = True,
        refresh: bool = True,
    ) -> bool:
        current_index = self._pipeline_index(name)
        if current_index is None:
            return False

        rows = list(self.pipeline_rows)
        pipeline = rows.pop(current_index)
        target_index = max(0, min(int(target_index), len(rows)))
        if target_index > current_index:
            target_index -= 1
        rows.insert(target_index, pipeline)
        if rows == self.pipeline_rows:
            return False

        self.pipeline_rows = rows
        if persist:
            self._persist_pipeline_order()
        if refresh:
            self._populate_pipeline_library(self.pipeline_rows)
        return True

    def _move_pipeline_to_top(
        self,
        name: str,
        *,
        persist: bool = True,
        refresh: bool = True,
    ) -> bool:
        return self._move_pipeline_to_index(
            name,
            0,
            persist=persist,
            refresh=refresh,
        )

    def _pipeline_drop_index(self, root_y: int) -> int:
        if not self.pipeline_rows:
            return 0
        self.pipeline_library_inner.update_idletasks()
        for idx, pipeline in enumerate(self.pipeline_rows):
            widget = self.pipeline_row_widgets.get(pipeline.name)
            if widget is None or not widget.winfo_exists():
                continue
            midpoint = widget.winfo_rooty() + (widget.winfo_height() / 2.0)
            if root_y < midpoint:
                return idx
        return len(self.pipeline_rows)

    def _pipeline_drop_indicator_y(self, drop_index: int) -> int:
        if not self.pipeline_rows:
            return 0
        clamped_index = max(0, min(drop_index, len(self.pipeline_rows)))
        if clamped_index >= len(self.pipeline_rows):
            last_widget = self.pipeline_row_widgets.get(self.pipeline_rows[-1].name)
            if last_widget is None or not last_widget.winfo_exists():
                return 0
            return int(last_widget.winfo_y() + last_widget.winfo_height())

        widget = self.pipeline_row_widgets.get(self.pipeline_rows[clamped_index].name)
        if widget is None or not widget.winfo_exists():
            return 0
        return int(widget.winfo_y())

    def _show_pipeline_drop_indicator(self, drop_index: int) -> None:
        indicator = getattr(self, "_pipeline_drop_indicator", None)
        if indicator is None:
            return
        self.pipeline_library_inner.update_idletasks()
        indicator_y = self._pipeline_drop_indicator_y(drop_index)
        indicator_width = max(self.pipeline_library_inner.winfo_width(), 1)
        indicator.place(
            x=0,
            y=max(indicator_y - 1, 0),
            width=indicator_width,
            height=2,
        )
        indicator.lift()

    def _hide_pipeline_drop_indicator(self) -> None:
        indicator = getattr(self, "_pipeline_drop_indicator", None)
        if indicator is not None:
            indicator.place_forget()

    def _start_pipeline_drag(self, event, name: str) -> str:
        self._dragging_pipeline_name = name
        self._dragging_pipeline_active = False
        self._drag_start_root_y = int(event.y_root)
        try:
            event.widget.grab_set()
        except tk.TclError:
            pass
        return "break"

    def _on_pipeline_drag_motion(self, event) -> str:
        if getattr(self, "_dragging_pipeline_name", None) is None:
            return "break"
        if not getattr(self, "_dragging_pipeline_active", False):
            if abs(int(event.y_root) - self._drag_start_root_y) < 4:
                return "break"
            self._dragging_pipeline_active = True
        drop_index = self._pipeline_drop_index(int(event.y_root))
        self._show_pipeline_drop_indicator(drop_index)
        return "break"

    def _finish_pipeline_drag(self, event) -> str:
        name = getattr(self, "_dragging_pipeline_name", None)
        self._dragging_pipeline_name = None
        was_active = getattr(self, "_dragging_pipeline_active", False)
        self._dragging_pipeline_active = False
        try:
            event.widget.grab_release()
        except tk.TclError:
            pass
        self._hide_pipeline_drop_indicator()
        if not name:
            return "break"
        if not was_active:
            return "break"
        target_index = self._pipeline_drop_index(int(event.y_root))
        self._move_pipeline_to_index(name, target_index)
        return "break"

    def _sync_pipeline_visibility(self, rows: list[PipelineDescriptor]) -> None:
        visibility, changed = normalize_pipeline_visibility(
            (pipeline.name for pipeline in rows),
            self.settings_store.load_pipeline_visibility(),
        )
        for pipeline in rows:
            if not pipeline.available and visibility.get(pipeline.name, False):
                visibility[pipeline.name] = False
                changed = True
        self.pipeline_visibility = visibility
        if changed:
            self._persist_pipeline_visibility()

    def _persist_pipeline_visibility(self) -> None:
        try:
            self.settings_store.save_pipeline_visibility(self.pipeline_visibility)
        except OSError as exc:
            self._show_settings_warning(
                "Settings not saved",
                f"Could not save pipeline selection preferences:\n{exc}",
            )

    def _set_pipeline_visibility(self, name: str, visible: bool) -> None:
        pipeline = self.pipeline_catalog.get(name)
        if pipeline is not None and not pipeline.available:
            visible = False
        if self.pipeline_visibility.get(name) == visible:
            return
        self.pipeline_visibility[name] = visible
        self._persist_pipeline_visibility()
        if visible:
            if self._move_pipeline_to_top(name):
                return
        self._update_pipeline_library_summary()

    def _set_all_pipeline_visibility(self, visible: bool) -> None:
        changed = False
        target_values = {
            pipeline.name: visible and pipeline.available
            for pipeline in self.pipeline_rows
        }
        for name, target_value in target_values.items():
            if self.pipeline_visibility.get(name) != target_value:
                self.pipeline_visibility[name] = target_value
                changed = True
        if not changed:
            return
        for name, var in self.pipeline_visibility_vars.items():
            var.set(self.pipeline_visibility.get(name, False))
        self._persist_pipeline_visibility()
        self._update_pipeline_library_summary()

    def _update_pipeline_library_summary(self) -> None:
        selected_count = sum(
            1
            for pipeline in self.pipeline_rows
            if pipeline.available and self.pipeline_visibility.get(pipeline.name, False)
        )
        available_count = sum(
            1 for pipeline in self.pipeline_rows if pipeline.available
        )
        self.pipeline_library_summary_var.set(
            f"Selected: {selected_count}/{available_count}"
        )

    def _package_folder(self, package_name: str) -> Path | None:
        module = sys.modules.get(package_name)
        module_path = getattr(module, "__path__", None)
        if module_path:
            for path_value in module_path:
                folder = Path(path_value).resolve()
                if folder.is_dir():
                    return folder

        module_file = getattr(module, "__file__", None)
        if module_file:
            folder = Path(module_file).resolve().parent
            if folder.is_dir():
                return folder

        for root in self._resource_roots():
            folder = root / package_name
            if folder.is_dir():
                return folder
        return None

    def _open_folder(self, folder: Path | None, label: str) -> None:
        if folder is None or not folder.is_dir():
            messagebox.showerror(label, f"Could not find the {label.lower()}.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(label, f"Could not open folder:\n{folder}\n\n{exc}")

    def open_pipeline_folder(self) -> None:
        self._open_folder(self._package_folder("pipelines"), "Pipeline folder")

    def select_all_pipelines(self) -> None:
        self._set_all_pipeline_visibility(True)

    def deselect_all_pipelines(self) -> None:
        self._set_all_pipeline_visibility(False)

    def refresh_pipeline_catalog(self) -> None:
        self._register_pipelines()

    def _reset_batch_output(
        self,
        message: str = (
            "Select one or more .holo files and output path, "
            "then run."
        ),
    ) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.delete("1.0", "end")
        self.batch_output.insert("end", message)
        self.batch_output.configure(state="disabled")
        self._persist_batch_log_snapshot()

    def _log_batch(self, text: str) -> None:
        self.batch_output.configure(state="normal")
        self.batch_output.insert("end", f"{text}\n")
        self.batch_output.see("end")
        self.batch_output.configure(state="disabled")
        self._persist_batch_log_snapshot()
        self.batch_output.update_idletasks()
        self.update_idletasks()

    def _show_batch_error_dialog(self, message: str) -> None:
        self.bell()
        if self._last_saved_batch_log_path is not None:
            message = (
                f"{message}\n\nLatest log saved to:\n{self._last_saved_batch_log_path}"
            )
        messagebox.showwarning(
            "Batch completed with errors",
            message,
        )

    def choose_holo_file(self) -> None:
        selected_paths = self._selected_holo_paths()
        selected_holo = selected_paths[0] if selected_paths else None
        initial_dir = (
            str(selected_holo.parent)
            if selected_holo is not None
            else os.path.abspath("example_file")
        )
        paths = filedialog.askopenfilenames(
            filetypes=[("HOLO", "*.holo"), ("All files", "*.*")],
            initialdir=initial_dir,
            title="Select .holo file(s)",
        )
        if paths:
            self._assign_holo_input_paths([Path(path) for path in paths])

    def choose_batch_output(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.batch_output_var.set(path)

    def _validate_selected_inputs(
        self,
        holo_paths: Sequence[Path],
    ) -> list[_ResolvedBatchInputs] | None:
        if not holo_paths:
            messagebox.showwarning(
                "Missing input",
                "Select one or more .holo files.",
            )
            return None

        try:
            return self._resolve_selected_inputs(holo_paths)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return None
        except FileNotFoundError as exc:
            messagebox.showerror("Missing data", str(exc))
            return None

    def run_batch(self) -> None:
        self._reset_progress()
        holo_paths = self._selected_holo_paths()

        selected_names = [
            pipeline.name
            for pipeline in self.pipeline_rows
            if pipeline.available and self.pipeline_visibility.get(pipeline.name, False)
        ]
        if not selected_names:
            messagebox.showwarning(
                "No pipelines",
                "Select at least one pipeline in Pipeline Library.",
            )
            return

        pipelines: list[PipelineDescriptor] = []
        missing: list[str] = []
        for name in selected_names:
            pipeline = self.pipeline_registry.get(name)
            if pipeline is None:
                missing.append(name)
            else:
                pipelines.append(pipeline)
        if missing:
            messagebox.showerror(
                "Pipeline missing", f"Pipeline(s) not registered: {', '.join(missing)}"
            )
            return

        resolved_inputs = self._validate_selected_inputs(holo_paths)
        if resolved_inputs is None:
            return
        is_batch_input = len(resolved_inputs) > 1
        if is_batch_input and self.batch_zip_var.get():
            messagebox.showwarning(
                "Unsupported output",
                "ZIP output is not available when multiple .holo files are selected.",
            )
            return

        base_output_value = (self.batch_output_var.get() or "").strip()
        base_output_dir = (
            Path(base_output_value).expanduser() if base_output_value else Path.cwd()
        )
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir

        self._reset_batch_output("Starting pipeline run...\n")
        if is_batch_input:
            self._log_batch(f"[INPUT] HOLO batch -> {len(resolved_inputs)} files")
            self._log_batch(f"[OUTPUT ROOT] -> {base_output_dir}")
            base_output_dir.mkdir(parents=True, exist_ok=True)
            self._start_progress(
                len(pipelines) * len(resolved_inputs),
                style_name=self._progress_primary_style,
                status_text="Running pipelines...",
            )

            completed_outputs: list[Path] = []
            for index, resolved_input in enumerate(resolved_inputs, start=1):
                item_prefix = f"[ITEM {index}/{len(resolved_inputs)}]"
                item_output_dir = self._batch_output_dir_for_resolved_input(
                    base_output_dir,
                    resolved_input,
                )
                self._log_batch(f"{item_prefix} HOLO -> {resolved_input.holo_path}")
                self._log_batch(f"{item_prefix} DATA DIR -> {resolved_input.data_dir}")
                self._log_batch(f"{item_prefix} HD -> {resolved_input.hd_h5}")
                self._log_batch(f"{item_prefix} DV -> {resolved_input.dv_h5}")
                if self._replace_existing_output_dir_if_needed(
                    item_output_dir,
                    holo_path=resolved_input.holo_path,
                    force_replace=True,
                ):
                    self._log_batch(
                        f"{item_prefix} Replaced existing output directory: "
                        f"{item_output_dir}"
                    )
                item_output_dir.mkdir(parents=True, exist_ok=True)
                output_h5_path = (
                    item_output_dir
                    / self._default_work_h5_name_for_input(resolved_input.holo_path)
                )
                self._log_batch(f"{item_prefix} OUTPUT -> {output_h5_path}")
                try:
                    self._run_pipelines_to_output(
                        output_h5_path=output_h5_path,
                        pipelines=pipelines,
                        holodoppler_h5=resolved_input.hd_h5,
                        doppler_vision_h5=resolved_input.dv_h5,
                    )
                except Exception as exc:  # noqa: BLE001
                    failure_message = f"{resolved_input.holo_path.name}: {exc}"
                    self._log_batch(f"[FAIL] {failure_message}")
                    self._set_minimal_status("Run failed.")
                    messagebox.showerror("Run failed", failure_message)
                    return
                completed_outputs.append(output_h5_path)

            self._set_progress_units(self._progress_total_units)
            self._log_batch(
                f"Completed. Created {len(completed_outputs)} output file(s)."
            )
            self._set_minimal_status("Process ended.")
            return

        resolved_input = resolved_inputs[0]
        self._log_batch(f"[INPUT] HOLO -> {resolved_input.holo_path}")
        self._log_batch(f"[INPUT] DATA DIR -> {resolved_input.data_dir}")
        self._log_batch(f"[RESOLVED] HD -> {resolved_input.hd_h5}")
        self._log_batch(f"[RESOLVED] DV -> {resolved_input.dv_h5}")
        if self._replace_existing_output_dir_if_needed(
            base_output_dir,
            holo_path=resolved_input.holo_path,
        ):
            self._log_batch(
                f"[OUTPUT] Replaced existing output directory: {base_output_dir}"
            )
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._start_progress(
            len(pipelines),
            style_name=self._progress_primary_style,
            status_text="Running pipelines...",
        )

        work_output_dir: Path | None = None
        clean_work_output = False
        output_h5_path: Path | None = None
        try:
            if self.batch_zip_var.get():
                work_output_dir = Path(tempfile.mkdtemp(dir=base_output_dir))
                output_h5_path = work_output_dir / self._default_work_h5_name()
            else:
                output_name = (
                    self._minimal_output_filename_for_run()
                    or self._default_work_h5_name_for_input(resolved_input.holo_path)
                )
                output_h5_path = base_output_dir / output_name

            self._log_batch(f"[OUTPUT] {output_h5_path}")

            try:
                self._run_pipelines_to_output(
                    output_h5_path=output_h5_path,
                    pipelines=pipelines,
                    holodoppler_h5=resolved_input.hd_h5,
                    doppler_vision_h5=resolved_input.dv_h5,
                )
            except Exception as exc:  # noqa: BLE001
                failure_message = str(exc)
                self._log_batch(f"[FAIL] {failure_message}")
                if work_output_dir is not None:
                    self._log_batch(f"[FAIL] Partial output kept under: {work_output_dir}")
                self._set_minimal_status("Run failed.")
                messagebox.showerror("Run failed", failure_message)
                return

            summary_msg = f"Output file: {output_h5_path}"
            if self.batch_zip_var.get():
                try:
                    self._start_progress(
                        1,
                        style_name=self._progress_final_style,
                        status_text="Creating ZIP...",
                    )
                    self._set_minimal_status("Creating ZIP...")
                    self._log_batch("[ZIP] Preparing archive...")
                    last_progress_log = 0.0

                    def _zip_progress(done: int, total: int, _rel_path: Path) -> None:
                        nonlocal last_progress_log
                        self._set_progress_units(1.0 if total == 0 else done / total)
                        now = time.monotonic()
                        if done == total or (now - last_progress_log) >= 0.5:
                            pct = 100 if total == 0 else int((done * 100) / total)
                            self._log_batch(f"[ZIP] {done}/{total} files ({pct}%)")
                            last_progress_log = now
                            try:
                                self.update()
                            except tk.TclError:
                                pass

                    zip_name = self.batch_zip_name_var.get().strip() or "outputs.zip"
                    if not zip_name.lower().endswith(".zip"):
                        zip_name += ".zip"
                    zip_path = self._zip_output_dir(
                        work_output_dir,
                        target_path=base_output_dir / zip_name,
                        progress_callback=_zip_progress,
                    )
                    self._log_batch(f"[ZIP] Archive created: {zip_path}")
                    summary_msg = f"ZIP archive: {zip_path}"
                    clean_work_output = True
                except Exception as exc:  # noqa: BLE001
                    self._log_batch(f"[ZIP FAIL] {exc}")
                    self._set_progress_units(1.0)
                    messagebox.showerror(
                        "Zip failed", f"Could not create ZIP archive: {exc}"
                    )
                    summary_msg = f"Outputs stored under: {work_output_dir}"

            self._set_progress_units(self._progress_total_units)
            self._log_batch(f"Completed. {summary_msg}")
            self._set_minimal_status("Process ended.")
        finally:
            if clean_work_output and work_output_dir is not None:
                shutil.rmtree(work_output_dir, ignore_errors=True)

    def _run_pipelines_to_output(
        self,
        *,
        output_h5_path: Path,
        pipelines: Sequence[PipelineDescriptor],
        holodoppler_h5: Path | None,
        doppler_vision_h5: Path | None,
    ) -> Path:
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            work_h5 = stack.enter_context(h5py.File(output_h5_path, "w"))
            hd_h5 = (
                stack.enter_context(h5py.File(holodoppler_h5, "r"))
                if holodoppler_h5 is not None
                else None
            )
            dv_h5 = (
                stack.enter_context(h5py.File(doppler_vision_h5, "r"))
                if doppler_vision_h5 is not None
                else None
            )
            initialize_output_h5(
                work_h5,
                holodoppler_source_file=(
                    str(holodoppler_h5) if holodoppler_h5 is not None else None
                ),
                doppler_vision_source_file=(
                    str(doppler_vision_h5) if doppler_vision_h5 is not None else None
                ),
            )
            work_h5.attrs["trim_h5source"] = True
            work_h5.attrs["pipeline_order"] = [pipeline.name for pipeline in pipelines]

            for pipeline_desc in pipelines:
                pipeline = pipeline_desc.instantiate()
                pipeline_input = _PipelineInputView(
                    work_h5=work_h5,
                    holodoppler_h5=hd_h5,
                    doppler_vision_h5=dv_h5,
                )
                try:
                    result = pipeline.run(pipeline_input)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        format_pipeline_exception(exc, pipeline)
                    ) from exc
                append_result_group(work_h5, pipeline.name, result)
                result.output_h5_path = str(output_h5_path)
                self._log_batch(f"[OK] {pipeline.name}")
                self._advance_progress()
        return output_h5_path

    def _zip_output_dir(
        self,
        folder: Path,
        target_path: Path | None = None,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> Path:
        folder = folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Output folder does not exist: {folder}")
        if target_path is None:
            zip_name = f"{folder.name}_outputs.zip" if folder.name else "outputs.zip"
            zip_path = folder.parent / zip_name
        else:
            zip_path = target_path.expanduser().resolve()
        if zip_path.exists():
            zip_path.unlink()
        files = sorted(
            (file_path for file_path in folder.rglob("*") if file_path.is_file()),
            key=lambda path: str(path.relative_to(folder)),
        )
        total_files = len(files)
        if progress_callback is not None:
            progress_callback(0, total_files, Path("."))
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as zf:
            for idx, file_path in enumerate(files, start=1):
                rel_path = file_path.relative_to(folder)
                zf.write(file_path, rel_path)
                if progress_callback is not None:
                    progress_callback(idx, total_files, rel_path)
        return zip_path


def main():
    app = ProcessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
