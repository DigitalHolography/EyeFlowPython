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
from pipelines.core.utils import append_result_group, initialize_output_h5

_BaseAppTk = TkinterDnD.Tk if TkinterDnD is not None else tk.Tk


class _Tooltip:
    """Lightweight tooltip that shows on hover."""

    def __init__(
        self, widget: tk.Widget, text: str, bg: str = "#333333", fg: str = "#f7f7f7"
    ) -> None:
        self.widget = widget
        self.text = text
        self.bg = bg
        self.fg = fg
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
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
        self.batch_hd_input_var = tk.StringVar()
        self.batch_dv_input_var = tk.StringVar()
        self.batch_output_var = tk.StringVar(value=str(Path.cwd()))
        self.batch_zip_var = tk.BooleanVar(value=False)
        self.batch_zip_name_var = tk.StringVar(value="outputs.zip")
        self.batch_progress_var = tk.DoubleVar(value=0.0)
        self.minimal_status_var = tk.StringVar(value="Ready.")
        self.pipeline_library_summary_var = tk.StringVar(value="")
        self.minimal_hd_input_path_var = tk.StringVar(value="No HD input selected")
        self.minimal_dv_input_path_var = tk.StringVar(value="No DV input selected")
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
        self.batch_hd_input_var.trace_add("write", self._on_batch_paths_changed)
        self.batch_dv_input_var.trace_add("write", self._on_batch_paths_changed)
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

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)
        self.main_container = container

        self.minimal_view = ttk.Frame(container, padding=10)
        self.advanced_view = ttk.Frame(container, padding=10)

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
        parent.grid_anchor("center")

        content = ttk.Frame(parent, padding=(24, 24))
        content.grid(row=0, column=0)
        content.columnconfigure(0, minsize=420)
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

        self.minimal_hd_browse_button = ttk.Button(
            content,
            text="Select HD .h5",
            command=self.choose_hd_file,
        )
        self.minimal_hd_browse_button.grid(row=2, column=0, pady=(0, 10))
        self.minimal_hd_input_path_label = tk.Label(
            content,
            textvariable=self.minimal_hd_input_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_hd_input_path_label.grid(
            row=3,
            column=0,
            pady=(0, 18),
            sticky="ew",
        )

        self.minimal_dv_browse_button = ttk.Button(
            content,
            text="Select DV .h5",
            command=self.choose_dv_file,
        )
        self.minimal_dv_browse_button.grid(row=4, column=0, pady=(0, 10))
        self.minimal_dv_input_path_label = tk.Label(
            content,
            textvariable=self.minimal_dv_input_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_dv_input_path_label.grid(
            row=5,
            column=0,
            pady=(0, 18),
            sticky="ew",
        )

        self.minimal_output_button = ttk.Button(
            content,
            text="Select output folder",
            command=self.choose_batch_output,
        )
        self.minimal_output_button.grid(row=6, column=0, pady=(0, 10))
        self.minimal_output_path_label = tk.Label(
            content,
            textvariable=self.minimal_output_path_var,
            bg=self._bg_color,
            fg=self._muted_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_path_label.grid(row=7, column=0, pady=(0, 6), sticky="ew")
        self.minimal_output_name_label = tk.Label(
            content,
            textvariable=self.minimal_output_name_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_output_name_label.grid(
            row=8,
            column=0,
            pady=(0, 18),
            sticky="ew",
        )

        self.minimal_run_button = ttk.Button(
            content, text="Run", command=self.run_batch
        )
        self.minimal_run_button.grid(row=9, column=0, pady=(0, 18))

        self.minimal_progress = ttk.Progressbar(
            content,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.batch_progress_var,
            length=340,
            style=self._progress_primary_style,
        )
        self.minimal_progress.grid(row=10, column=0, sticky="ew")
        self.minimal_status_label = tk.Label(
            content,
            textvariable=self.minimal_status_var,
            bg=self._bg_color,
            fg=self._text_fg,
            justify="center",
            wraplength=420,
        )
        self.minimal_status_label.grid(row=11, column=0, pady=(8, 0), sticky="ew")

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
            "hd": (
                getattr(self, "minimal_hd_browse_button", None),
                getattr(self, "minimal_hd_input_path_label", None),
                getattr(self, "batch_hd_input_entry", None),
                getattr(self, "batch_hd_browse_button", None),
            ),
            "dv": (
                getattr(self, "minimal_dv_browse_button", None),
                getattr(self, "minimal_dv_input_path_label", None),
                getattr(self, "batch_dv_input_entry", None),
                getattr(self, "batch_dv_browse_button", None),
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

        ttk.Label(parent, text="HD input").grid(row=0, column=0, sticky="w")
        self.batch_hd_input_entry = ttk.Entry(
            parent, textvariable=self.batch_hd_input_var
        )
        self.batch_hd_input_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.batch_hd_browse_button = ttk.Button(
            parent, text="Browse", command=self.choose_hd_file
        )
        self.batch_hd_browse_button.grid(
            row=0,
            column=2,
            sticky="w",
        )

        ttk.Label(parent, text="DV input").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self.batch_dv_input_entry = ttk.Entry(
            parent, textvariable=self.batch_dv_input_var
        )
        self.batch_dv_input_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(0, 4),
            pady=(8, 0),
        )
        self.batch_dv_browse_button = ttk.Button(
            parent, text="Browse", command=self.choose_dv_file
        )
        self.batch_dv_browse_button.grid(
            row=1,
            column=2,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Label(parent, text="Output").grid(row=2, column=0, sticky="w", pady=(8, 0))
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
            height = min(640, max(520, screen_height - 240), screen_height)
            min_width = min(620, width)
            min_height = min(420, height)
        else:
            width = max(560, min(660, screen_width - 260))
            height = max(420, min(520, screen_height - 260))
            min_width = min(500, width)
            min_height = min(520, height)
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
                self.minimal_view.update_idletasks()
                requested_width = self.minimal_view.winfo_reqwidth() + 24
                requested_height = self.minimal_view.winfo_reqheight() + 24
                width = min(
                    max(requested_width, min_width),
                    min(target_width, screen_width),
                )
                height = min(
                    max(requested_height, min_height),
                    min(target_height, screen_height),
                )
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
        previous_mode = self.ui_mode
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
            force_target_size=(
                normalized_mode == "minimal"
                and (previous_mode == "advanced" or not persist)
            ),
        )
        if persist:
            self._persist_ui_mode()

    def _on_close(self) -> None:
        self._persist_ui_mode()
        self._persist_trim_h5source()
        self.destroy()

    def _on_batch_paths_changed(self, *_args) -> None:
        self._update_minimal_path_labels()

    @staticmethod
    def _input_slot_label(slot: str) -> str:
        return {
            "hd": "HD",
            "dv": "DV",
            "both": "HD + DV",
            "work": "Work",
        }.get(slot, slot.upper())

    def _batch_input_var_for(self, slot: str) -> tk.StringVar:
        if slot == "dv":
            return self.batch_dv_input_var
        return self.batch_hd_input_var

    def _path_from_var(self, raw_value: str) -> Path | None:
        value = raw_value.strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _selected_input_paths(self) -> tuple[Path | None, Path | None]:
        return (
            self._path_from_var(self.batch_hd_input_var.get() or ""),
            self._path_from_var(self.batch_dv_input_var.get() or ""),
        )

    def _primary_input_path(self) -> Path | None:
        hd_path, dv_path = self._selected_input_paths()
        return hd_path or dv_path

    def _classify_dropped_input(self, dropped_path: Path) -> str | None:
        normalized = dropped_path.stem.lower()
        if re.search(r"(doppler[_\-\s]*vision|(?:^|[_\-\s])dv(?:$|[_\-\s]))", normalized):
            return "dv"
        if re.search(r"(holodoppler|(?:^|[_\-\s])hd(?:$|[_\-\s]))", normalized):
            return "hd"
        return None

    def _assign_input_path(self, slot: str, input_path: Path) -> None:
        normalized_path = input_path.expanduser()
        if not normalized_path.is_absolute():
            normalized_path = Path.cwd() / normalized_path
        self._batch_input_var_for(slot).set(str(normalized_path))
        self._apply_input_defaults(normalized_path)

    def _handle_dropped_paths(
        self,
        dropped_paths: Sequence[Path],
        *,
        slot_hint: str | None = None,
    ) -> bool:
        valid_paths: list[Path] = []
        for dropped_path in dropped_paths:
            cleaned = str(dropped_path).strip().strip("{}")
            if not cleaned:
                continue
            candidate = Path(cleaned).expanduser()
            if (
                candidate.is_file()
                and candidate.suffix.lower() in {".h5", ".hdf5"}
            ):
                valid_paths.append(candidate)

        if not valid_paths:
            return False

        assigned: list[tuple[str, Path]] = []
        assigned_slots: set[str] = set()
        for dropped_path in valid_paths:
            slot = slot_hint or self._classify_dropped_input(dropped_path)
            if slot is None:
                for candidate_slot in ("hd", "dv"):
                    current_value = (
                        self._batch_input_var_for(candidate_slot).get() or ""
                    ).strip()
                    if not current_value and candidate_slot not in assigned_slots:
                        slot = candidate_slot
                        break
                if slot is None:
                    for candidate_slot in ("hd", "dv"):
                        if candidate_slot not in assigned_slots:
                            slot = candidate_slot
                            break
            if slot is None:
                continue
            self._assign_input_path(slot, dropped_path)
            assigned.append((slot, dropped_path))
            assigned_slots.add(slot)
            if slot_hint is not None:
                break

        if not assigned:
            return False

        for slot, dropped_path in assigned:
            self._log_batch(
                f"[INPUT] Drag and drop {self._input_slot_label(slot)} -> {dropped_path}"
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
            "Drop one or two .h5/.hdf5 files into the HD or DV input area.",
        )

    def _normalized_input_token(self, input_path: Path) -> str:
        token = re.sub(
            r"(?i)(holodoppler|doppler[_\-\s]*vision|(?:^|[_\-\s])hd(?:$|[_\-\s])|(?:^|[_\-\s])dv(?:$|[_\-\s]))",
            "_",
            input_path.stem,
        )
        token = re.sub(r"[_\-\s]+", "_", token).strip("_")
        return token or input_path.stem or "output"

    def _default_output_stem(self) -> str:
        hd_path, dv_path = self._selected_input_paths()
        tokens: list[str] = []
        for input_path in (hd_path, dv_path):
            if input_path is None:
                continue
            token = self._normalized_input_token(input_path)
            if token not in tokens:
                tokens.append(token)
        base_name = "_".join(tokens) if tokens else "output"
        return f"{base_name}_eyeflow"

    def _default_work_h5_name(self) -> str:
        return f"{self._default_output_stem()}.h5"

    def _default_archive_name(self) -> str:
        return f"{self._default_output_stem()}.zip"

    def _default_output_artifact_name(self) -> str:
        if self.batch_zip_var.get():
            return self._default_archive_name()
        return self._default_work_h5_name()

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
        hd_path, dv_path = self._selected_input_paths()
        self.minimal_hd_input_path_var.set(
            str(hd_path) if hd_path is not None else "No HD input selected"
        )
        self.minimal_dv_input_path_var.set(
            str(dv_path) if dv_path is not None else "No DV input selected"
        )
        if hd_path is None and dv_path is None:
            self.minimal_output_name_var.set("Output name: -")
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

    def _apply_input_defaults(self, input_path: Path) -> None:
        output_dir = input_path.parent if input_path.is_file() else input_path

        self.batch_output_var.set(str(output_dir))
        self.batch_zip_name_var.set(self._default_archive_name())
        self._reset_progress()
        self._set_minimal_status("Ready.")

    def _minimal_output_filename_for_run(self) -> str | None:
        if self.ui_mode != "minimal":
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
            "Select an input/output path, choose pipelines in Pipeline Library, "
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

    def choose_hd_file(self) -> None:
        initial_dir = self.batch_hd_input_var.get() or os.path.abspath("example_file")
        path = filedialog.askopenfilename(
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=initial_dir,
            title="Select holodoppler HDF5 file",
        )
        if path:
            self._assign_input_path("hd", Path(path))

    def choose_dv_file(self) -> None:
        initial_dir = self.batch_dv_input_var.get() or os.path.abspath("example_file")
        path = filedialog.askopenfilename(
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")],
            initialdir=initial_dir,
            title="Select doppler vision HDF5 file",
        )
        if path:
            self._assign_input_path("dv", Path(path))

    def choose_batch_output(self) -> None:
        path = filedialog.askdirectory(
            initialdir=self.batch_output_var.get() or None,
            title="Select base output folder",
        )
        if path:
            self.batch_output_var.set(path)

    def _validate_selected_inputs(
        self,
        holodoppler_h5: Path | None,
        doppler_vision_h5: Path | None,
        pipelines: Sequence[PipelineDescriptor],
    ) -> bool:
        del pipelines
        if holodoppler_h5 is None or doppler_vision_h5 is None:
            messagebox.showwarning(
                "Missing input",
                "Select both HD and DV HDF5 files.",
            )
            return False

        for slot, path in (("hd", holodoppler_h5), ("dv", doppler_vision_h5)):
            if not path.exists():
                messagebox.showerror(
                    f"Missing {self._input_slot_label(slot)} input",
                    f"{self._input_slot_label(slot)} input does not exist:\n{path}",
                )
                return False
            if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
                messagebox.showerror(
                    f"Invalid {self._input_slot_label(slot)} input",
                    f"{self._input_slot_label(slot)} input must be a .h5 or .hdf5 file:\n{path}",
                )
                return False

        return True

    def run_batch(self) -> None:
        self._reset_progress()
        holodoppler_h5, doppler_vision_h5 = self._selected_input_paths()

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

        if not self._validate_selected_inputs(
            holodoppler_h5,
            doppler_vision_h5,
            pipelines,
        ):
            return

        base_output_value = (self.batch_output_var.get() or "").strip()
        base_output_dir = (
            Path(base_output_value).expanduser() if base_output_value else Path.cwd()
        )
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        base_output_dir.mkdir(parents=True, exist_ok=True)

        self._reset_batch_output("Starting pipeline run...\n")
        if holodoppler_h5 is not None:
            self._log_batch(f"[INPUT] HD -> {holodoppler_h5}")
        if doppler_vision_h5 is not None:
            self._log_batch(f"[INPUT] DV -> {doppler_vision_h5}")

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
                    or self._default_work_h5_name()
                )
                output_h5_path = self._next_available_output_path(
                    base_output_dir / output_name
                )

            self._log_batch(f"[OUTPUT] {output_h5_path}")

            try:
                self._run_pipelines_to_output(
                    output_h5_path=output_h5_path,
                    pipelines=pipelines,
                    holodoppler_h5=holodoppler_h5,
                    doppler_vision_h5=doppler_vision_h5,
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
