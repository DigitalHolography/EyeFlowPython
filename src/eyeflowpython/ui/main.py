from __future__ import annotations

from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import sv_ttk

from eyeflowpython.processing import default_output_root, process_input


class EyeFlowApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EyeFlowPython")
        self.root.minsize(760, 520)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select an input path.")

        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        title = ttk.Label(frame, text="EyeFlow H5 Processor")
        title.grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(frame, text="Input").grid(row=1, column=0, sticky="w", pady=(18, 8))
        input_entry = ttk.Entry(frame, textvariable=self.input_var)
        input_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(18, 8))
        self.file_button = ttk.Button(
            frame, text="File or Zip", command=self._choose_file
        )
        self.file_button.grid(row=1, column=2, padx=(0, 8), pady=(18, 8))
        self.folder_button = ttk.Button(
            frame, text="Folder", command=self._choose_folder
        )
        self.folder_button.grid(row=1, column=3, pady=(18, 8))

        ttk.Label(frame, text="Output Folder").grid(row=2, column=0, sticky="w", pady=8)
        output_entry = ttk.Entry(frame, textvariable=self.output_var)
        output_entry.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=8)
        self.output_button = ttk.Button(
            frame, text="Browse", command=self._choose_output
        )
        self.output_button.grid(row=2, column=2, pady=8, sticky="w")

        self.run_button = ttk.Button(frame, text="Run", command=self._run)
        self.run_button.grid(row=3, column=0, pady=(12, 12), sticky="w")
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=3, column=1, columnspan=3, sticky="w", pady=(12, 12)
        )

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=12)
        log_frame.pack(fill="both", expand=True, pady=(16, 0))
        self.log_widget = ScrolledText(log_frame, height=18, wrap="word")
        self.log_widget.pack(fill="both", expand=True)
        self.log_widget.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"{message}\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select H5 or Zip",
            filetypes=(
                ("H5 and Zip", "*.h5 *.zip"),
                ("H5 Files", "*.h5"),
                ("Zip Files", "*.zip"),
                ("All Files", "*.*"),
            ),
        )
        if selected:
            self._set_input(Path(selected))

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select Folder")
        if selected:
            self._set_input(Path(selected))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Select Output Folder")
        if selected:
            self.output_var.set(selected)

    def _set_input(self, input_path: Path) -> None:
        self.input_var.set(str(input_path))
        self.output_var.set(str(default_output_root(input_path)))
        self.status_var.set("Ready to run.")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.file_button.configure(state=state)
        self.folder_button.configure(state=state)
        self.output_button.configure(state=state)
        self.run_button.configure(state=state)

    def _run(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        input_text = self.input_var.get().strip()
        output_text = self.output_var.get().strip()

        if not input_text:
            messagebox.showerror(
                "Missing Input", "Select an .h5, a folder, or a .zip first."
            )
            return

        input_path = Path(input_text)
        output_root = (
            Path(output_text) if output_text else default_output_root(input_path)
        )

        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")

        self._append_log(f"Input: {input_path}")
        self._append_log(f"Output: {output_root}")
        self.status_var.set("Processing...")
        self._set_running(True)

        def worker() -> None:
            try:
                results = process_input(
                    input_path,
                    output_root=output_root,
                    logger=lambda msg: self._queue.put(("log", msg)),
                )
            except Exception:
                self._queue.put(("error", traceback.format_exc()))
            else:
                self._queue.put(("done", results))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self.root.after(100, self._poll_queue)

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
                continue

            self._set_running(False)
            if kind == "error":
                self.status_var.set("Processing failed.")
                self._append_log(str(payload))
                messagebox.showerror(
                    "Processing Failed", "Processing failed. See the log for details."
                )
            elif kind == "done":
                results = payload
                self.status_var.set(f"Done. Processed {len(results)} file(s).")
                self._append_log(f"Done. Processed {len(results)} file(s).")
                messagebox.showinfo(
                    "Processing Complete", f"Processed {len(results)} file(s)."
                )
            return

        if self._worker and self._worker.is_alive():
            self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    EyeFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
