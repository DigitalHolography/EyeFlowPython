"""Launch the Streamlit GUI from the installed package."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - import-time environment guard
        raise SystemExit(
            "Streamlit is not installed. Install the GUI dependencies with "
            "`pip install -e .[gui]`."
        ) from exc

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    return stcli.main()
