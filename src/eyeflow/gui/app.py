"""Initial Streamlit GUI entrypoint for EyeFlow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="EyeFlowPython",
        layout="wide",
    )

    st.title("EyeFlowPython")
    st.caption("Downstream H5 analysis for precomputed EyeFlow inputs.")

    st.markdown(
        """
        This is the initial GUI scaffold.

        Current state:

        - the Python package is installed
        - the CLI validator is available
        - the Streamlit app can be launched from the package

        The analysis workflow itself will be added incrementally.
        """
    )

    st.subheader("Current Launch Options")
    st.code("eyeflow validate <input_path>", language="bash")
    st.code("eyeflow-gui", language="bash")

    st.subheader("Expected Input")
    st.markdown(
        """
        The application expects H5 inputs containing:

        - `/moment0`
        - `/moment1`
        - `/moment2`
        - `/masks/artery`
        - `/masks/vein`
        - metadata for timing and spatial calibration
        """
    )

    st.subheader("Project Files")
    readme_path = Path.cwd() / "README.md"
    st.write(f"Working directory: `{Path.cwd()}`")
    st.write(f"README detected: `{readme_path.exists()}`")


if __name__ == "__main__":
    main()
