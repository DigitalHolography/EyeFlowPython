"""Streamlit GUI focused on the EyeFlow metrics backlog."""

from __future__ import annotations

import streamlit as st

from eyeflow.gui.metrics_catalog import METRIC_GROUPS, iter_metrics


def main() -> None:
    st.set_page_config(
        page_title="EyeFlow Metrics",
        layout="wide",
    )

    st.title("EyeFlow metrics backlog")
    st.caption("Single-screen checklist of the MATLAB metrics that still need Python implementations.")

    total_metrics = len(iter_metrics())
    total_groups = len(METRIC_GROUPS)

    backlog_tab = st.tabs(["Metrics to implement"])[0]

    with backlog_tab:
        search = st.text_input(
            "Filter metrics",
            placeholder="Search by metric name, description, or MATLAB source file",
        ).strip().lower()

        selected_groups = st.multiselect(
            "Filter groups",
            options=[group.title for group in METRIC_GROUPS],
            default=[group.title for group in METRIC_GROUPS],
        )

        left_col, right_col = st.columns(2)
        left_col.metric("Tracked outputs", total_metrics)
        right_col.metric("Metric groups", total_groups)

        for group in METRIC_GROUPS:
            if group.title not in selected_groups:
                continue

            rows = [
                {
                    "Metric": item.name,
                    "Description": item.description,
                    "MATLAB source": item.source,
                }
                for item in group.items
                if not search
                or search in item.name.lower()
                or search in item.description.lower()
                or search in item.source.lower()
            ]

            if not rows:
                continue

            with st.expander(f"{group.title} ({len(rows)})", expanded=True):
                st.caption(group.description)
                st.dataframe(rows, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
