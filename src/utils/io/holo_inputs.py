"""Compatibility exports for `.holo` input resolvers."""

from .resolvers import (
    HoloInputStatus,
    ResolvedHoloInput,
    holo_input_status,
    resolve_holo_input,
    resolve_selected_holo_inputs,
)

__all__ = [
    "HoloInputStatus",
    "ResolvedHoloInput",
    "holo_input_status",
    "resolve_holo_input",
    "resolve_selected_holo_inputs",
]
