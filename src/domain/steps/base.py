"""Shared metadata contract for pure domain steps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import ClassVar

import numpy as np


class DomainStep:
    name: ClassVar[str] = ""
    requires: ClassVar[frozenset[str]] = frozenset()
    produces: ClassVar[frozenset[str]] = frozenset()

    def run(
        self,
        inputs: Mapping[str, object],
        config: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError

    def fingerprint(
        self,
        inputs: Mapping[str, object],
        config: Mapping[str, object] | None = None,
    ) -> str:
        payload = {
            "config": dict(config or {}),
            "inputs": self._input_signature(inputs),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _input_signature(self, inputs: Mapping[str, object]) -> dict[str, object]:
        return {
            key: _hash_value(inputs[key])
            for key in sorted(self.requires)
            if key in inputs
        }


def _hash_value(value) -> object:
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return {"shape": value.shape, "dtype": str(value.dtype), "sha256": digest}
    if isinstance(value, Mapping):
        return {str(key): _hash_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hash_value(item) for item in value]
    return value
