"""Project-specific exception types."""


class EyeFlowError(Exception):
    """Base exception for the package."""


class InputDiscoveryError(EyeFlowError):
    """Raised when a provided input path cannot be resolved into H5 files."""


class H5ValidationError(EyeFlowError):
    """Raised when a fatal H5 validation problem occurs outside a report."""
