"""Domain exceptions."""


class SwingRankError(Exception):
    """Base exception for the package."""


class ConfigurationError(SwingRankError):
    """Raised when the methodology configuration is internally inconsistent."""


class DataValidationError(SwingRankError):
    """Raised when provider data violates the point-in-time data contract."""
