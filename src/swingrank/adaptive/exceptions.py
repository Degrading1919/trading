"""Adaptive-layer-specific failures."""

from swingrank.exceptions import SwingRankError


class ModelFitError(SwingRankError):
    """Raised when a probabilistic model cannot produce a valid fit."""
