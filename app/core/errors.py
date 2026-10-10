"""Explicit model failures with safe, actionable public messages."""


class ModelConfigurationError(ValueError):
    """Required model settings are missing or unsupported."""


class ProviderUnavailableError(RuntimeError):
    """A model dependency, initialization, or provider request failed."""
