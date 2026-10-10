"""Registry and construction errors independent of the legacy application.

Public messages exclude credential values. Chained upstream exceptions may contain
secrets; callers must not expose full exception chains to clients.
"""


class InvalidProviderNameError(ValueError):
    """A registry name is blank after whitespace trimming."""


class DuplicateProviderError(ValueError):
    """The canonical name is already registered in this instance."""


class ProviderNotRegisteredError(LookupError):
    """No factory has been explicitly registered for this name."""


class ProviderConfigurationError(ValueError):
    """Required provider configuration is missing or blank."""


class ProviderDependencyError(ImportError):
    """A provider integration or its dependency cannot be imported."""


class ProviderInitializationError(RuntimeError):
    """An integration failed to load or construct a configured model."""
