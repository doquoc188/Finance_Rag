"""Registration errors independent of legacy application/provider failures."""


class InvalidProviderNameError(ValueError):
    """A registry name is blank after whitespace trimming."""


class DuplicateProviderError(ValueError):
    """The canonical name is already registered in this instance."""


class ProviderNotRegisteredError(LookupError):
    """No factory has been explicitly registered for this name."""
