"""Explicit instance-owned chat factory registration; no discovery or construction."""

from .base import ChatProviderFactory
from .errors import (
    DuplicateProviderError,
    InvalidProviderNameError,
    ProviderNotRegisteredError,
)


def _canonical_name(name: str) -> str:
    canonical = name.strip().lower()
    if not canonical:
        raise InvalidProviderNameError("Provider name must not be blank")
    return canonical


class ChatProviderRegistry:
    """Own registrations per instance, keyed by whitespace-trimmed lowercase names.

    Register/get never call factories. The caller constructs and owns model instances
    by passing ChatModelSettings to the returned factory; settings are not rewritten.
    """

    def __init__(self) -> None:
        self._factories: dict[str, ChatProviderFactory] = {}

    def register(self, name: str, factory: ChatProviderFactory) -> None:
        """Register a factory, rejecting duplicates rather than overwriting."""
        canonical = _canonical_name(name)
        if canonical in self._factories:
            raise DuplicateProviderError(
                f"Chat provider {canonical!r} is already registered in this registry"
            )
        self._factories[canonical] = factory

    def get(self, name: str) -> ChatProviderFactory:
        """Resolve a factory without initializing a client or choosing a fallback."""
        canonical = _canonical_name(name)
        try:
            return self._factories[canonical]
        except KeyError:
            raise ProviderNotRegisteredError(
                f"Chat provider {canonical!r} is not registered; "
                "call register(name, factory) on this registry before lookup"
            ) from None
