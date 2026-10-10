"""Chat model-role composition; importing this package constructs no models."""

from .roles import ModelRoles, resolve_model_roles

__all__ = ["ModelRoles", "resolve_model_roles"]
