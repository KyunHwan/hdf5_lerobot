"""Custom hooks for the picknplace bimanual robot schema.

This hooks file demonstrates how to add schema-specific logic.
The filtering here mirrors what the YAML config already supports,
but serves as a template for more complex custom logic.
"""


class Hooks:
    """Picknplace schema hooks."""

    def filter_episode(self, h5f, mapping):
        """Skip episodes flagged with data collection errors."""
        if "/metadata/data_collection_error" in h5f:
            if bool(h5f["/metadata/data_collection_error"][()]):
                return False
        return True
