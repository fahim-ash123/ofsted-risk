"""Shared pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def frame_of():
    """Provide a helper that turns row mappings into a text frame.

    Returns:
        A function taking row mappings and returning a DataFrame.
    """

    def build(*rows: dict) -> pd.DataFrame:
        """Turn row mappings into a DataFrame of nullable strings.

        The loaders return nullable strings, so the tests use the same type
        and exercise the same comparison behaviour as real data.

        Args:
            *rows: One mapping per row.

        Returns:
            A frame with one row per mapping.
        """
        return pd.DataFrame(list(rows)).astype("string")

    return build
