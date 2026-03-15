"""Tests for the voip.codecs package (voip/codecs/__init__.py)."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("av")

from voip.codecs import G722, PCMA, PCMU, Opus, get  # noqa: E402


class TestGet:
    def test_get__opus(self):
        """Get returns Opus for the encoding name 'opus'."""
        assert get("opus") is Opus

    def test_get__g722(self):
        """Get returns G722 for the encoding name 'g722'."""
        assert get("g722") is G722

    def test_get__pcma(self):
        """Get returns PCMA for the encoding name 'pcma'."""
        assert get("pcma") is PCMA

    def test_get__pcmu(self):
        """Get returns PCMU for the encoding name 'pcmu'."""
        assert get("pcmu") is PCMU

    def test_get__case_insensitive(self):
        """Get normalises the encoding name to lowercase before lookup."""
        assert get("OPUS") is Opus
        assert get("G722") is G722
        assert get("PCMA") is PCMA
        assert get("PCMU") is PCMU

    def test_get__raise_not_implemented_error(self):
        """Get raises NotImplementedError for an unrecognised encoding name."""
        with pytest.raises(NotImplementedError, match="Unsupported codec"):
            get("unknown")
