"""Tests for the voip.codecs package (voip/codecs/__init__.py)."""

import importlib
import sys

import pytest

pytest.importorskip("numpy")

from voip.codecs import PCMA, PCMU, get  # noqa: E402


class TestGet:
    def test_get__pcma(self):
        """Get returns PCMA for the encoding name 'pcma'."""
        assert get("pcma") is PCMA

    def test_get__pcmu(self):
        """Get returns PCMU for the encoding name 'pcmu'."""
        assert get("pcmu") is PCMU

    def test_get__opus(self):
        """Get returns Opus for the encoding name 'opus'."""
        pytest.importorskip("av")
        from voip.codecs import Opus  # noqa: PLC0415

        assert get("opus") is Opus

    def test_get__g722(self):
        """Get returns G722 for the encoding name 'g722'."""
        pytest.importorskip("av")
        from voip.codecs import G722  # noqa: PLC0415

        assert get("g722") is G722

    def test_get__case_insensitive(self):
        """Get normalises the encoding name to lowercase before lookup."""
        assert get("PCMA") is PCMA
        assert get("PCMU") is PCMU

    def test_get__raise_not_implemented_error(self):
        """Get raises NotImplementedError for an unrecognised encoding name."""
        with pytest.raises(NotImplementedError, match="Unsupported codec"):
            get("unknown")


_PYAV_MODULE_KEYS: frozenset[str] = frozenset(
    {"av", "voip.codecs", "voip.codecs.av", "voip.codecs.g722", "voip.codecs.opus"}
)


class TestRegistry:
    def test_registry__always_contains_numpy_codecs(self):
        """REGISTRY always contains PCMA and PCMU regardless of PyAV availability."""
        import voip.codecs as m  # noqa: PLC0415

        assert "pcma" in m.REGISTRY
        assert "pcmu" in m.REGISTRY

    def test_registry__excludes_pyav_codecs_when_av_unavailable(self):
        """REGISTRY excludes G722 and Opus when av is not importable."""
        import voip.codecs as target  # noqa: PLC0415

        keys_to_remove = [k for k in sys.modules if k in _PYAV_MODULE_KEYS]
        saved = {k: sys.modules.pop(k) for k in keys_to_remove}
        sys.modules["av"] = None  # causes ImportError on `import av`

        try:
            import voip.codecs as fresh  # noqa: PLC0415

            assert "pcma" in fresh.REGISTRY
            assert "pcmu" in fresh.REGISTRY
            assert "g722" not in fresh.REGISTRY
            assert "opus" not in fresh.REGISTRY
        finally:
            for k in keys_to_remove:
                sys.modules.pop(k, None)
            sys.modules.update(saved)
            importlib.reload(target)

    def test_all__excludes_pyav_names_when_av_unavailable(self):
        """__all__ excludes G722, Opus, and PyAVCodec when av is not importable."""
        keys_to_remove = [k for k in sys.modules if k in _PYAV_MODULE_KEYS]
        saved = {k: sys.modules.pop(k) for k in keys_to_remove}
        sys.modules["av"] = None  # causes ImportError on `import av`

        try:
            import voip.codecs as fresh  # noqa: PLC0415

            for name in ("G722", "Opus", "PyAVCodec"):
                assert name not in fresh.__all__, (
                    f"{name!r} must not be in __all__ without av"
                )
        finally:
            for k in list(sys.modules):
                if k in _PYAV_MODULE_KEYS:
                    sys.modules.pop(k, None)
            sys.modules.update(saved)
