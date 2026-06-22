"""Tests for SDP field types (`voip.sdp.types`)."""

import pytest
from voip.sdp.types import RTPPayloadFormat, StaticPayloadType


class TestRTPPayloadFormatFromPt:
    def test_from_pt__mono_static_pt_applies_defaults(self):
        """A mono static payload type fills sample rate and encoding name."""
        fmt = RTPPayloadFormat.from_pt(0)
        assert fmt.payload_type == 0
        assert fmt.encoding_name == "PCMU"
        assert fmt.sample_rate == 8000
        assert fmt.channels == 1

    def test_from_pt__stereo_static_pt_applies_channel_count(self):
        """A multi-channel static payload type applies its channel count."""
        fmt = RTPPayloadFormat.from_pt(10)
        assert fmt.payload_type == 10
        assert fmt.encoding_name == "L16"
        assert fmt.sample_rate == 44100
        assert fmt.channels == 2

    def test_from_pt__dynamic_pt_keeps_mono_default(self):
        """A dynamic payload type (no static entry) defaults to one channel."""
        fmt = RTPPayloadFormat.from_pt(96)
        assert fmt.payload_type == 96
        assert fmt.encoding_name is None
        assert fmt.sample_rate is None
        assert fmt.channels == 1


class TestRTPPayloadFormatBytes:
    def test_bytes__mono_omits_channel_suffix(self):
        """Mono formats serialise without a trailing channel count."""
        fmt = RTPPayloadFormat(
            payload_type=0, encoding_name="PCMU", sample_rate=8000, channels=1
        )
        assert bytes(fmt) == b"0 PCMU/8000"

    def test_bytes__stereo_includes_channel_suffix(self):
        """Multi-channel formats serialise with a trailing channel count."""
        fmt = RTPPayloadFormat(
            payload_type=10, encoding_name="L16", sample_rate=44100, channels=2
        )
        assert bytes(fmt) == b"10 L16/44100/2"


class TestRTPPayloadFormatParse:
    def test_parse__stereo_rtpmap(self):
        """Parse a multi-channel rtpmap value."""
        fmt = RTPPayloadFormat.parse("10 L16/44100/2")
        assert fmt.payload_type == 10
        assert fmt.encoding_name == "L16"
        assert fmt.sample_rate == 44100
        assert fmt.channels == 2

    def test_parse__mono_rtpmap_defaults_to_one_channel(self):
        """Parse a mono rtpmap value without an explicit channel count."""
        fmt = RTPPayloadFormat.parse("0 PCMU/8000")
        assert fmt.payload_type == 0
        assert fmt.encoding_name == "PCMU"
        assert fmt.sample_rate == 8000
        assert fmt.channels == 1

    def test_parse__invalid_value_raises(self):
        """An rtpmap value without a clock rate raises ValueError."""
        with pytest.raises(ValueError):
            RTPPayloadFormat.parse("0 PCMU")


class TestRTPPayloadFormatFrameSize:
    def test_frame_size__static_pt_uses_spec_frame_size(self):
        """Static payload types report the frame size from the spec table."""
        fmt = RTPPayloadFormat.from_pt(0)
        assert fmt.frame_size == 160

    def test_frame_size__stereo_static_pt_uses_spec_frame_size(self):
        """Stereo static payload types report the spec frame size."""
        fmt = RTPPayloadFormat.from_pt(10)
        assert fmt.frame_size == 882

    def test_frame_size__dynamic_pt_derives_from_sample_rate(self):
        """Dynamic payload types derive the frame size from the sample rate."""
        fmt = RTPPayloadFormat(payload_type=96, encoding_name="opus", sample_rate=48000)
        assert fmt.frame_size == 48000 * 20 // 1000


class TestStaticPayloadTypeFromPt:
    def test_from_pt__known_pt_returns_member(self):
        """Lookup of a known static payload type returns the member."""
        assert StaticPayloadType.from_pt(0) is StaticPayloadType.PCMU

    def test_from_pt__unknown_pt_raises(self):
        """Lookup of an unknown static payload type raises ValueError."""
        with pytest.raises(ValueError):
            StaticPayloadType.from_pt(96)
