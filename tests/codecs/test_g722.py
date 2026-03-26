"""Tests for the G.722 codec (voip.codecs.g722)."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.g722 import G722  # noqa: E402


class TestG722Constants:
    def test_payload_type(self):
        """G.722 payload type is 9 per RFC 3551."""
        assert G722.payload_type == 9

    def test_encoding_name(self):
        """G.722 encoding name is g722 (lowercase)."""
        assert G722.encoding_name == "g722"

    def test_sample_rate_hz(self):
        """G.722 actual audio sample rate is 16 000 Hz."""
        assert G722.sample_rate_hz == 16000

    def test_rtp_clock_rate_hz(self):
        """G.722 RTP clock rate is 8 000 Hz per RFC 3551 (despite 16 kHz audio)."""
        assert G722.rtp_clock_rate_hz == 8000

    def test_frame_size(self):
        """G.722 frame size is 320 audio samples per 20 ms."""
        assert G722.frame_size == 320

    def test_timestamp_increment(self):
        """G.722 timestamp increment is 160 ticks at the 8 kHz RTP clock."""
        assert G722.timestamp_increment == 160


class TestG722Decode:
    def test_decode__uses_g722_format(self):
        """Decode calls decode_pcm with the g722 PyAV format string."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000)
        args = mock_decode_pcm.call_args[0]
        assert args[1] == "g722"

    def test_decode__passes_rtp_clock_rate_as_input(self):
        """Decode passes rtp_clock_rate_hz (8 000) as the PyAV input rate hint."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000)
        kwargs = mock_decode_pcm.call_args[1]
        assert kwargs.get("input_rate_hz") == G722.rtp_clock_rate_hz

    def test_decode__ignores_input_rate_hz_argument(self):
        """Decode ignores the input_rate_hz argument and always uses rtp_clock_rate_hz."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000, input_rate_hz=16000)
        kwargs = mock_decode_pcm.call_args[1]
        assert kwargs.get("input_rate_hz") == G722.rtp_clock_rate_hz

    def test_decode__real_decode_returns_float32(self):
        """Decode produces a float32 array for real G.722 encoded input."""
        sample = G722.encode(np.zeros(320, dtype=np.float32))
        result = G722.decode(sample, 16000)
        assert result.dtype == np.float32


class TestG722Encode:
    def test_encode__returns_bytes(self):
        """Encode produces non-empty bytes for silent PCM input."""
        result = G722.encode(np.zeros(320, dtype=np.float32))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode__uses_g722_codec(self):
        """Encode delegates to encode_pcm with the g722 codec name."""
        with patch.object(G722, "encode_pcm", return_value=b"enc") as mock_enc:
            G722.encode(np.zeros(320, dtype=np.float32))
        mock_enc.assert_called_once_with(
            pytest.approx(np.zeros(320, dtype=np.float32)),
            "g722",
            G722.sample_rate_hz,
        )


class TestG722Packetize:
    def test_packetize__g722_encodes_whole_buffer_at_once(self):
        """Packetize calls encode on the full buffer to preserve ADPCM state."""
        audio = np.zeros(640, dtype=np.float32)
        fake_encoded = b"\xab" * 320
        with patch.object(G722, "encode", return_value=fake_encoded) as mock_enc:
            packets = list(G722.packetize(audio))
        mock_enc.assert_called_once_with(audio)
        assert len(packets) == 2
        assert packets[0] == b"\xab" * 160
        assert packets[1] == b"\xab" * 160

    def test_packetize__yields_160_byte_chunks(self):
        """Packetize yields 160-byte chunks (frame_size // 2 for G.722 2:1 ratio)."""
        audio = np.zeros(320, dtype=np.float32)
        payload_size = G722.frame_size // 2
        with patch.object(G722, "encode", return_value=b"\x00" * payload_size):
            packets = list(G722.packetize(audio))
        assert len(packets) == 1
        assert len(packets[0]) == payload_size


class TestG722CreateDecoder:
    def test_create_decoder__returns_g722_decoder(self):
        """create_decoder returns a G722Decoder instance."""
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        decoder = G722.create_decoder(16000)
        assert isinstance(decoder, G722Decoder)

    def test_create_decoder__output_rate_hz_set(self):
        """create_decoder stores the output_rate_hz on the returned decoder."""
        decoder = G722.create_decoder(16000)
        assert decoder.output_rate_hz == 16000

    def test_create_decoder__ignores_input_rate_hz(self):
        """create_decoder ignores input_rate_hz (G.722 always decodes at 16 kHz)."""
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        decoder = G722.create_decoder(16000, input_rate_hz=8000)
        assert isinstance(decoder, G722Decoder)
        assert decoder.output_rate_hz == 16000


class TestG722Decoder:
    def _make_encoded_packets(self, packet_count: int = 3) -> list[bytes]:
        """Encode *packet_count* 20 ms G.722 packets from a continuous sine wave."""
        import av  # noqa: PLC0415

        encoder = av.CodecContext.create("g722", "w")
        encoder.sample_rate = 16000
        encoder.format = av.AudioFormat("s16")
        encoder.layout = av.AudioLayout("mono")
        encoder.open()
        t = np.linspace(
            0, packet_count * 0.02, packet_count * G722.frame_size, endpoint=False
        )
        signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        pcm = np.clip(np.round(signal * 32768.0), -32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            pcm[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = 16000
        frame.pts = 0
        return [bytes(p) for p in encoder.encode(frame)]

    def test_decoder__initialises_codec_context(self):
        """G722Decoder creates a persistent PyAV codec context on init."""
        import av  # noqa: PLC0415
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        decoder = G722Decoder(16000)
        assert isinstance(decoder.codec_context, av.CodecContext)

    def test_decoder__initialises_resampler(self):
        """G722Decoder creates an AudioResampler targeting output_rate_hz."""
        import av  # noqa: PLC0415
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        decoder = G722Decoder(8000)
        assert isinstance(decoder.resampler, av.audio.resampler.AudioResampler)

    def test_decoder__decode_returns_float32(self):
        """Decode produces a float32 array for a real G.722 encoded packet."""
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        packet = self._make_encoded_packets(1)[0]
        decoder = G722Decoder(16000)
        result = decoder.decode(packet)
        assert result.dtype == np.float32

    def test_decoder__decode_yields_correct_sample_count(self):
        """Decode returns 320 float32 samples for a 160-byte G.722 packet at 16 kHz."""
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        packet = self._make_encoded_packets(1)[0]
        assert len(packet) == 160
        decoder = G722Decoder(16000)
        result = decoder.decode(packet)
        assert len(result) == G722.frame_size  # 320

    def test_decoder__preserves_adpcm_state_across_packets(self):
        """Stateful G722Decoder matches decoding all bytes together (reference).

        A per-packet stateless decoder resets the ADPCM predictor and produces
        near-silent output for packets 1+.  A stateful G722Decoder feeds each
        packet into the same persistent context and matches the reference (all
        packets decoded together in one container).
        """
        import io  # noqa: PLC0415

        import av  # noqa: PLC0415
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        packets = self._make_encoded_packets(3)

        # Reference: decode all bytes in one container (correct, stateful).
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=16000
        )
        ref_frames: list[np.ndarray] = []
        with av.open(
            io.BytesIO(b"".join(packets)),
            mode="r",
            format="g722",
            options={"sample_rate": "8000"},
        ) as container:
            for f in container.decode(audio=0):
                for rs in resampler.resample(f):
                    ref_frames.append(rs.to_ndarray().flatten())
        reference = np.concatenate(ref_frames)

        # Stateful G722Decoder: should match the reference exactly.
        stateful_decoder = G722Decoder(16000)
        stateful_parts = [stateful_decoder.decode(p) for p in packets]
        stateful = np.concatenate(stateful_parts)

        assert len(stateful) == len(reference)
        assert np.allclose(stateful, reference, atol=1e-5), (
            "Stateful G722Decoder output differs from reference: "
            "ADPCM state may not be preserved across packets"
        )

    def test_stateless_decode__diverges_after_first_packet(self):
        """Per-packet stateless decoding diverges from reference for packets 1+.

        This test documents the original bug: resetting ADPCM state each
        packet causes the decoded signal to be near-silent for all but the
        first packet, making the echo 'too short' and 'robotic'.
        """
        import io  # noqa: PLC0415

        import av  # noqa: PLC0415

        packets = self._make_encoded_packets(3)

        # Reference: decode all bytes together.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=16000
        )
        ref_frames: list[np.ndarray] = []
        with av.open(
            io.BytesIO(b"".join(packets)),
            mode="r",
            format="g722",
            options={"sample_rate": "8000"},
        ) as container:
            for f in container.decode(audio=0):
                for rs in resampler.resample(f):
                    ref_frames.append(rs.to_ndarray().flatten())
        reference = np.concatenate(ref_frames)

        # Stateless (original buggy behaviour): fresh context per packet.
        stateless_parts = [G722.decode(p, 16000) for p in packets]

        # Packet 0 is identical (both start from zero state).
        assert np.allclose(stateless_parts[0], reference[: G722.frame_size], atol=1e-5)
        # Packets 1+ diverge: stateless is near-silent, reference has full signal.
        for i, part in enumerate(stateless_parts[1:], start=1):
            ref_segment = reference[i * G722.frame_size : (i + 1) * G722.frame_size]
            mse = float(np.mean((part - ref_segment) ** 2))
            assert mse > 0.01, (  # near-silence vs full-amplitude signal
                f"Packet {i}: expected stateless decoder to diverge from reference "
                f"(ADPCM state reset), but MSE={mse:.6f} is too low"
            )
