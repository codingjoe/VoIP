"""Tests for audio call handler and codec utilities."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.audio import AudioCall, _build_ogg_opus  # noqa: E402
from voip.rtp import RealtimeTransportProtocol, RTPPayloadType  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402
from voip.sip.types import CallerID  # noqa: E402


def _make_media(fmt: str, rtpmap: str | None = None) -> MediaDescription:
    """Build a single-codec MediaDescription for use in tests."""
    if rtpmap:
        payload_format = RTPPayloadFormat.parse(rtpmap)
    else:
        payload_format = RTPPayloadFormat(payload_type=int(fmt))
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[payload_format]
    )


OPUS_MEDIA = _make_media("111", "111 opus/48000/2")
PCMA_MEDIA = _make_media("8", "8 PCMA/8000")
PCMU_MEDIA = _make_media("0")  # static PT, no rtpmap
G722_MEDIA = _make_media("9", "9 G722/8000")


class TestBuildOggOpus:
    def test_build_ogg_opus__starts_with_ogg_magic(self):
        """The resulting container starts with the Ogg capture pattern 'OggS'."""
        assert _build_ogg_opus(b"packet").startswith(b"OggS")

    def test_build_ogg_opus__contains_opus_head(self):
        """The resulting container includes the OpusHead identification header."""
        assert b"OpusHead" in _build_ogg_opus(b"packet")

    def test_build_ogg_opus__contains_opus_tags(self):
        """The resulting container includes the OpusTags comment header."""
        assert b"OpusTags" in _build_ogg_opus(b"packet")

    def test_build_ogg_opus__non_empty_for_single_packet(self):
        """Produce a non-empty Ogg container for a single Opus packet."""
        assert len(_build_ogg_opus(b"x" * 100)) > 100

    def test_build_ogg_opus__empty_payload(self):
        """Produce a valid Ogg container even when the payload is empty bytes."""
        result = _build_ogg_opus(b"")
        assert b"OggS" in result

    def test_build_ogg_opus__large_packet_uses_255_lacing(self):
        """Produce a valid Ogg container when a packet exceeds 254 bytes (lacing spans 255)."""
        result = _build_ogg_opus(b"x" * 256)
        assert b"OggS" in result
        assert len(result) > 256

    def test_build_ogg_opus__produces_three_pages(self):
        """Produce exactly three Ogg pages: BOS, tags, and data."""
        result = _build_ogg_opus(b"x" * 10)
        assert result.count(b"OggS") == 3


def make_audio_call(**kwargs) -> AudioCall:
    """Create an AudioCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
        "media": PCMA_MEDIA,
        "caller": CallerID(""),
    }
    defaults.update(kwargs)
    return AudioCall(**defaults)


class TestAudioCall:
    def test_caller__returns_caller_arg(self):
        """Return the caller string passed at construction."""
        call = make_audio_call(caller="sip:bob@biloxi.com")
        assert call.caller == "sip:bob@biloxi.com"

    def test_caller__defaults_to_empty_string(self):
        """Return an empty string when no caller is given."""
        assert make_audio_call().caller == ""

    def test_audio_received__noop_by_default(self):
        """audio_received is a no-op in the base AudioCall class."""
        make_audio_call().audio_received(audio=np.array([]), rms=0.0)  # must not raise

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored as dataclass fields."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = AudioCall(
            rtp=mock_rtp, sip=mock_sip, media=PCMA_MEDIA, caller=CallerID("")
        )
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_init__stores_media(self):
        """Media parameter is stored on the AudioCall instance."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.media is media

    def test_init__derives_sample_rate_from_media(self):
        """sample_rate is derived from the RTPPayloadFormat sample_rate."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=9, encoding_name="G722", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.sample_rate == 8000

    def test_init__default_sample_rate_without_media(self):
        """Default sample_rate is 8000 Hz for G.711 codecs."""
        assert make_audio_call().sample_rate == 8000

    def test_init__derives_payload_type_from_media(self):
        """payload_type is derived from the first fmt entry of the MediaDescription."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        call = make_audio_call(media=media)
        assert call.payload_type == 8

    def test_init__default_payload_type_without_media(self):
        """Default payload_type is 8 (PCMA) when using the default test media."""
        assert make_audio_call().payload_type == 8

    def test_init__logs_codec_info(self, caplog):
        """Log codec name, sample rate and payload type at INFO level on init."""
        import logging  # noqa: PLC0415

        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        with caplog.at_level(logging.INFO, logger="voip.audio"):
            make_audio_call(media=media)
        assert any("PCMA" in r.message and "8000" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_packet_received__dispatches_audio_for_non_empty_payload(self):
        """packet_received schedules audio decoding when the packet has a payload."""
        from voip.rtp import RTPPacket  # noqa: PLC0415

        received: list = []

        class ConcreteCall(AudioCall):
            def _decode_raw(self, packet: bytes) -> np.ndarray:
                return np.array([1.0], dtype=np.float32)

            def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
                received.append(audio)

        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"audio"
        )
        call = ConcreteCall(
            rtp=MagicMock(), sip=MagicMock(), media=PCMA_MEDIA, caller=CallerID("")
        )
        call.packet_received(packet, ("127.0.0.1", 5004))
        await asyncio.sleep(0.05)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_packet_received__ignores_empty_payload(self):
        """packet_received does not schedule decoding when the payload is empty."""
        from voip.rtp import RTPPacket  # noqa: PLC0415

        received: list = []

        class ConcreteCall(AudioCall):
            def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
                received.append(audio)

        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b""
        )
        call = ConcreteCall(
            rtp=MagicMock(), sip=MagicMock(), media=PCMA_MEDIA, caller=CallerID("")
        )
        call.packet_received(packet, ("127.0.0.1", 5004))
        await asyncio.sleep(0.05)
        assert len(received) == 0


class TestNegotiateCodec:
    def _make_media(self, fmts: list[str], rtpmaps: list[str] | None = None):
        """Build a MediaDescription with given format list and optional rtpmap attributes."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        rtpmap_by_pt: dict[int, RTPPayloadFormat] = {}
        for rtpmap in rtpmaps or []:
            f = RTPPayloadFormat.parse(rtpmap)
            rtpmap_by_pt[f.payload_type] = f
        formats = [
            rtpmap_by_pt.get(int(pt)) or RTPPayloadFormat(payload_type=int(pt))
            for pt in fmts
        ]
        return MediaDescription(media="audio", port=49170, proto="RTP/AVP", fmt=formats)

    def test_negotiate_codec__prefers_opus(self):
        """Select Opus when offered alongside lower-priority codecs."""
        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2", "8 PCMA/8000"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 111
        assert result.fmt[0].sample_rate == 48000

    def test_negotiate_codec__falls_back_to_pcma(self):
        """Select PCMA when Opus and G.722 are not offered."""
        media = self._make_media(["0", "8"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8
        assert result.fmt[0].sample_rate == 8000

    def test_negotiate_codec__falls_back_to_pcmu(self):
        """Select PCMU when only PCMU is offered."""
        media = self._make_media(["0"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 0

    def test_negotiate_codec__matches_by_encoding_name_when_payload_type_differs(self):
        """Select a codec by encoding name when its dynamic payload type differs from preferred."""
        # Dynamic PT 99 is not in preferred PTs, but encoding name "opus" matches.
        media = self._make_media(["99"], ["99 opus/48000/2"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].encoding_name.lower() == "opus"

    def test_negotiate_codec__empty_fmt__raises(self):
        """Raise NotImplementedError when the remote side offers no audio formats."""
        media = self._make_media([])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__unknown_codec__raises(self):
        """Raise NotImplementedError when no offered codec matches PREFERRED_CODECS."""
        media = self._make_media(["126"], ["126 telephone-event/8000"])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__returns_media_description(self):
        """negotiate_codec returns a MediaDescription object."""
        from voip.sdp.types import MediaDescription  # noqa: PLC0415

        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2"])
        result = AudioCall.negotiate_codec(media)
        assert isinstance(result, MediaDescription)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""
        from voip.sdp.types import RTPPayloadFormat  # noqa: PLC0415

        class PCMAOnlyCall(AudioCall):
            PREFERRED_CODECS = [
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8

    def test_preferred_codecs__class_attribute(self):
        """PREFERRED_CODECS is a class attribute on AudioCall with Opus first."""
        from voip.sdp.types import RTPPayloadFormat  # noqa: PLC0415

        codecs = AudioCall.PREFERRED_CODECS
        assert isinstance(codecs, list)
        assert all(isinstance(c, RTPPayloadFormat) for c in codecs)
        pts = [c.payload_type for c in codecs]
        assert pts[0] == 111  # Opus is highest priority
        assert 8 in pts  # PCMA present
        assert 0 in pts  # PCMU present


class TestOutboundRTPInit:
    """Tests for outbound RTP state initialised in AudioCall.__post_init__."""

    def test_opus_media__sets_rtp_fields(self):
        """Opus media sets 48 kHz sample rate, 960-sample chunks, and 960 ts-increment."""
        call = make_audio_call(media=OPUS_MEDIA)
        assert call._rtp_sample_rate == 48000
        assert call._rtp_chunk_samples == 960
        assert call._rtp_ts_increment == 960

    def test_g722_media__sets_rtp_fields(self):
        """G.722 media sets 16 kHz audio but uses 8 kHz RTP clock (160 ts-increment)."""
        call = make_audio_call(media=G722_MEDIA)
        assert call._rtp_sample_rate == 16000
        assert call._rtp_chunk_samples == 320
        assert call._rtp_ts_increment == 160

    def test_pcmu_media__sets_rtp_fields(self):
        """PCMU media sets 8 kHz sample rate, 160-sample chunks and ts-increment."""
        call = make_audio_call(media=PCMU_MEDIA)
        assert call._rtp_sample_rate == 8000
        assert call._rtp_chunk_samples == 160
        assert call._rtp_ts_increment == 160


class TestResample:
    """Tests for AudioCall._resample."""

    def test_resample__downsamples_from_24khz_to_8khz(self):
        """_resample reduces 24 000 samples at 24 kHz to 8 000 samples at 8 kHz."""
        audio = np.zeros(24000, dtype=np.float32)
        assert len(AudioCall._resample(audio, 24000, 8000)) == 8000

    def test_resample__passthrough_when_rate_matches(self):
        """_resample returns the original array unchanged when rates are equal."""
        audio = np.zeros(8000, dtype=np.float32)
        assert AudioCall._resample(audio, 8000, 8000) is audio


class TestEncodePCMU:
    """Tests for AudioCall._encode_pcmu."""

    def test_encode_pcmu__returns_one_byte_per_sample(self):
        """_encode_pcmu returns a bytes object with one byte per input sample."""
        assert len(AudioCall._encode_pcmu(np.zeros(160, dtype=np.float32))) == 160

    def test_encode_pcmu__silence_encodes_to_midpoint(self):
        """Zero-amplitude samples encode to the µ-law midpoint value."""
        result = AudioCall._encode_pcmu(np.zeros(10, dtype=np.float32))
        assert result[0] in (127, 128)

    def test_encode_pcmu__silence_is_0x7f(self):
        """Silence (0.0) must encode to 0x7F (127) per ITU-T G.711."""
        assert AudioCall._encode_pcmu(np.zeros(1, dtype=np.float32))[0] == 0x7F

    def test_encode_pcmu__max_positive_is_0x00(self):
        """Maximum positive amplitude must encode to 0x00 per ITU-T G.711."""
        assert AudioCall._encode_pcmu(np.array([1.0], dtype=np.float32))[0] == 0x00

    def test_encode_pcmu__max_negative_is_0x80(self):
        """Maximum negative amplitude must encode to 0x80 per ITU-T G.711."""
        assert AudioCall._encode_pcmu(np.array([-1.0], dtype=np.float32))[0] == 0x80


class TestEncodePCMA:
    """Tests for AudioCall._encode_pcma."""

    def test_encode_pcma__returns_one_byte_per_sample(self):
        """_encode_pcma returns a bytes object with one byte per input sample."""
        assert len(AudioCall._encode_pcma(np.zeros(160, dtype=np.float32))) == 160

    def test_encode_pcma__silence_encodes_consistently(self):
        """Zero-amplitude samples all encode to the same A-law codeword."""
        result = AudioCall._encode_pcma(np.zeros(10, dtype=np.float32))
        assert all(b == result[0] for b in result)


class TestEncodeViaAV:
    """Tests for AudioCall._encode_via_av."""

    def test_encode_via_av__g722_returns_bytes(self):
        """_encode_via_av produces non-empty bytes for G.722."""
        result = AudioCall._encode_via_av(
            np.zeros(320, dtype=np.float32), "g722", 16000
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_via_av__opus_returns_bytes(self):
        """_encode_via_av produces non-empty bytes for Opus (libopus)."""
        result = AudioCall._encode_via_av(
            np.zeros(960, dtype=np.float32), "libopus", 48000
        )
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestEncodeAudio:
    """Tests for AudioCall._encode_audio dispatch."""

    def test_encode_audio__dispatches_to_pcmu(self):
        """_encode_audio delegates to _encode_pcmu for PCMU media."""
        call = make_audio_call(media=PCMU_MEDIA)
        with patch.object(
            AudioCall, "_encode_pcmu", return_value=b"x" * 160
        ) as mock_enc:
            call._encode_audio(np.zeros(160, dtype=np.float32))
        mock_enc.assert_called_once()

    def test_encode_audio__dispatches_to_pcma(self):
        """_encode_audio delegates to _encode_pcma for PCMA media."""
        call = make_audio_call(media=PCMA_MEDIA)
        with patch.object(
            AudioCall, "_encode_pcma", return_value=b"x" * 160
        ) as mock_enc:
            call._encode_audio(np.zeros(160, dtype=np.float32))
        mock_enc.assert_called_once()

    def test_encode_audio__dispatches_to_opus(self):
        """_encode_audio delegates to _encode_via_av with libopus for Opus media."""
        call = make_audio_call(media=OPUS_MEDIA)
        samples = np.zeros(960, dtype=np.float32)
        with patch.object(AudioCall, "_encode_via_av", return_value=b"x") as mock_enc:
            call._encode_audio(samples)
        mock_enc.assert_called_once_with(samples, "libopus", 48000)

    def test_encode_audio__raises_for_unsupported_codec(self):
        """_encode_audio raises NotImplementedError for unrecognised encoding names."""
        call = make_audio_call(media=PCMU_MEDIA)
        call._encoding_name = "h264"
        with pytest.raises(NotImplementedError, match="Unsupported outbound codec"):
            call._encode_audio(np.zeros(160, dtype=np.float32))

    def test_encode_audio__raises_for_g722(self):
        """_encode_audio raises NotImplementedError for G.722 (handled by _packetize)."""
        call = make_audio_call(media=PCMU_MEDIA)
        call._encoding_name = "g722"
        with pytest.raises(NotImplementedError, match="Unsupported outbound codec"):
            call._encode_audio(np.zeros(320, dtype=np.float32))


class TestPacketize:
    """Tests for AudioCall._packetize."""

    def test_packetize__g722_encodes_whole_buffer_at_once(self):
        """_packetize calls _encode_via_av on the full audio buffer for G.722.

        Encoding the whole buffer at once preserves the ADPCM predictor state
        across 20 ms packet boundaries so the decoded audio is continuous.
        """
        call = make_audio_call(media=G722_MEDIA)
        audio = np.zeros(640, dtype=np.float32)
        fake_encoded = b"\xab" * 320
        with patch.object(
            AudioCall, "_encode_via_av", return_value=fake_encoded
        ) as mock_enc:
            packets = list(call._packetize(audio))
        mock_enc.assert_called_once_with(audio, "g722", 16000)
        assert len(packets) == 2
        assert packets[0] == b"\xab" * 160
        assert packets[1] == b"\xab" * 160

    def test_packetize__pcmu_encodes_per_chunk(self):
        """_packetize calls _encode_audio once per 20 ms chunk for PCMU."""
        call = make_audio_call(media=PCMU_MEDIA)
        audio = np.zeros(320, dtype=np.float32)
        with patch.object(
            call, "_encode_audio", return_value=b"\x7f" * 160
        ) as mock_enc:
            packets = list(call._packetize(audio))
        assert mock_enc.call_count == 2
        assert len(packets) == 2


class TestNextRTPPacket:
    """Tests for AudioCall._next_rtp_packet."""

    def test_next_rtp_packet__has_twelve_byte_header(self):
        """_next_rtp_packet produces a packet whose build() has a 12-byte RTP header."""
        call = make_audio_call(media=PCMU_MEDIA)
        data = bytes(call._next_rtp_packet(b"\x00" * 160))
        assert len(data) == 12 + 160
        assert data[0] == 0x80  # V=2, P=0, X=0, CC=0

    def test_next_rtp_packet__increments_seq_and_ts_each_call(self):
        """Each _next_rtp_packet call increments seq by 1 and ts by chunk size."""
        call = make_audio_call(media=PCMU_MEDIA)
        call._next_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 1
        assert call._rtp_ts == 160
        call._next_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 2
        assert call._rtp_ts == 320

    def test_next_rtp_packet__uses_negotiated_payload_type(self):
        """_next_rtp_packet uses the negotiated payload type in the RTP header."""
        call = make_audio_call(media=PCMA_MEDIA)
        assert bytes(call._next_rtp_packet(b"\x00" * 160))[1] == RTPPayloadType.PCMA


class TestSendRTPAudio:
    """Tests for AudioCall._send_rtp_audio."""

    async def test_send_rtp_audio__sends_to_remote_addr(self):
        """_send_rtp_audio sends RTP packets to the caller's registered address."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet") as mock_send:
            await call._send_rtp_audio(np.zeros(160, dtype=np.float32))
            mock_send.assert_called_once()
        data, addr = mock_send.call_args[0]
        assert addr == remote_addr
        assert len(bytes(data)) == 12 + 160  # 12-byte RTP header + 160 PCMU bytes

    async def test_send_rtp_audio__drops_audio_when_no_remote_addr(self, caplog):
        """Log a warning and drop audio when no RTP address is registered."""
        import logging  # noqa: PLC0415

        call = make_audio_call()
        call.rtp.calls = {}

        with (
            caplog.at_level(logging.WARNING, logger="voip.audio"),
            patch.object(call, "send_packet") as mock_send,
        ):
            await call._send_rtp_audio(np.zeros(160, dtype=np.float32))
        mock_send.assert_not_called()
        assert any("dropping audio" in r.message for r in caplog.records)

    async def test_send_rtp_audio__paces_packets_at_20ms_intervals(self):
        """_send_rtp_audio sleeps _rtp_packet_duration seconds between each packet."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.2", 5006)
        call.rtp.calls = {remote_addr: call}

        audio = np.zeros(320, dtype=np.float32)
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def _capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await original_sleep(0)

        with (
            patch("voip.audio.asyncio.sleep", side_effect=_capture_sleep),
            patch.object(call, "send_packet"),
        ):
            await call._send_rtp_audio(audio)

        assert len(sleep_calls) == 2
        assert all(s == call._rtp_packet_duration for s in sleep_calls)


class TestEstimatePayloadRMS:
    """Tests for AudioCall._estimate_payload_rms."""

    def test_estimate_payload_rms__silent_bytes_return_zero(self):
        """Constant bytes (silence-like signal) give near-zero std-dev RMS."""
        assert AudioCall._estimate_payload_rms(bytes([127] * 160)) == pytest.approx(0.0)

    def test_estimate_payload_rms__varying_bytes_return_positive(self):
        """Mixed byte values (speech-like signal) give positive RMS."""
        assert AudioCall._estimate_payload_rms(bytes([0, 255] * 80)) > 0.5


class TestAudioCallInit:
    def test_init__raises_value_error_for_none_encoding_name(self):
        """Raise ValueError when the negotiated format has no encoding name."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=96)
            ],  # dynamic PT, no rtpmap → no encoding name
        )
        with pytest.raises(ValueError, match="No encoding name"):
            make_audio_call(media=media)


class TestDecodeRaw:
    def test_decode_raw__raises_not_implemented_for_unsupported_codec(self):
        """Raise NotImplementedError when _decode_raw is called with an unknown codec."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(
                    payload_type=96, encoding_name="speex", sample_rate=8000
                )
            ],
        )
        call = make_audio_call(media=media)
        with pytest.raises(NotImplementedError, match="Unsupported inbound codec"):
            call._decode_raw(b"\x00" * 160)
