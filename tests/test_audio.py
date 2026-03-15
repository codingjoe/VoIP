"""Tests for audio call handler and codec utilities."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.audio import AudioCall, EchoCall, VoiceActivityCall  # noqa: E402
from voip.codecs.g722 import G722  # noqa: E402
from voip.codecs.opus import Opus  # noqa: E402
from voip.codecs.pcma import PCMA  # noqa: E402
from voip.codecs.pcmu import PCMU  # noqa: E402
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
            def decode_payload(self, packet: bytes) -> np.ndarray:
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
        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2"])
        result = AudioCall.negotiate_codec(media)
        assert isinstance(result, MediaDescription)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""

        class PCMAOnlyCall(AudioCall):
            PREFERRED_CODECS = [PCMA]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8

    def test_preferred_codecs__class_attribute(self):
        """PREFERRED_CODECS is a class attribute on AudioCall with Opus first when PyAV is available."""
        codec_classes = AudioCall.PREFERRED_CODECS
        assert isinstance(codec_classes, list)
        pts = [c.payload_type for c in codec_classes]
        assert pts[0] == 111  # Opus is highest priority when PyAV is present
        assert 8 in pts  # PCMA present
        assert 0 in pts  # PCMU present


class TestCodecAssignment:
    """Tests that __post_init__ assigns the correct codec class."""

    def test_opus_media__codec_is_opus(self):
        """Opus media assigns the Opus codec class."""
        call = make_audio_call(media=OPUS_MEDIA)
        assert call.codec is Opus
        assert call.codec.sample_rate_hz == 48000
        assert call.codec.frame_size == 960
        assert call.codec.timestamp_increment == 960

    def test_g722_media__codec_is_g722(self):
        """G.722 media assigns the G722 codec class with correct rates."""
        call = make_audio_call(media=G722_MEDIA)
        assert call.codec is G722
        assert call.codec.sample_rate_hz == 16000
        assert call.codec.frame_size == 320
        assert call.codec.timestamp_increment == 160

    def test_pcmu_media__codec_is_pcmu(self):
        """PCMU media assigns the PCMU codec class."""
        call = make_audio_call(media=PCMU_MEDIA)
        assert call.codec is PCMU
        assert call.codec.sample_rate_hz == 8000
        assert call.codec.frame_size == 160
        assert call.codec.timestamp_increment == 160

    def test_pcma_media__codec_is_pcma(self):
        """PCMA media assigns the PCMA codec class."""
        call = make_audio_call(media=PCMA_MEDIA)
        assert call.codec is PCMA


class TestResample:
    """Tests for AudioCall.resample."""

    def test_resample__downsamples_from_24khz_to_8khz(self):
        """Resample reduces 24 000 samples at 24 kHz to 8 000 samples at 8 kHz."""
        audio = np.zeros(24000, dtype=np.float32)
        assert len(AudioCall.resample(audio, 24000, 8000)) == 8000

    def test_resample__passthrough_when_rate_matches(self):
        """Resample returns the original array unchanged when rates are equal."""
        audio = np.zeros(8000, dtype=np.float32)
        assert AudioCall.resample(audio, 8000, 8000) is audio


class TestDecodePayload:
    """Tests for AudioCall.decode_payload."""

    def test_decode_payload__delegates_to_codec(self):
        """decode_payload calls self.codec.decode with output and input rates."""
        call = make_audio_call(media=PCMA_MEDIA)
        with patch.object(
            PCMA, "decode", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call.decode_payload(b"payload")
        mock_decode.assert_called_once_with(
            b"payload",
            AudioCall.RESAMPLING_RATE_HZ,
            input_rate_hz=call.sample_rate,
        )

    def test_decode_payload__passes_sample_rate_from_media(self):
        """decode_payload passes the SDP-negotiated sample rate as input_rate_hz."""
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_audio_call(media=wideband_pcma)
        assert call.sample_rate == 16000
        with patch.object(
            PCMA, "decode", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call.decode_payload(b"pkt")
        mock_decode.assert_called_once_with(
            b"pkt",
            AudioCall.RESAMPLING_RATE_HZ,
            input_rate_hz=16000,
        )

    def test_decode_payload__raises_for_unsupported_codec(self):
        """Raise NotImplementedError when constructed with an unsupported codec."""
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
        with pytest.raises(NotImplementedError, match="Unsupported codec"):
            make_audio_call(media=media)


class TestAudioCallInit:
    def test_init__raises_value_error_for_none_encoding_name(self):
        """Raise ValueError when the negotiated format has no encoding name."""
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=96)
            ],  # dynamic PT, no rtpmap -> no encoding name
        )
        with pytest.raises(ValueError, match="No encoding name"):
            make_audio_call(media=media)


class TestNextRTPPacket:
    """Tests for AudioCall.next_rtp_packet."""

    def test_next_rtp_packet__has_twelve_byte_header(self):
        """next_rtp_packet produces a packet whose build() has a 12-byte RTP header."""
        call = make_audio_call(media=PCMU_MEDIA)
        data = bytes(call.next_rtp_packet(b"\x00" * 160))
        assert len(data) == 12 + 160
        assert data[0] == 0x80  # V=2, P=0, X=0, CC=0

    def test_next_rtp_packet__increments_seq_and_ts_each_call(self):
        """Each next_rtp_packet call increments seq by 1 and ts by chunk size."""
        call = make_audio_call(media=PCMU_MEDIA)
        call.next_rtp_packet(b"\x00" * 160)
        assert call.rtp_sequence_number == 1
        assert call.rtp_timestamp == 160
        call.next_rtp_packet(b"\x00" * 160)
        assert call.rtp_sequence_number == 2
        assert call.rtp_timestamp == 320

    def test_next_rtp_packet__uses_negotiated_payload_type(self):
        """next_rtp_packet uses the negotiated payload type in the RTP header."""
        call = make_audio_call(media=PCMA_MEDIA)
        assert bytes(call.next_rtp_packet(b"\x00" * 160))[1] == RTPPayloadType.PCMA


class TestSendRTPAudio:
    """Tests for AudioCall.send_rtp_audio."""

    async def test_send_rtp_audio__sends_to_remote_addr(self):
        """send_rtp_audio sends RTP packets to the caller's registered address."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet") as mock_send:
            await call.send_rtp_audio(np.zeros(160, dtype=np.float32))
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
            await call.send_rtp_audio(np.zeros(160, dtype=np.float32))
        mock_send.assert_not_called()
        assert any("dropping audio" in r.message for r in caplog.records)

    async def test_send_rtp_audio__paces_packets_at_20ms_intervals(self):
        """send_rtp_audio sleeps RTP_PACKET_DURATION_SECS between each packet."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.2", 5006)
        call.rtp.calls = {remote_addr: call}

        audio = np.zeros(320, dtype=np.float32)
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await original_sleep(0)

        with (
            patch("voip.audio.asyncio.sleep", side_effect=capture_sleep),
            patch.object(call, "send_packet"),
        ):
            await call.send_rtp_audio(audio)

        assert len(sleep_calls) == 2
        assert all(s == call.RTP_PACKET_DURATION_SECS for s in sleep_calls)


def make_echo_call(**kwargs) -> EchoCall:
    """Create an EchoCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
        "media": PCMU_MEDIA,
        "caller": CallerID(""),
    }
    defaults.update(kwargs)
    return EchoCall(**defaults)


def make_vac_call(**kwargs) -> VoiceActivityCall:
    """Create a VoiceActivityCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
        "media": PCMU_MEDIA,
        "caller": CallerID(""),
    }
    defaults.update(kwargs)
    return VoiceActivityCall(**defaults)


class TestVoiceActivityCall:
    """Tests for the shared VAD infrastructure in VoiceActivityCall."""

    def test_voice_activity_call__is_audio_call(self):
        """VoiceActivityCall is a subclass of AudioCall."""
        assert issubclass(VoiceActivityCall, AudioCall)

    def test_collect_audio__returns_true_for_speech(self):
        """collect_audio returns True when RMS exceeds speech_threshold."""
        call = make_vac_call()
        assert call.collect_audio(np.ones(160, dtype=np.float32), rms=1.0) is True

    def test_collect_audio__returns_false_for_silence(self):
        """collect_audio returns False when RMS is at or below speech_threshold."""
        call = make_vac_call()
        assert call.collect_audio(np.zeros(160, dtype=np.float32), rms=0.0) is False

    def test_audio_received__buffers_speech_frames(self):
        """audio_received buffers frames with RMS above speech_threshold."""
        call = make_vac_call()
        call.audio_received(audio=np.ones(160, dtype=np.float32), rms=1.0)
        assert len(call.speech_buffer) == 1

    def test_audio_received__does_not_buffer_silence_frames(self):
        """audio_received skips frames with RMS at or below speech_threshold."""
        call = make_vac_call()
        call.audio_received(audio=np.zeros(160, dtype=np.float32), rms=0.0)
        assert len(call.speech_buffer) == 0

    def test_on_audio_speech__cancels_silence_timer(self):
        """on_audio_speech cancels a running silence timer."""
        call = make_vac_call()
        handle = MagicMock()
        call.silence_handle = handle
        call.on_audio_speech()
        handle.cancel.assert_called_once()
        assert call.silence_handle is None

    def test_on_audio_speech__noop_when_no_timer(self):
        """on_audio_speech does nothing when no silence timer is running."""
        call = make_vac_call()
        call.on_audio_speech()  # must not raise
        assert call.silence_handle is None

    @pytest.mark.asyncio
    async def test_on_audio_silence__arms_timer_when_buffer_has_speech(self):
        """on_audio_silence schedules the silence timer when speech is buffered."""
        call = make_vac_call()
        call.speech_buffer.append(np.ones(160, dtype=np.float32))
        call.on_audio_silence()
        assert call.silence_handle is not None
        call.silence_handle.cancel()

    @pytest.mark.asyncio
    async def test_on_audio_silence__noop_when_buffer_is_empty(self):
        """on_audio_silence does not schedule a timer when no speech is buffered."""
        call = make_vac_call()
        call.on_audio_silence()
        assert call.silence_handle is None

    @pytest.mark.asyncio
    async def test_on_audio_silence__noop_when_timer_already_running(self):
        """on_audio_silence does not replace a running silence timer."""
        call = make_vac_call()
        call.speech_buffer.append(np.ones(160, dtype=np.float32))
        call.on_audio_silence()
        first_handle = call.silence_handle
        call.on_audio_silence()
        assert call.silence_handle is first_handle
        call.silence_handle.cancel()

    @pytest.mark.asyncio
    async def test_flush_speech_buffer__resets_handle_and_schedules_ready(self):
        """flush_speech_buffer clears the buffer and schedules speech_buffer_ready."""
        call = make_vac_call()
        call.speech_buffer.append(np.ones(160, dtype=np.float32))
        with patch.object(
            call, "speech_buffer_ready", new_callable=AsyncMock
        ) as mock_ready:
            call.flush_speech_buffer()
            await asyncio.sleep(0)
        assert call.silence_handle is None
        assert len(call.speech_buffer) == 0
        mock_ready.assert_awaited_once()

    def test_flush_speech_buffer__noop_when_buffer_empty(self):
        """flush_speech_buffer does nothing when no speech is buffered."""
        call = make_vac_call()
        with patch("voip.audio.asyncio.create_task") as mock_ct:
            call.flush_speech_buffer()
        mock_ct.assert_not_called()

    @pytest.mark.asyncio
    async def test_speech_buffer_ready__noop_in_base(self):
        """speech_buffer_ready is a no-op in the base VoiceActivityCall."""
        call = make_vac_call()
        await call.speech_buffer_ready(
            np.zeros(160, dtype=np.float32)
        )  # must not raise


class TestEchoCall:
    """Tests for EchoCall speech echo playback."""

    def test_echo_call__is_voice_activity_call(self):
        """EchoCall is a subclass of VoiceActivityCall."""
        assert issubclass(EchoCall, VoiceActivityCall)

    @pytest.mark.asyncio
    async def test_speech_buffer_ready__sends_resampled_audio(self):
        """speech_buffer_ready resamples from RESAMPLING_RATE_HZ to codec rate and sends via RTP."""
        call = make_echo_call(media=PCMU_MEDIA)
        audio = np.ones(160, dtype=np.float32)
        with patch.object(call, "send_rtp_audio", new_callable=AsyncMock) as mock_send:
            await call.speech_buffer_ready(audio)
        mock_send.assert_awaited_once()
        sent_audio = mock_send.call_args[0][0]
        # PCMU sample_rate_hz == 8000; RESAMPLING_RATE_HZ == 16000 → half length
        expected_len = round(
            len(audio) * call.codec.sample_rate_hz / call.RESAMPLING_RATE_HZ
        )
        assert len(sent_audio) == expected_len

    @pytest.mark.asyncio
    async def test_audio_received__echoes_after_sustained_silence(self):
        """Speech followed by silence_gap of silence triggers echo playback."""
        call = make_echo_call(silence_gap=0.01)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        speech = np.ones(160, dtype=np.float32) * 0.5
        silence = np.zeros(160, dtype=np.float32)

        sent: list[np.ndarray] = []

        async def capture_send(audio: np.ndarray) -> None:
            sent.append(audio)

        with patch.object(call, "send_rtp_audio", side_effect=capture_send):
            call.audio_received(audio=speech, rms=1.0)
            call.audio_received(audio=silence, rms=0.0)
            await asyncio.sleep(0.05)

        assert len(sent) == 1
