"""Tests for audio call handler and codec utilities."""

from __future__ import annotations

import asyncio
import datetime
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
            supported_codecs = [PCMA]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8

    def test_preferred_codecs__class_attribute(self):
        """PREFERRED_CODECS is a class attribute on AudioCall with Opus first when PyAV is available."""
        codec_classes = AudioCall.supported_codecs
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
        """decode_payload routes through PerPacketDecoder which calls codec.decode."""
        call = make_audio_call(media=PCMA_MEDIA)
        with patch.object(
            PCMA, "decode", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call.decode_payload(b"payload")
        mock_decode.assert_called_once_with(
            b"payload",
            call.sampling_rate_hz,
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
            call.sampling_rate_hz,
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

    def test_decode_payload__g722_uses_stateful_decoder(self):
        """G.722 AudioCall uses a G722Decoder that preserves ADPCM state."""
        from voip.codecs.g722 import G722Decoder  # noqa: PLC0415

        call = make_audio_call(media=G722_MEDIA)
        assert isinstance(call.payload_decoder, G722Decoder)

    def test_decode_payload__pcma_uses_per_packet_decoder(self):
        """PCMA AudioCall uses a PerPacketDecoder (stateless)."""
        from voip.codecs.base import PerPacketDecoder  # noqa: PLC0415

        call = make_audio_call(media=PCMA_MEDIA)
        assert isinstance(call.payload_decoder, PerPacketDecoder)


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
        """The first RTP packet is transmitted synchronously inside send_audio."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet") as mock_send:
            await call.send_audio(np.zeros(160, dtype=np.float32))
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
            await call.send_audio(np.zeros(160, dtype=np.float32))
        mock_send.assert_not_called()
        assert any("dropping audio" in r.message for r in caplog.records)

    async def test_send_rtp_audio__paces_packets_at_20ms_intervals(self):
        """Subsequent packets are scheduled at rpt_packet_duration intervals via call_later."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.2", 5006)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet"):
            await call.send_audio(np.zeros(320, dtype=np.float32))

        loop = asyncio.get_event_loop()
        assert call.outbound_handle is not None
        assert call.outbound_handle.when() == pytest.approx(
            loop.time() + call.rpt_packet_duration.total_seconds(), abs=0.01
        )

    async def test_cancel_outbound_audio__cancels_handle_and_clears(self):
        """cancel_outbound_audio cancels the pending handle and sets outbound_handle to None."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet"):
            await call.send_audio(np.zeros(320, dtype=np.float32))

        handle = call.outbound_handle
        call.cancel_outbound_audio()

        assert handle.cancelled()
        assert call.outbound_handle is None

    async def test_send_audio__preempts_pending_handle(self):
        """A second send_audio cancels the pending handle from the first call."""
        call = make_audio_call(media=PCMU_MEDIA)
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        with patch.object(call, "send_packet"):
            await call.send_audio(np.zeros(320, dtype=np.float32))
            first_handle = call.outbound_handle
            await call.send_audio(np.zeros(320, dtype=np.float32))

        assert first_handle.cancelled()


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

    def test_audio_received__appends_to_speech_buffer(self):
        """audio_received concatenates frames to the speech buffer regardless of RMS."""
        call = make_vac_call()
        audio = np.ones(160, dtype=np.float32) * 0.5
        with patch("voip.audio.asyncio.get_event_loop"):
            call.audio_received(audio=audio, rms=0.5)
        assert call._speech_buffer.size == 160

    def test_audio_received__speech_cancels_flush_timer(self):
        """audio_received with speech-level RMS cancels any running flush timer."""
        call = make_vac_call()
        handle = MagicMock()
        call._flush_voice_buffer_handle = handle
        call.audio_received(audio=np.ones(160, dtype=np.float32), rms=1.0)
        handle.cancel.assert_called_once()
        assert call._flush_voice_buffer_handle is None

    def test_audio_received__silence_arms_flush_timer(self):
        """audio_received with silence-level RMS schedules the flush timer once."""
        call = make_vac_call()
        with patch("voip.audio.asyncio.get_event_loop") as mock_loop:
            handle = MagicMock()
            mock_loop.return_value.call_later.return_value = handle
            call.audio_received(audio=np.zeros(160, dtype=np.float32), rms=0.0)
        mock_loop.return_value.call_later.assert_called_once_with(
            call.silence_gap.total_seconds(),
            call.flush_voice_buffer,
        )
        assert call._flush_voice_buffer_handle is handle

    def test_audio_received__silence_does_not_rearm_when_timer_running(self):
        """A second silence frame does not replace a running flush timer."""
        call = make_vac_call()
        call._flush_voice_buffer_handle = MagicMock()
        with patch("voip.audio.asyncio.get_event_loop") as mock_loop:
            call.audio_received(audio=np.zeros(160, dtype=np.float32), rms=0.0)
        mock_loop.return_value.call_later.assert_not_called()

    def test_on_audio_speech__cancels_flush_handle(self):
        """on_audio_speech cancels the flush timer when one is running."""
        call = make_vac_call()
        handle = MagicMock()
        call._flush_voice_buffer_handle = handle
        call.on_audio_speech()
        handle.cancel.assert_called_once()
        assert call._flush_voice_buffer_handle is None

    def test_on_audio_speech__noop_when_no_timer(self):
        """on_audio_speech does nothing when no flush timer is running."""
        call = make_vac_call()
        call.on_audio_speech()  # must not raise
        assert call._flush_voice_buffer_handle is None

    @pytest.mark.asyncio
    async def test_on_audio_silence__arms_timer(self):
        """on_audio_silence schedules the flush timer via the event loop."""
        call = make_vac_call()
        call.on_audio_silence()
        assert call._flush_voice_buffer_handle is not None
        call._flush_voice_buffer_handle.cancel()

    @pytest.mark.asyncio
    async def test_on_audio_silence__noop_when_timer_already_running(self):
        """on_audio_silence does not replace a running flush timer."""
        call = make_vac_call()
        call.on_audio_silence()
        first_handle = call._flush_voice_buffer_handle
        call.on_audio_silence()
        assert call._flush_voice_buffer_handle is first_handle
        call._flush_voice_buffer_handle.cancel()

    @pytest.mark.asyncio
    async def test_flush_voice_buffer__clears_buffer_and_resets_handle(self):
        """flush_voice_buffer resets the handle and clears the speech buffer."""
        call = make_vac_call()
        call._speech_buffer = np.ones(call.sampling_rate_hz * 2, dtype=np.float32)
        call.flush_voice_buffer()
        assert call._flush_voice_buffer_handle is None
        assert call._speech_buffer.size == 0

    @pytest.mark.asyncio
    async def test_flush_voice_buffer__schedules_voice_received_for_loud_audio(self):
        """flush_voice_buffer schedules voice_received for utterances above RMS threshold."""
        call = make_vac_call()
        # Fill buffer with 2 seconds of loud audio.
        call._speech_buffer = np.ones(call.sampling_rate_hz * 2, dtype=np.float32)
        with patch("voip.audio.asyncio.create_task") as mock_ct:
            call.flush_voice_buffer()
        mock_ct.assert_called_once()

    def test_flush_voice_buffer__discards_short_utterances(self):
        """flush_voice_buffer drops utterances shorter than silence_gap seconds."""
        call = make_vac_call()
        # 10 samples is much shorter than sampling_rate_hz * silence_gap_secs.
        call._speech_buffer = np.ones(10, dtype=np.float32)
        with patch("voip.audio.asyncio.create_task") as mock_ct:
            call.flush_voice_buffer()
        mock_ct.assert_not_called()

    def test_flush_voice_buffer__discards_quiet_utterances(self):
        """flush_voice_buffer drops utterances below utterances_rms_threshold."""
        call = make_vac_call()
        # Buffer long enough but nearly silent.
        call._speech_buffer = np.zeros(call.sampling_rate_hz * 2, dtype=np.float32)
        with patch("voip.audio.asyncio.create_task") as mock_ct:
            call.flush_voice_buffer()
        mock_ct.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_received__noop_in_base(self):
        """voice_received is a no-op in the base VoiceActivityCall."""
        call = make_vac_call()
        await call.voice_received(np.zeros(160, dtype=np.float32))  # must not raise


class TestEchoCall:
    """Tests for EchoCall speech echo playback."""

    def test_echo_call__is_voice_activity_call(self):
        """EchoCall is a subclass of VoiceActivityCall."""
        assert issubclass(EchoCall, VoiceActivityCall)

    @pytest.mark.asyncio
    async def test_voice_received__sends_resampled_audio(self):
        """voice_received resamples from sampling_rate_hz to codec rate and sends via RTP."""
        call = make_echo_call(media=PCMU_MEDIA)
        audio = np.ones(160, dtype=np.float32)
        with patch.object(call, "send_audio", new_callable=AsyncMock) as mock_send:
            await call.voice_received(audio)
        mock_send.assert_awaited_once()
        sent_audio = mock_send.call_args[0][0]
        # PCMU sample_rate_hz == 8000; sampling_rate_hz == 16000 → half length
        expected_len = round(
            len(audio) * call.codec.sample_rate_hz / call.sampling_rate_hz
        )
        assert len(sent_audio) == expected_len

    @pytest.mark.asyncio
    async def test_audio_received__echoes_after_sustained_silence(self):
        """Speech followed by silence_gap of silence triggers echo playback."""
        call = make_echo_call(
            silence_gap=datetime.timedelta(milliseconds=10),
        )
        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        speech = np.ones(160, dtype=np.float32) * 0.5
        silence = np.zeros(160, dtype=np.float32)

        sent: list[np.ndarray] = []

        async def capture_send(audio: np.ndarray) -> None:
            sent.append(audio)

        with patch.object(call, "send_audio", side_effect=capture_send):
            call.audio_received(audio=speech, rms=1.0)
            call.audio_received(audio=silence, rms=0.0)
            await asyncio.sleep(0.05)

        assert len(sent) == 1
