"""Tests for Whisper-based audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")
pytest.importorskip("faster_whisper")

from voip.audio import AudioCall, WhisperCall, _build_ogg_opus  # noqa: E402
from voip.rtp import RealtimeTransportProtocol, RTPPayloadType  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402


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


def make_whisper_call(
    model_mock: MagicMock, call_class=None, media: MediaDescription | None = None
) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or WhisperCall
    med = media if media is not None else OPUS_MEDIA
    with patch("voip.audio.WhisperModel", return_value=model_mock):
        return cls(
            rtp=MagicMock(),
            sip=MagicMock(),
            caller="sip:bob@biloxi.com",
            media=med,
        )


class TestWhisperCall:
    def test_whisper_call__is_audio_call(self):
        """WhisperCall is a subclass of AudioCall."""
        from voip.audio import AudioCall  # noqa: PLC0415

        assert issubclass(WhisperCall, AudioCall)

    def test_class_attrs__chunk_duration(self):
        """chunk_duration controls how many seconds are buffered before transcription."""
        assert WhisperCall.chunk_duration == 5

    def test_init__packet_threshold__opus(self):
        """_packet_threshold is 250 for Opus with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__g722(self):
        """_packet_threshold is 250 for G.722 with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__pcma(self):
        """_packet_threshold is 250 for PCMA with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__pcmu(self):
        """_packet_threshold is 250 for PCMU with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call._packet_threshold == 250

    def test_init__stores_media(self):
        """Media is stored and accessible as self.media."""
        call = make_whisper_call(MagicMock())
        assert call.media is OPUS_MEDIA

    def test_init__derives_payload_type_from_opus_media(self):
        """payload_type is 111 (Opus) when given OPUS_MEDIA."""
        call = make_whisper_call(MagicMock())
        assert call.payload_type == RTPPayloadType.OPUS

    def test_init__derives_payload_type_from_pcma_media(self):
        """payload_type is 8 (PCMA) when given PCMA_MEDIA."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA

    def test_audio_received__does_not_buffer_on_whisper_call(self):
        """audio_received on WhisperCall schedules a task, not a packet buffer."""
        call = make_whisper_call(MagicMock())
        assert not hasattr(call, "_audio_packets")

    def test_audio_received__below_threshold_no_transcription(self):
        """Don't trigger transcription until the packet threshold is reached."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        # Feed one packet to the base-class buffer (threshold=1500, so no emit)
        call._audio_buffer.append(b"opus_packet")
        model_mock.transcribe.assert_not_called()

    async def test_audio_received__triggers_transcription_when_buffer_full(self):
        """Transcription fires when audio_received is called with decoded PCM."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "hello"
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1  # 1 s @ 48 kHz / 960 samples = 50 packets

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, SmallChunkCall)
        pcm_samples = np.zeros(16000, dtype=np.float32)
        # audio_received now takes decoded PCM directly (np.ndarray)
        call.audio_received(pcm_samples)
        await asyncio.sleep(0.1)
        assert transcriptions == ["hello"]

    async def test_transcribe__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription text."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "  hello world  "
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(WhisperCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        await call._transcribe(np.zeros(16000, dtype=np.float32))
        assert transcriptions == ["hello world"]

    def test_run_transcription__passes_numpy_array_directly(self):
        """Pass a numpy float32 array to the Whisper model without file I/O."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "test"
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        audio = np.zeros(16000, dtype=np.float32)
        assert call._run_transcription(audio) == "test"
        model_mock.transcribe.assert_called_once_with(audio)

    def test_run_transcription__no_file_written(self):
        """The transcription path must not write any files to disk."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = ""
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        with patch(
            "builtins.open", side_effect=AssertionError("open() must not be called")
        ):
            call._run_transcription(np.zeros(16000, dtype=np.float32))

    def test_decode_via_av__opus(self):
        """Pipe Ogg Opus data through PyAV and return a float32 PCM array."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_array = np.zeros(16000, dtype=np.float32)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = pcm_array
        mock_resampled = [mock_frame]
        with patch("voip.audio.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.side_effect = [mock_resampled, []]
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = [MagicMock()]
            mock_av.open.return_value = mock_container
            mock_av.AVError = Exception
            result = call._decode_via_av(
                b"fake_ogg_data", input_format="ogg", input_sample_rate=None
            )
        mock_av.open.assert_called_once()
        assert result.dtype == np.float32

    def test_decode_via_av__raises_on_av_error(self):
        """Raise RuntimeError when PyAV raises AVError."""
        call = make_whisper_call(MagicMock())
        with (
            patch("voip.audio.av") as mock_av,
            pytest.raises(RuntimeError, match="Audio decoding failed"),
        ):
            mock_av.AVError = RuntimeError
            mock_av.audio.resampler.AudioResampler.return_value = MagicMock()
            mock_av.open.side_effect = RuntimeError("av error")
            call._decode_via_av(b"bad_data", input_format="ogg", input_sample_rate=None)

    def test_decode_raw__opus__wraps_in_ogg(self):
        """Decode Opus packets by wrapping them in an Ogg container before calling PyAV."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call.payload_type == RTPPayloadType.OPUS
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "ogg"

    def test_decode_raw__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "alaw"

    def test_decode_raw__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMU
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "mulaw"

    def test_decode_raw__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call.payload_type == RTPPayloadType.G722
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "g722"

    def test_decode_raw__unknown__raises(self):
        """Raise NotImplementedError for unsupported encoding names."""
        unknown_media = _make_media("99")
        call = make_whisper_call(MagicMock(), media=unknown_media)
        with pytest.raises(NotImplementedError, match="Unsupported"):
            call._decode_raw([b"pkt"])

    def test_decode_raw__uses_sample_rate_from_media(self):
        """Pass the sample rate from the MediaDescription to _decode_via_av."""
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_whisper_call(MagicMock(), media=wideband_pcma)
        assert call.sample_rate == 16000
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_sample_rate") == 16000

    @pytest.mark.asyncio
    async def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received audio frame."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="voip.audio"):
            call.audio_received(np.zeros(1, dtype=np.float32))
            await asyncio.sleep(0)  # let the task run so coroutine is not left dangling
        assert any("Audio received" in r.message for r in caplog.records)


class TestBuildOggOpus:
    def test_build_ogg_opus__starts_with_ogg_magic(self):
        """The resulting container starts with the Ogg capture pattern 'OggS'."""
        assert _build_ogg_opus([b"packet"]).startswith(b"OggS")

    def test_build_ogg_opus__contains_opus_head(self):
        """The resulting container includes the OpusHead identification header."""
        assert b"OpusHead" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__contains_opus_tags(self):
        """The resulting container includes the OpusTags comment header."""
        assert b"OpusTags" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__non_empty_for_single_packet(self):
        """Produce a non-empty Ogg container for a single Opus packet."""
        assert len(_build_ogg_opus([b"x" * 100])) > 100

    def test_build_ogg_opus__empty_packets_list(self):
        """Produce a valid Ogg container even with an empty packet list."""
        result = _build_ogg_opus([])
        assert b"OggS" in result

    def test_build_ogg_opus__large_packet_uses_255_lacing(self):
        """Produce a valid Ogg container when a packet exceeds 254 bytes (lacing spans 255)."""
        large_packet = b"x" * 256
        result = _build_ogg_opus([large_packet])
        assert b"OggS" in result
        assert len(result) > 256

    def test_build_ogg_opus__multiple_pages(self):
        """Produce multiple Ogg pages when more than 50 packets are provided."""
        packets = [b"x" * 10] * 55
        result = _build_ogg_opus(packets)
        assert result.count(b"OggS") > 3


def make_audio_call(**kwargs) -> AudioCall:
    """Create an AudioCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
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
        sentinel = object()
        make_audio_call().audio_received(sentinel)  # must not raise

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored as dataclass fields."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = AudioCall(rtp=mock_rtp, sip=mock_sip)
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
        """Default sample_rate is 8000 Hz when no MediaDescription is given."""
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
        """Default payload_type is 0 when no MediaDescription is given."""
        assert make_audio_call().payload_type == 0

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

    def test_chunk_duration__default_is_zero(self):
        """chunk_duration defaults to 0 (per-packet mode)."""
        assert AudioCall.chunk_duration == 0

    @pytest.mark.asyncio
    async def test_datagram_received__forwards_audio_payload(self):
        """datagram_received parses the RTP packet and calls audio_received after decode."""
        import struct  # noqa: PLC0415

        DECODED = object()  # sentinel: returned by mocked _decode_raw
        received: list = []

        class ConcreteCall(AudioCall):
            def _decode_raw(self, raw_packets: list[bytes]):
                return DECODED  # skip real av decode in unit tests

            def audio_received(self, audio) -> None:
                received.append(audio)

        rtp_packet = struct.pack(">BBHII", 0x80, 111 & 0x7F, 1, 0, 0) + b"audio"
        call = ConcreteCall(rtp=MagicMock(), sip=MagicMock())
        call.datagram_received(rtp_packet, ("127.0.0.1", 5004))
        await asyncio.sleep(0.05)  # let the executor task run
        assert received == [DECODED]

    @pytest.mark.asyncio
    async def test_datagram_received__chunk_duration__buffers_until_threshold(self):
        """Buffer packets and emit audio_received only when the threshold is reached."""
        import struct  # noqa: PLC0415

        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        received: list = []
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )

        class ChunkedCall(AudioCall):
            chunk_duration = 1  # 1 s @ 8 kHz / 160 samples = 50 packets

            def _decode_raw(self, raw_packets: list[bytes]):
                return raw_packets  # skip real av decode; pass raw list as "audio"

            def audio_received(self, audio) -> None:
                received.append(audio)

        call = ChunkedCall(rtp=MagicMock(), sip=MagicMock(), media=media)
        assert call._packet_threshold == 50
        rtp_packet = struct.pack(">BBHII", 0x80, 8 & 0x7F, 1, 0, 0) + b"x"
        for _ in range(49):
            call.datagram_received(rtp_packet, ("127.0.0.1", 5004))
        assert len(call._audio_buffer) == 49
        assert received == []  # threshold not yet reached
        call.datagram_received(
            struct.pack(">BBHII", 0x80, 8 & 0x7F, 2, 0, 0) + b"last",
            ("127.0.0.1", 5004),
        )
        assert len(call._audio_buffer) == 0  # buffer drained
        await asyncio.sleep(0.05)  # let the executor task run
        assert len(received) == 1
        assert len(received[0]) == 50


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
