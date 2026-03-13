"""Tests for AI-powered call handlers (WhisperCall and AgentCall)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faster_whisper")
pytest.importorskip("ollama")
pytest.importorskip("pocket_tts")

from voip.ai import AgentCall, WhisperCall  # noqa: E402
from voip.audio import AudioCall  # noqa: E402
from voip.rtp import RTPPayloadType  # noqa: E402
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
    with patch("voip.ai.WhisperModel", return_value=model_mock):
        return cls(
            rtp=MagicMock(),
            sip=MagicMock(),
            caller="sip:bob@biloxi.com",
            media=med,
        )


def make_agent_call(
    model_mock: MagicMock,
    tts_mock: MagicMock,
    call_class=None,
    media: MediaDescription | None = None,
) -> AgentCall:
    """Return an AgentCall with mocked Whisper model and Pocket TTS model."""
    cls = call_class or AgentCall
    med = media if media is not None else OPUS_MEDIA
    with (
        patch("voip.ai.WhisperModel", return_value=model_mock),
        patch("voip.ai.TTSModel") as tts_cls,
    ):
        tts_cls.load_model.return_value = tts_mock
        return cls(
            rtp=MagicMock(),
            sip=MagicMock(),
            caller="sip:bob@biloxi.com",
            media=med,
        )


class TestWhisperCall:
    def test_whisper_call__is_audio_call(self):
        """WhisperCall is a subclass of AudioCall."""
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

    async def test_transcribe__empty_transcription_not_delivered(self):
        """Whitespace-only transcription is silently discarded."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "   "
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(WhisperCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        await call._transcribe(np.zeros(16000, dtype=np.float32))
        assert transcriptions == []

    @pytest.mark.asyncio
    async def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received audio frame."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="voip.ai"):
            call.audio_received(np.zeros(1, dtype=np.float32))
            await asyncio.sleep(0)  # let the task run so coroutine is not left dangling
        assert any("Audio received" in r.message for r in caplog.records)


class TestAgentCall:
    def test_agent_call__is_whisper_call(self):
        """AgentCall is a subclass of WhisperCall."""
        assert issubclass(AgentCall, WhisperCall)

    def test_init__loads_tts_model_when_none(self):
        """Load the default Pocket TTS model when tts_model is None."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            tts_cls.load_model.return_value = tts_mock
            call = AgentCall(rtp=MagicMock(), sip=MagicMock(), media=OPUS_MEDIA)
        tts_cls.load_model.assert_called_once()
        assert call._tts_instance is tts_mock

    def test_init__uses_provided_tts_model(self):
        """Use the provided TTSModel instance instead of loading a new one."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            call = AgentCall(
                rtp=MagicMock(),
                sip=MagicMock(),
                media=OPUS_MEDIA,
                tts_model=tts_mock,
            )
        tts_cls.load_model.assert_not_called()
        assert call._tts_instance is tts_mock

    def test_init__loads_voice_state(self):
        """Get the voice state from the TTS model on init."""
        tts_mock = MagicMock()
        voice_state = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = voice_state
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            tts_cls.load_model.return_value = tts_mock
            call = AgentCall(
                rtp=MagicMock(), sip=MagicMock(), media=OPUS_MEDIA, voice="alba"
            )
        tts_mock.get_state_for_audio_prompt.assert_called_once_with("alba")
        assert call._voice_state is voice_state

    def test_transcription_received__ignores_empty_text(self):
        """transcription_received does not schedule _respond for empty text."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with patch("asyncio.create_task") as mock_create_task:
            call.transcription_received("")
        mock_create_task.assert_not_called()

    def test_init__initializes_chat_history_with_system_prompt(self):
        """Chat history is seeded with a system prompt mentioning a phone call."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert len(call._messages) == 1
        assert call._messages[0]["role"] == "system"
        assert "phone" in call._messages[0]["content"].lower()

    async def test_respond__calls_ollama_and_sends_speech(self):
        """_respond fetches an Ollama reply, records it in history, and sends speech."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        mock_response = MagicMock()
        mock_response.message.content = "I am an AI assistant."
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", new_callable=AsyncMock) as mock_send_speech,
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond("hello")

        mock_send_speech.assert_awaited_once_with("I am an AI assistant.")
        assert {"role": "user", "content": "hello"} in call._messages
        assert {"role": "assistant", "content": "I am an AI assistant."} in call._messages

    async def test_respond__passes_full_history_to_ollama(self):
        """_respond passes the full message history (including system prompt) to Ollama."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        mock_response = MagicMock()
        mock_response.message.content = "reply"
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", new_callable=AsyncMock),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
        _, kwargs = mock_client.chat.call_args
        messages = kwargs.get("messages") or mock_client.chat.call_args[0][0]
        # First message is the system prompt
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_respond__logs_exception_on_error(self, caplog):
        """Log an exception when Ollama raises an error."""
        import logging

        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            caplog.at_level(logging.ERROR, logger="voip.ai"),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=RuntimeError("ollama error"))
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
        assert any("agent response" in r.message for r in caplog.records)

    async def test_respond__re_raises_cancelled_error(self):
        """Re-raise CancelledError from Ollama without logging."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            pytest.raises(asyncio.CancelledError),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=asyncio.CancelledError())
            mock_client_cls.return_value = mock_client
            await call._respond("hello")

    async def test_send_speech__streams_chunks_to_send_rtp_audio(self):
        """_send_speech iterates generate_audio_stream and passes each chunk to _send_rtp_audio."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        arr1 = np.zeros(160, dtype=np.float32)
        arr2 = np.ones(160, dtype=np.float32)
        chunk1, chunk2 = MagicMock(), MagicMock()
        chunk1.numpy.return_value = arr1
        chunk2.numpy.return_value = arr2
        tts_mock.generate_audio_stream.return_value = iter([chunk1, chunk2])
        tts_mock.sample_rate = 24000

        call = make_agent_call(MagicMock(), tts_mock)
        received: list[np.ndarray] = []
        with patch.object(call, "_send_rtp_audio", side_effect=received.append):
            await call._send_speech("hello")

        assert len(received) == 2
        assert received[0] is arr1
        assert received[1] is arr2

    def test_send_rtp_audio__sends_to_remote_addr(self):
        """_send_rtp_audio sends RTP packets to the caller's registered address."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        tts_mock.sample_rate = 8000  # same as PCMU — no resampling needed
        call = make_agent_call(MagicMock(), tts_mock)

        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        audio = np.zeros(160, dtype=np.float32)
        with patch.object(call, "send_datagram") as mock_send:
            call._send_rtp_audio(audio)
            mock_send.assert_called_once()
        data, addr = mock_send.call_args[0]
        assert addr == remote_addr
        assert len(data) == 12 + 160  # 12-byte RTP header + 160 PCMU bytes

    def test_send_rtp_audio__drops_audio_when_no_remote_addr(self, caplog):
        """Log a warning and drop audio when no RTP address is registered."""
        import logging

        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        tts_mock.sample_rate = 8000
        call = make_agent_call(MagicMock(), tts_mock)
        call.rtp.calls = {}

        with (
            caplog.at_level(logging.WARNING, logger="voip.ai"),
            patch.object(call, "send_datagram") as mock_send,
        ):
            call._send_rtp_audio(np.zeros(160, dtype=np.float32))
        mock_send.assert_not_called()
        assert any("dropping audio" in r.message for r in caplog.records)

    def test_resample__downsamples_from_24khz_to_8khz(self):
        """_resample reduces 24 000 samples at 24 kHz to 8 000 samples at 8 kHz."""
        audio = np.zeros(24000, dtype=np.float32)
        result = AgentCall._resample(audio, 24000)
        assert len(result) == 8000

    def test_resample__passthrough_when_rate_matches(self):
        """_resample returns the original array unchanged when already at 8 kHz."""
        audio = np.zeros(8000, dtype=np.float32)
        result = AgentCall._resample(audio, 8000)
        assert result is audio

    def test_encode_pcmu__returns_one_byte_per_sample(self):
        """_encode_pcmu returns a bytes object with one byte per input sample."""
        samples = np.zeros(160, dtype=np.float32)
        result = AgentCall._encode_pcmu(samples)
        assert isinstance(result, bytes)
        assert len(result) == 160

    def test_encode_pcmu__silence_encodes_to_midpoint(self):
        """Zero-amplitude samples encode to the µ-law midpoint value."""
        samples = np.zeros(10, dtype=np.float32)
        result = AgentCall._encode_pcmu(samples)
        assert result[0] in (127, 128)

    def test_build_rtp_packet__has_twelve_byte_header(self):
        """_build_rtp_packet prepends a 12-byte RTP header to the payload."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        payload = b"\x00" * 160
        packet = call._build_rtp_packet(payload)
        assert len(packet) == 12 + len(payload)
        assert packet[0] == 0x80  # V=2, P=0, X=0, CC=0

    def test_build_rtp_packet__increments_seq_and_ts_each_call(self):
        """Each _build_rtp_packet call increments seq by 1 and ts by chunk size."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 1
        assert call._rtp_ts == 160
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 2
        assert call._rtp_ts == 320
