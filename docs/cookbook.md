# Cookbook

## Call Transcription

Subclass \[`TranscribeCall`\][voip.ai.TranscribeCall] and override
\[`transcription_received`\][voip.ai.TranscribeCall.transcription_received] to
handle each utterance after a silence gap:

```python
import asyncio
import ssl

from voip.ai import TranscribeCall
from voip.sip.protocol import SIP


class MyCall(TranscribeCall):
    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class MySession(SIP):
    def call_received(self, request) -> None:
        asyncio.create_task(self.answer(request=request, call_class=MyCall))


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@example.com",
            username="alice",
            password="secret",
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl.create_default_context(),
    )
    await asyncio.Future()


asyncio.run(main())
```

## Sharing a Whisper Model Across Calls

Loading the model is expensive. Pass a pre-loaded
[`WhisperModel`](https://github.com/SYSTRAN/faster-whisper) instance as a
class attribute to share it across all incoming calls:

```python
from faster_whisper import WhisperModel
from voip.ai import TranscribeCall


shared_model = WhisperModel("kyutai/stt-1b-en_fr-trfs")


class MyCall(TranscribeCall):
    model = shared_model
```

## AI Call Agent

\[`AgentCall`\][voip.ai.AgentCall] extends transcription with an
[Ollama](https://ollama.com/) LLM response loop and
[Pocket TTS](https://github.com/pocket-ai/pocket-tts) voice synthesis.
Share both heavy models across calls to avoid reloading them per call:

```python
import asyncio
import ssl

from pocket_tts import TTSModel

from voip.ai import AgentCall
from voip.sip.protocol import SIP

shared_tts = TTSModel.load_model()


class MyCall(AgentCall):
    tts_model = shared_tts
    system_prompt = "You are a friendly hotel receptionist. Keep answers brief."
    llm_model = "llama3"
    voice = "azelma"


class MySession(SIP):
    def call_received(self, request) -> None:
        asyncio.create_task(self.answer(request=request, call_class=MyCall))


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@example.com", username="alice", password="secret"
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl.create_default_context(),
    )
    await asyncio.Future()


asyncio.run(main())
```

## Raw Audio Access

Subclass \[`AudioCall`\][voip.audio.AudioCall] and override
\[`audio_received`\][voip.audio.AudioCall.audio_received] to receive decoded
float32 PCM frames without transcription:

```python
import numpy as np
from voip.audio import AudioCall, SAMPLE_RATE


class RecordCall(AudioCall):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._frames: list[np.ndarray] = []

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        self._frames.append(audio)

    def save(self, path: str) -> None:
        import soundfile as sf

        sf.write(path, np.concatenate(self._frames), SAMPLE_RATE)
```

## Sending Audio to the Caller

Use \[`_send_rtp_audio`\][voip.audio.AudioCall.\_send_rtp_audio] inside any
\[`AudioCall`\][voip.audio.AudioCall] subclass to stream float32 PCM back to
the caller using the negotiated codec:

```python
import asyncio
import numpy as np
import soundfile as sf
from voip.audio import AudioCall, SAMPLE_RATE


class GreetingCall(AudioCall):
    async def play_greeting(self) -> None:
        audio, file_rate = sf.read("greeting.wav", dtype="float32", always_2d=False)
        resampled = self._resample(audio, file_rate, SAMPLE_RATE)
        await self._send_rtp_audio(resampled)
```

## Low-Level RTP Packet Handling

For protocols other than audio, subclass \[`Session`\][voip.rtp.Session]
directly and override \[`packet_received`\]\[voip.rtp.Session.packet_received\]:

```python
from voip.rtp import Session, RTPPacket


class EchoCall(Session):
    def packet_received(self, packet: RTPPacket, addr: tuple[str, int]) -> None:
        # Echo every packet straight back to the sender.
        self.send_packet(packet, addr)
```

## Outbound Proxy

Some SIP carriers use a dedicated proxy address that differs from the AOR
domain. Pass `outbound_proxy` to route all signalling through it:

```python
from voip.sip.protocol import SIP

session = SIP(
    aor="sips:alice@carrier.com",
    username="alice",
    password="secret",
    outbound_proxy=("proxy.carrier.com", 5061),
)
```

## Disabling STUN

When the application runs on a public-facing host with no NAT, skip the STUN
discovery round-trip by setting `rtp_stun_server_address=None`:

```python
from voip.sip.protocol import SIP

session = SIP(
    aor="sips:alice@example.com",
    username="alice",
    password="secret",
    rtp_stun_server_address=None,
)
```
