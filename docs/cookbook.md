# Cookbook

To build anything we need to understand two fundamental concepts: **[Dialogs][voip.sip.Dialog] and [Sessions](sessions.md).**

Dial, accept, reject, hold, transfer, etc. are part of a **dialog** between you and the remote party.

The actual audio or video exchange happens in a multimedia **session**. A session is established by a dialog.

## Call Transcription

Subclass [TranscribeCall][voip.ai.TranscribeCall] and override
[transcription_received][voip.ai.TranscribeCall.transcription_received] to
handle each utterance after a silence gap:

```python
import asyncio
import ssl

from voip.ai import TranscribeCall
from voip.sip.dialog import Dialog
from voip.sip.protocol import SIP


class PrintTranscribeCall(TranscribeCall):
    """Print the transcription to the console."""

    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class AutoAcceptDialog(Dialog):
    """Accept every incoming call and transcribe it using MyCall."""

    def call_received(self) -> None:
        self.ringing()
        self.answer(session_class=PrintTranscribeCall)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_connection(
        lambda: SIP(
            aor="sips:alice@example.com",
            dialog_class=AutoAcceptDialog,
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl.create_default_context(),
    )
    await asyncio.Future()


asyncio.run(main())
```

## Sharing a Whisper Model Across Calls

Loading the model is expensive. Pass a preloaded
[WhisperModel](https://github.com/SYSTRAN/faster-whisper) instance as a
class attribute to share it across all incoming calls:

```python
from faster_whisper import WhisperModel
from voip.ai import TranscribeCall


shared_model = WhisperModel("kyutai/stt-1b-en_fr-trfs")


class MyCall(TranscribeCall):
    model = shared_model
```

## AI Call Agent

[AgentCall][voip.ai.AgentCall] extends transcription with an
[Ollama](https://ollama.com/) LLM response loop and
[Pocket TTS](https://github.com/pocket-ai/pocket-tts) voice synthesis.
Share both heavy models across calls to avoid reloading them per call:

```python
import asyncio
import ssl

from pocket_tts import TTSModel

from voip.ai import AgentCall
from voip.sip.dialog import Dialog
from voip.sip.protocol import SIP

shared_tts = TTSModel.load_model()


class MyCall(AgentCall):
    tts_model = shared_tts
    system_prompt = "You are a friendly hotel receptionist. Keep answers brief."
    llm_model = "llama3"
    voice = "azelma"


class MyDialog(Dialog):
    def call_received(self) -> None:
        self.ringing()
        self.answer(session_class=MyCall)


class MySession(SIP):
    dialog_class = MyDialog


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

Subclass [AudioCall][voip.audio.AudioCall] and override
[audio_received][voip.audio.AudioCall.audio_received] to receive decoded
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

Use [\_send_rtp_audio][voip.audio.AudioCall.\_send_rtp_audio] inside any
[AudioCall][voip.audio.AudioCall] subclass to stream float32 PCM back to
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

For protocols other than audio, subclass [Session][voip.rtp.Session]
directly and override [packet_received]\[voip.rtp.Session.packet_received\]:

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

## Hanging Up a Call

Every [Session][voip.rtp.Session] subclass exposes a
[hang_up][voip.rtp.Session.hang_up] coroutine that sends a proper SIP BYE
request (RFC 3261 §15) by delegating to
[Dialog.bye][voip.sip.Dialog.bye]. It deregisters the RTP
handler and awaits the 200 OK acknowledgment before returning.

Override [Dialog.call_received][voip.sip.Dialog.call_received]
to hook into the call lifecycle, and call `await self.hang_up()` from within
the call class when you want to terminate:

```python
import asyncio
import ssl

import numpy as np

from voip.audio import AudioCall
from voip.sip.dialog import Dialog
from voip.sip.protocol import SIP


class OneUtteranceCall(AudioCall):
    """Hang up as soon as the first voice utterance is received."""

    async def voice_received(self, audio: np.ndarray) -> None:
        await self.hang_up()
        # dialog.sip.close() to also shut down the SIP transport:
        if self.dialog and self.dialog.sip:
            self.dialog.sip.close()


class MyDialog(Dialog):
    def call_received(self) -> None:
        self.ringing()
        self.answer(session_class=OneUtteranceCall)


class MySession(SIP):
    dialog_class = MyDialog


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

[hang_up][voip.rtp.Session.hang_up] sends the BYE and cleans up the dialog
and RTP handler — it does **not** close the SIP transport so that the same
[SIP][voip.sip.protocol.SessionInitiationProtocol] instance can continue
handling other calls. Access `self.dialog.sip.close()` when you also want to
tear down the transport.

## Making Outbound Calls

Create a [Dialog][voip.sip.Dialog] subclass, set it as
`dialog_class` on your SIP session, and call
[dial][voip.sip.Dialog.dial] from
[on_registered]\[voip.sip.protocol.SessionInitiationProtocol.on_registered\]:

```python
import asyncio
import ssl

from voip.audio import AudioCall
from voip.sip.dialog import Dialog
from voip.sip.protocol import SIP


class MyCall(AudioCall):
    pass


class OutboundDialog(Dialog):
    def hangup_received(self) -> None:
        """Remote party hung up — close the SIP transport."""
        if self.sip:
            self.sip.close()


class MySession(SIP):
    dialog_class = OutboundDialog

    def on_registered(self) -> None:
        dialog = OutboundDialog(sip=self)
        asyncio.create_task(
            dialog.dial("sip:+15551234567@carrier.com", session_class=MyCall)
        )


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@carrier.com",
            username="alice",
            password="secret",
        ),
        host="sip.carrier.com",
        port=5061,
        ssl=ssl.create_default_context(),
    )
    await asyncio.Future()


asyncio.run(main())
```
