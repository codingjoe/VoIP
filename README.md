<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
    <img alt="Python VoIP" src="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
  </picture>
<br>
  <a href="https://codingjoe.dev/VoIP">Documentation</a> |
  <a href="https://github.com/codingjoe/VoIP/issues/new/choose">Issues</a> |
  <a href="https://github.com/codingjoe/VoIP/releases">Changelog</a> |
  <a href="https://github.com/sponsors/codingjoe">Funding</a> ♥
</p>

# Python VoIP

Async VoIP Python library for the AI age.

> [!WARNING]
> This library is in early development and may contain breaking changes. Use with caution.

## Usage

### CLI

Answer calls and transcribe them live from the terminal:

```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com transcribe
```

A simple echo server can be started with:

````console
```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com echo
````

You can also talk to a local agent (needs [Ollama]):

```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com agent
```

### Python API

```console
uv add voip[audio,ai,pygments]
```

Subclass `TranscribeCall` and override `transcription_received` to handle results.
Pass it as `call_class` when answering an incoming call:

```python
import asyncio
import dataclasses
import ssl
from voip.ai import TranscribeCall
from voip.sip.protocol import SIP
from voip.sip.types import SipUri
from voip.sip.transactions import InviteTransaction
from voip.rtp import RealtimeTransportProtocol
from faster_whisper import WhisperModel


@dataclasses.dataclass(kw_only=True, slots=True)
class TranscribingCall(TranscribeCall):
    def transcription_received(self, text) -> None:
        print(text)


class TranscribeInviteTransaction(InviteTransaction):
    def invite_received(self, request) -> None:
        self.ringing()
        self.answer(
            call_class=TranscribingCall,
            stt_model=WhisperModel("kyutai/stt-1b-en_fr-trfs", device="cuda"),
        )


async def main():
    loop = asyncio.get_running_loop()
    _, rtp_protocol = await loop.create_datagram_endpoint(
        RealtimeTransportProtocol,
        local_addr=("0.0.0.0", 0),
    )
    ssl_context = ssl.create_default_context()
    await loop.create_connection(
        lambda: SIP(
            rtp=rtp_protocol,
            aor=SipUri.parse("sips:alice:********@example.com"),
            transaction_class=TranscribeInviteTransaction,
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl_context,
    )
    await asyncio.Future()


asyncio.run(main())
```

For raw audio access without transcription, subclass `AudioCall` and override
`audio_received(self, audio: np.ndarray)` instead.

[ollama]: https://ollama.com/
