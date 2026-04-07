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

To get started, you will need a SIP account. One is usually included with ISP.
Check your ISP's documentation or router for details.

You will need a SIP AOR (URI), which looks like this:

```INI
sip:USER:PASSWORD@SIP_SERVER;transport=TCP
```

> [!NOTE]
> This library uses secure defaults (TLS transport on port 5061).
> However, most SIP servers only support unencrypted connections.
> Therefore, you will need to provide an explict transport parameter.

### CLI

A simple echo call can be started with:

```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com echo
```

Each command supports an optional `--dial` argument to initiate an
outbound call instead of waiting for an inbound one.

To dial a number, say a message, and hang up automatically:

```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com say sip:+15551234567@sip.example.com "Your package has arrived."
```

You can also talk to a local agent (needs [Ollama]):

```console
uvx 'voip[cli]' sip sips:alice:********@sip.example.com agent --initial-prompt "Hi, I am looking for a Mr. Ron, first name Mo?"
```

### Python API

```console
uv add voip[audio,ai,pygments]
```

Subclass `TranscribeCall` and override `transcription_received` to handle results.
Pass it as `session_class` when answering an incoming call:

```python
import asyncio
import dataclasses
import ssl
from voip.ai import TranscribeCall
from voip.sip.protocol import SIP
from voip.sip.types import SipURI
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
            session_class=TranscribingCall,
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
            aor=SipURI.parse("sips:alice:********@example.com"),
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
