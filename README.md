<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
    <img alt="Python VoIP" src="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
  </picture>
<br>
  <a href="https://github.com/codingjoe">Documentation</a> |
  <a href="https://github.com/codingjoe/VoIP/issues/new/choose">Issues</a> |
  <a href="https://github.com/codingjoe/VoIP/releases">Changelog</a> |
  <a href="https://github.com/sponsors/codingjoe">Funding</a> 💚
</p>

# Python VoIP library

> [!WARNING]
> This library is in early development and may contain breaking changes. Use with caution.

Python asyncio library for SIP telephony ([RFC 3261](https://tools.ietf.org/html/rfc3261)).

## Setup

```console
pip install voip[audio,cli,pygments]
```

## Usage

### CLI

Answer calls and transcribe them live from the terminal:

```console
voip sip transcribe --server sip.example.com --username alice --password secret
```

### Python API

Subclass `WhisperCall` and override `transcription_received` to handle results.
Pass it as `call_class` when answering an incoming call:

```python
import asyncio
from voip.audio import WhisperCall
from voip.sip.protocol import SIP


class MyCall(WhisperCall):
    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class MySession(SIP):
    def call_received(self, request) -> None:
        self.answer(request=request, call_class=MyCall)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(MySession, local_addr=("0.0.0.0", 5060))
    await asyncio.sleep(3600)


asyncio.run(main())
```

For raw audio access without transcription, subclass `RTP` and override
`audio_received(self, packets: list[bytes])` instead.
