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
  <a href="https://github.com/sponsors/codingjoe">Funding</a> ♥
</p>

# Python VoIP library

> [!WARNING]
> This library is in early development and may contain breaking changes. Use with caution.

Python asyncio library for SIP telephony ([RFC 3261](https://tools.ietf.org/html/rfc3261)).

All signalling uses **SIP over TLS** (SIPS, RFC 3261 §26) and all media is
protected with **SRTP** ([RFC 3711](https://tools.ietf.org/html/rfc3711))
using the `AES_CM_128_HMAC_SHA1_80` cipher suite with SDES key exchange
([RFC 4568](https://tools.ietf.org/html/rfc4568)).

## Setup

```console
pip install voip[audio,cli,pygments]
```

## Usage

### CLI

Answer calls and transcribe them live from the terminal:

```console
voip sip transcribe sips:alice@sip.example.com --password secret
```

### Python API

Subclass `WhisperCall` and override `transcription_received` to handle results.
Pass it as `call_class` when answering an incoming call:

```python
import asyncio
import ssl
from voip.audio import WhisperCall
from voip.sip.protocol import SIP


class MyCall(WhisperCall):
    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class MySession(SIP):
    def call_received(self, request) -> None:
        asyncio.create_task(self.answer(request=request, call_class=MyCall))


async def main():
    loop = asyncio.get_running_loop()
    ssl_context = ssl.create_default_context()
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@example.com",
            username="alice",
            password="secret",
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
