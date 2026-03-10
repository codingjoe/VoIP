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
python3 -m pip install voip[cli,pygments,whisper]
```

## Usage

### Python API

#### Messages

The SIP library provides two classes for SIP messages: `Request` and `Response`.

- `Message.parse`: Parse a SIP message from bytes.
- `__bytes__`: Convert the SIP message to bytes.

```python
>>> from voip.sip.messages import Message
>>> Message.parse(
  b"INVITE sip:bob@biloxi.com SIP/2.0\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n")
Request(method='INVITE', uri='sip:bob@biloxi.com',
        headers={'Via': 'SIP/2.0/UDP pc33.atlanta.com'}, version='SIP/2.0')
>>> Message.parse(b"SIP/2.0 200 OK\r\n\r\n")
Response(status_code=200, reason='OK', headers={}, version='SIP/2.0')
```

#### SIP session and RTP call handler

`SIP` is the session handler. Subclass it and override `call_received` to accept or
reject incoming calls. `RTP` is the call handler — subclass it and override
`audio_received` to process the incoming audio stream.

```python
import asyncio
from voip.sip import SIP
from voip.rtp import RTP


class MyCall(RTP):
    def audio_received(self, data: bytes) -> None:
        print(f"Audio from {self.caller}: {len(data)} bytes")


class MySession(SIP):
    def call_received(self, request) -> None:
        self.answer(request=request, call_class=MyCall)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(MySession, local_addr=("0.0.0.0", 5060))
    await asyncio.sleep(3600)


asyncio.run(main())
```

#### Registering with a SIP carrier

Use `RegisterSIP` to register with a SIP carrier and receive inbound calls:

```python
import asyncio
from voip.sip import RegisterSIP
from voip.rtp import RTP


class MyCall(RTP):
    def audio_received(self, data: bytes) -> None:
        print(f"Audio from {self.caller}: {len(data)} bytes")


class MySession(RegisterSIP):
    def registered(self) -> None:
        print("Registration successful — waiting for calls")

    def call_received(self, request) -> None:
        self.answer(request=request, call_class=MyCall)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        lambda: MySession(
            server_address=("sip.example.com", 5060),
            aor="sip:alice@example.com",
            username="alice",
            password="secret",
        ),
        local_addr=("0.0.0.0", 5060),
    )
    await asyncio.sleep(3600)


asyncio.run(main())
```
