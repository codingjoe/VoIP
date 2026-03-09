<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
    <img alt="Django HealthCheck: Pluggable health checks for Django applications" src="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
  </picture>
<br>
  <a href="https://codingjoe.dev/VoIP/">Documentation</a> |
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

#### Asyncio SIP Protocol datagram endpoint

`SessionInitiationProtocol` (aliased as `SIP`) is a subclass of `asyncio.DatagramProtocol` that dispatches received UDP
datagrams to the appropriate handler:

- `request_received`: Called when a SIP request is received.
- `response_received`: Called when a SIP response is received.

```python
import asyncio
import voip


class MyProtocol(voip.SIP):
    def request_received(self, request: voip.Request, addr: tuple[str, int]) -> None:
        print(request, addr)

    def response_received(self, response: voip.Response, addr: tuple[str, int]) -> None:
        print(response, addr)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(MyProtocol, local_addr=("0.0.0.0", 5060))
    await asyncio.sleep(3600)


asyncio.run(main())
```
