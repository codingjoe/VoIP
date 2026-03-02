# Python SIP - Session Initiation Protocol (SIP)

Python asyncio library for SIP telephony ([RFC 3261](https://tools.ietf.org/html/rfc3261)).

## Setup

```bash
python3 -m pip install libsip  # lightweight, without any dependencies
# or
python3 -m pip install libsip[pygments]  # with Pygments syntax highlighting support
```

## Usage

### Python API

#### Messages

The SIP library provides two classes for SIP messages: `Request` and `Response`.

- `Message.parse`: Parse a SIP message from bytes.
- `__bytes__`: Convert the SIP message to bytes.

```python
>>> from sip import Message
>>> Message.parse(b"INVITE sip:bob@biloxi.com SIP/2.0\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n")
Request(method='INVITE', uri='sip:bob@biloxi.com', headers={'Via': 'SIP/2.0/UDP pc33.atlanta.com'}, version='SIP/2.0')
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
import sip


class MyProtocol(sip.SIP):
    def request_received(self, request: sip.Request, addr: tuple[str, int]) -> None:
        print(request, addr)

    def response_received(self, response: sip.Response, addr: tuple[str, int]) -> None:
        print(response, addr)


async def main():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(MyProtocol, local_addr=("0.0.0.0", 5060))
    await asyncio.sleep(3600)


asyncio.run(main())
```

## SIP lexer plugin for [Pygments](https://pygments.org/)

The SIP library comes with a lexer plugin for [Pygments](https://pygments.org/) to
highlight SIP messages. It's based on the HTTP lexer and adds SIP-specific keywords.

Install the plugin with:

```bash
pip install libsip[pygments]
```

You can get the lexer by name:

```python
>>> from pygments.lexers import get_lexer_by_name
>>> get_lexer_by_name("sip")
<sip.lexers.SIPLexer>
```

Highlighting a SIP message could look like this:

```python
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name

if __name__ == "__main__":
    lexer = get_lexer_by_name("sip")
    formatter = TerminalFormatter()
    code = "INVITE sip:bob@biloxi.com SIP/2.0\r\nVia: SIP/2.0/UDP pc33.atlanta.com"
    print(highlight(code, lexer, formatter))
```
