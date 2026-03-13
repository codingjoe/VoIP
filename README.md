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

## Roadmap

### SIP Signaling

| RFC                                                       | Title                                         | Status      | Notes                                                                   |
| --------------------------------------------------------- | --------------------------------------------- | ----------- | ----------------------------------------------------------------------- |
| [RFC 3261](https://datatracker.ietf.org/doc/html/rfc3261) | SIP: Session Initiation Protocol              | 🚧 Partial  | UAC only; REGISTER, INVITE, BYE, and digest authentication over TLS/TCP |
| [RFC 5626](https://datatracker.ietf.org/doc/html/rfc5626) | Managing Client-Initiated Connections in SIP  | 🚧 Partial  | Double-CRLF keepalive ping/pong (§4.4.1)                                |
| [RFC 8760](https://datatracker.ietf.org/doc/html/rfc8760) | SIP Digest Authentication Using AES-HMAC-SHA2 | ✅ Complete | MD5, SHA-256, and SHA-512/256 digest responses                          |
| [RFC 3824](https://datatracker.ietf.org/doc/html/rfc3824) | Using E.164 Numbers with SIP                  | ❌ Planned  | Phone number mapping into SIP/ENUM                                      |
| [RFC 3966](https://datatracker.ietf.org/doc/html/rfc3966) | The tel URI for Telephone Numbers             | ❌ Planned  | Canonical `tel:` URI scheme                                             |
| [RFC 6116](https://datatracker.ietf.org/doc/html/rfc6116) | The E.164 to URI DDDS Application (ENUM)      | ❌ Planned  | DNS-based E.164 number-to-URI mapping                                   |

### Media Transport

| RFC                                                       | Title                                                               | Status      | Notes                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------- | ----------- | -------------------------------------------------------------- |
| [RFC 3550](https://datatracker.ietf.org/doc/html/rfc3550) | RTP: A Transport Protocol for Real-Time Applications                | ✅ Complete | Full RTP packet parsing and per-call multiplexing              |
| [RFC 3551](https://datatracker.ietf.org/doc/html/rfc3551) | RTP Profile for Audio and Video Conferences                         | 🚧 Partial  | PCMU (0), PCMA (8), G.722 (9), and Opus (111) payload types    |
| [RFC 3711](https://datatracker.ietf.org/doc/html/rfc3711) | Secure Real-time Transport Protocol (SRTP)                          | ✅ Complete | AES-CM-128-HMAC-SHA1-80 encryption and authentication          |
| [RFC 4566](https://datatracker.ietf.org/doc/html/rfc4566) | SDP: Session Description Protocol                                   | 🚧 Partial  | Offer/answer model for audio calls; connection and media lines |
| [RFC 4568](https://datatracker.ietf.org/doc/html/rfc4568) | SDP Security Descriptions for Media Streams (SDES)                  | ✅ Complete | Inline SRTP key exchange via `a=crypto:`                       |
| [RFC 5389](https://datatracker.ietf.org/doc/html/rfc5389) | STUN: Session Traversal Utilities for NAT                           | ✅ Complete | Binding Request/Response with XOR-MAPPED-ADDRESS               |
| [RFC 7983](https://datatracker.ietf.org/doc/html/rfc7983) | Multiplexing Scheme Updates for SRTP Extension for DTLS             | ✅ Complete | First-byte demultiplexing of STUN vs. RTP/SRTP                 |
| [RFC 7587](https://datatracker.ietf.org/doc/html/rfc7587) | RTP Payload Format for the Opus Speech and Audio Codec              | ✅ Complete | Dynamic payload type 111                                       |
| [RFC 3533](https://datatracker.ietf.org/doc/html/rfc3533) | The Ogg Encapsulation Format Version 0                              | 🚧 Partial  | Minimal Ogg page writer for Opus audio export                  |
| [RFC 4733](https://datatracker.ietf.org/doc/html/rfc4733) | RTP Payload for DTMF Digits, Telephony Tones, and Telephony Signals | ❌ Planned  | In-band DTMF over RTP                                          |

### IVR and Application Services

| RFC                                                       | Title                                                       | Status     | Notes                                                          |
| --------------------------------------------------------- | ----------------------------------------------------------- | ---------- | -------------------------------------------------------------- |
| [RFC 6230](https://datatracker.ietf.org/doc/html/rfc6230) | Media Control Channel Framework                             | ❌ Planned | SIP-based control of external media servers                    |
| [RFC 6231](https://datatracker.ietf.org/doc/html/rfc6231) | IVR Control Package for the Media Control Channel Framework | ❌ Planned | Interactive voice response over the media control channel      |
| [RFC 4458](https://datatracker.ietf.org/doc/html/rfc4458) | SIP URIs for Applications such as Voicemail and IVR         | ❌ Planned | Standardized SIP URI parameters for voicemail and IVR services |
| [RFC 3880](https://datatracker.ietf.org/doc/html/rfc3880) | Call Processing Language (CPL)                              | ❌ Planned | XML language for describing call-handling logic                |

### Voicemail

| RFC                                                       | Title                                                | Status     | Notes                                                  |
| --------------------------------------------------------- | ---------------------------------------------------- | ---------- | ------------------------------------------------------ |
| [RFC 3801](https://datatracker.ietf.org/doc/html/rfc3801) | Voice Profile for Internet Mail – version 2 (VPIMv2) | ❌ Planned | Voice mail exchange between servers over Internet mail |
| [RFC 4239](https://datatracker.ietf.org/doc/html/rfc4239) | Internet Voice Messaging (IVM)                       | ❌ Planned | Standardized Internet voice message format             |

### Telephony Routing

| RFC                                                       | Title                                                            | Status     | Notes                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------- | ---------- | ---------------------------------------------- |
| [RFC 2871](https://datatracker.ietf.org/doc/html/rfc2871) | A Framework for Telephony Routing over IP                        | ❌ Planned | Architectural framework for TRIP               |
| [RFC 3219](https://datatracker.ietf.org/doc/html/rfc3219) | Telephony Routing over IP (TRIP)                                 | ❌ Planned | Inter-domain routing of telephony destinations |
| [RFC 5115](https://datatracker.ietf.org/doc/html/rfc5115) | Telephony Routing over IP (TRIP) Attribute for Resource Priority | ❌ Planned | Priority service support over TRIP             |
