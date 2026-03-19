# Feature Roadmap

## Implemented

### SIP Signalling

SIP User Agent Client (UAC) over TLS/TCP ([RFC 3261]). Handles incoming
`INVITE`, `BYE`, `ACK`, `CANCEL`, and `OPTIONS` requests, carrier
`REGISTER` with digest authentication ([RFC 8760]: MD5, SHA-256,
SHA-512/256), and double-CRLF keepalive ping/pong ([RFC 5626 §4.4.1]).
Client-initiated keepalive pings, `Supported: outbound` and `;ob` Contact
parameter ([RFC 5626 §5]), and automatic reconnection with exponential
back-off ensure robust long-running sessions.

### Media Transport (RTP/SRTP)

Full RTP packet parsing and per-call multiplexing ([RFC 3550]). SRTP
encryption and authentication with `AES_CM_128_HMAC_SHA1_80` ([RFC 3711]),
with SDES key exchange carried inline in the SDP `a=crypto:` attribute
([RFC 4568]). First-byte STUN/RTP demultiplexing ([RFC 7983]).

### NAT Traversal (STUN)

STUN Binding Request / Response with `XOR-MAPPED-ADDRESS` and `MAPPED-ADDRESS`
for RTP public address discovery ([RFC 5389]). Full IPv4 and IPv6 address
parsing for both attribute types. Uses Cloudflare's STUN server by default;
configurable or disabled per session.

### Session Description (SDP)

Offer / answer model for audio calls. Codec negotiation for Opus ([RFC 7587]),
G.722, PCMU, and PCMA ([RFC 3551]). Full SDP lexer with Pygments syntax
highlighting. IPv6 connection addresses advertised with `IP6` address type
per [RFC 4566 §5.7].

### IPv6

Full dual-stack support across SIP signalling, RTP media, and STUN discovery.
IPv6 addresses in SIP URIs and Via/Contact headers are wrapped in square
brackets per [RFC 2732]. The RTP UDP socket is bound to `::` when the SIP
signalling connection is over IPv6. STUN XOR-MAPPED-ADDRESS and MAPPED-ADDRESS
attributes with IPv6 address family are correctly decoded per [RFC 5389 §15.2].

### Audio Codecs

Inbound decoding and outbound encoding via [PyAV] for all four negotiated
codecs (Opus, G.722, PCMU, PCMA). Audio is resampled to 16 kHz float32 PCM
for downstream processing.

### Speech Transcription

Energy-based voice activity detection (VAD) with configurable silence gap.
Utterances are transcribed in a thread pool via [faster-whisper] —
default model `kyutai/stt-1b-en_fr-trfs`.

### AI Voice Agent

LLM response loop powered by [Ollama], with streaming TTS via [Pocket TTS]
and real-time RTP delivery. Chat history is maintained across turns.
Inbound speech during a response cancels the current reply and hands control
back to the caller.

### CLI

`voip sip <aor> transcribe` — live call transcription to stdout.
`voip sip <aor> agent` — AI voice agent.
SIP message syntax highlighting via a Pygments lexer.

______________________________________________________________________

[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
[ollama]: https://ollama.com/
[pocket tts]: https://github.com/pocket-ai/pocket-tts
[pyav]: https://pyav.org/
[rfc 2732]: https://datatracker.ietf.org/doc/html/rfc2732
[rfc 3261]: https://datatracker.ietf.org/doc/html/rfc3261
[rfc 3550]: https://datatracker.ietf.org/doc/html/rfc3550
[rfc 3551]: https://datatracker.ietf.org/doc/html/rfc3551
[rfc 3711]: https://datatracker.ietf.org/doc/html/rfc3711
[rfc 4566 §5.7]: https://datatracker.ietf.org/doc/html/rfc4566#section-5.7
[rfc 4568]: https://datatracker.ietf.org/doc/html/rfc4568
[rfc 5389]: https://datatracker.ietf.org/doc/html/rfc5389
[rfc 5389 §15.2]: https://datatracker.ietf.org/doc/html/rfc5389#section-15.2
[rfc 7587]: https://datatracker.ietf.org/doc/html/rfc7587
[rfc 7983]: https://datatracker.ietf.org/doc/html/rfc7983
[rfc 8760]: https://datatracker.ietf.org/doc/html/rfc8760
