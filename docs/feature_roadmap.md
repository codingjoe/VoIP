# Feature Roadmap

## Implemented

### SIP Signalling

SIP User Agent Client (UAC) over TLS/TCP ([RFC 3261]). Handles incoming
`INVITE`, `BYE`, `ACK`, `CANCEL`, and `OPTIONS` requests, carrier
`REGISTER` with digest authentication ([RFC 8760]: MD5, SHA-256,
SHA-512/256), and double-CRLF keepalive ping/pong ([RFC 5626 §4.4.1]).

### Media Transport (RTP/SRTP)

Full RTP packet parsing and per-call multiplexing ([RFC 3550]). SRTP
encryption and authentication with `AES_CM_128_HMAC_SHA1_80` ([RFC 3711]),
with SDES key exchange carried inline in the SDP `a=crypto:` attribute
([RFC 4568]). First-byte STUN/RTP demultiplexing ([RFC 7983]).

### NAT Traversal (STUN)

STUN Binding Request / Response with `XOR-MAPPED-ADDRESS` for RTP public
address discovery ([RFC 5389]). Uses Cloudflare's STUN server by default;
configurable or disabled per session.

### Session Description (SDP)

Offer / answer model for audio calls. Codec negotiation for Opus ([RFC 7587]),
G.722, PCMU, and PCMA ([RFC 3551]). Full SDP lexer with Pygments syntax
highlighting.

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

## Planned

### DTMF (RFC 4733)

In-band DTMF digits, telephony tones, and telephony signals over RTP.

### E.164 / tel: URI

- [RFC 3966] — The `tel:` URI scheme for telephone numbers.
- [RFC 3824] — Using E.164 numbers with SIP.
- [RFC 6116] — ENUM: DNS-based E.164-to-URI mapping.

### IVR & Media Control

- [RFC 6230] — Media Control Channel Framework.
- [RFC 6231] — IVR Control Package for the Media Control Channel.
- [RFC 4458] — SIP URIs for voicemail and IVR applications.
- [RFC 3880] — Call Processing Language (CPL).

### Voicemail

- [RFC 3801] — Voice Profile for Internet Mail v2 (VPIMv2).
- [RFC 4239] — Internet Voice Messaging (IVM).

### Telephony Routing (TRIP)

- [RFC 2871] — Framework for Telephony Routing over IP.
- [RFC 3219] — Telephony Routing over IP (TRIP).
- [RFC 5115] — TRIP Attribute for Resource Priority.

______________________________________________________________________

[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
[ollama]: https://ollama.com/
[pocket tts]: https://github.com/pocket-ai/pocket-tts
[pyav]: https://pyav.org/
[rfc 2871]: https://datatracker.ietf.org/doc/html/rfc2871
[rfc 3219]: https://datatracker.ietf.org/doc/html/rfc3219
[rfc 3261]: https://datatracker.ietf.org/doc/html/rfc3261
[rfc 3550]: https://datatracker.ietf.org/doc/html/rfc3550
[rfc 3551]: https://datatracker.ietf.org/doc/html/rfc3551
[rfc 3711]: https://datatracker.ietf.org/doc/html/rfc3711
[rfc 3801]: https://datatracker.ietf.org/doc/html/rfc3801
[rfc 3824]: https://datatracker.ietf.org/doc/html/rfc3824
[rfc 3880]: https://datatracker.ietf.org/doc/html/rfc3880
[rfc 3966]: https://datatracker.ietf.org/doc/html/rfc3966
[rfc 4239]: https://datatracker.ietf.org/doc/html/rfc4239
[rfc 4458]: https://datatracker.ietf.org/doc/html/rfc4458
[rfc 4568]: https://datatracker.ietf.org/doc/html/rfc4568
[rfc 5115]: https://datatracker.ietf.org/doc/html/rfc5115
[rfc 5389]: https://datatracker.ietf.org/doc/html/rfc5389
[rfc 6116]: https://datatracker.ietf.org/doc/html/rfc6116
[rfc 6230]: https://datatracker.ietf.org/doc/html/rfc6230
[rfc 6231]: https://datatracker.ietf.org/doc/html/rfc6231
[rfc 7587]: https://datatracker.ietf.org/doc/html/rfc7587
[rfc 7983]: https://datatracker.ietf.org/doc/html/rfc7983
[rfc 8760]: https://datatracker.ietf.org/doc/html/rfc8760
