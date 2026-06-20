# Multimedia Sessions

[Session][voip.rtp.Session] and its subclasses handle the media exchange between call parties.
They are created by the [Dialog][voip.sip.Dialog] when a call is accepted or initiated.

Sessions can be audio, video, and more. This library provides audio sessions via
[AudioCall][voip.audio.AudioCall] and T.38 FAX sessions via [FaxSession][voip.fax.FaxSession].

::: voip.rtp.Session

## Audio Handling

::: voip.audio.AudioCall

::: voip.audio.VoiceActivityCall

::: voip.audio.EchoCall

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall

::: voip.ai.SayCall

## FAX

This library supports two fax transport methods:

- **T.38 over UDPTL** ([RFC 3362]) — efficient and reliable when both sides
    support it, but many providers (e.g. sipgate) do not.
- **G.711 pass-through** — fax modem tones sent as regular audio (PCMU).
    Works with any SIP provider that supports voice calls, at the cost of
    higher bandwidth and no packet-loss redundancy.
- **Dual offer** ([DualFaxSession][voip.fax.DualFaxSession]) — offers both
    T.38 and G.711 in a single SDP so the remote endpoint picks whichever it
    supports.

### T.38 (UDPTL)

Implements [RFC 3362] for T.38 fax over SIP/UDPTL.

::: voip.fax.FaxSession

::: voip.fax.OutboundFaxSession

::: voip.fax.InboundFaxSession

### G.711 pass-through

G.711 pass-through sends fax modem tones as regular PCMU audio over RTP.
RTP frames are paced at 20 ms intervals to match real-time delivery.

When the document is a PDF file, the [T30Modem][voip.t30.modem.T30Modem]
renders the PDF pages into bitonal images, T.4-compresses them, and
modulates the full T.30 exchange (CNG, DCS, V.29 training, image
data, EOP, DCN) into audio tones that the remote fax machine can
demodulate.

The outbound path uses a **phased transmit** approach: each phase
(CNG, DCS, V.29 training + TCF, image + EOP, DCN) is sent as real-time
audio, and the [T30Receiver][voip.t30.modem.T30Receiver] demodulates the
called machine's responses (CED, DIS, CFR, MCF, DCN) to time each phase
to the remote fax machine's actual handshake — rather than fixed delays.

Install the `fax` extra (`pymupdf`) to enable PDF rendering:

```console
uv pip install voip[fax]
```

Non-PDF bytes are interpreted as raw µ-law audio and sent directly.

::: voip.fax.G711FaxSession

::: voip.fax.OutboundG711FaxSession

::: voip.fax.InboundG711FaxSession

### Dual offer (T.38 + G.711)

::: voip.fax.DualFaxSession

::: voip.fax.OutboundDualFaxSession

::: voip.fax.InboundDualFaxSession

[rfc 3362]: https://datatracker.ietf.org/doc/html/rfc3362
