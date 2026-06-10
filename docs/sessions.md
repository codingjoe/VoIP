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

## FAX (T.38)

Implements [RFC 3362] for T.38 fax over SIP/UDPTL.

::: voip.fax.FaxSession

::: voip.fax.OutboundFaxSession

::: voip.fax.InboundFaxSession

[rfc 3362]: https://datatracker.ietf.org/doc/html/rfc3362
