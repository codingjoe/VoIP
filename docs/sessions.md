# Multimedia Sessions

[Session][voip.rtp.Session] and its subclasses handle the media exchange between call parties.
They are created by the [Dialog][voip.sip.Dialog] when a call is accepted or initiated.

Sessions can be audio, video, and more. However, this library currently only provides audio sessions via the [AudioCall][voip.audio.AudioCall] class. Video and other media types are fairly uncommon outside of consumer applications, and implementing them is on the roadmap but not yet a priority.

::: voip.rtp.Session

## Audio Handling

::: voip.audio.AudioCall

::: voip.audio.VoiceActivityCall

::: voip.audio.EchoCall

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall

::: voip.ai.SayCall
