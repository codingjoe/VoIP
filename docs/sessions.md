# Multimedia Sessions / Call Leg Handlers

[Session][voip.rtp.Session] is the base class for all call leg handlers.
Each call leg is associated with a [Dialog][voip.sip.messages.Dialog] that
carries the SIP dialog state and provides the call lifecycle hooks.

## Dialog

The \[`Dialog`\][voip.sip.messages.Dialog] class manages the SIP dialog state
and is the primary extension point for application logic. Override
\[`call_received`\][voip.sip.messages.Dialog.call_received] to accept or reject
inbound calls, and \[`hangup_received`\][voip.sip.messages.Dialog.hangup_received]
to react when the remote party hangs up.

::: voip.sip.messages.Dialog

## Base Session

::: voip.rtp.Session

## Audio Handling

::: voip.audio.AudioCall

::: voip.audio.VoiceActivityCall

::: voip.audio.EchoCall

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall

::: voip.ai.SayCall
