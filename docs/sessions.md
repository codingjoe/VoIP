# Multimedia Sessions / Call Leg Handlers

[Session][voip.rtp.Session] is the base class for all call leg handlers.
Each call leg is associated with a [Dialog][voip.sip.dialog.Dialog] that
carries the SIP dialog state and provides the call lifecycle hooks.

## Dialog

The \[`Dialog`\][voip.sip.dialog.Dialog] class manages the SIP dialog state
and is the primary extension point for application logic. Override
\[`call_received`\][voip.sip.dialog.Dialog.call_received] to accept or reject
inbound calls, and \[`hangup_received`\][voip.sip.dialog.Dialog.hangup_received]
to react when the remote party hangs up.

::: voip.sip.dialog.Dialog

## Audio Handling

::: voip.audio.AudioCall

::: voip.audio.VoiceActivityCall

::: voip.audio.EchoCall

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall

::: voip.ai.SayCall
