# Call Handling

The `voip.call` module provides the base classes for handling individual call legs.

## Call

The base `Call` class handles the association between a SIP dialog and the RTP media stream.

::: voip.call.Call

## Audio Handling

::: voip.audio.AudioCall

### Jitter Buffer

The `JitterBuffer` smooths out network-induced packet arrival jitter by holding
incoming RTP packets for a configurable playout delay before releasing them in
order at the codec frame rate (20 ms for G.722, PCMA, PCMU, and Opus).

::: voip.audio.JitterBuffer

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall
