---
name: transcribe
description: Start a phone line that transcribes incoming call audio to text. Use when the user asks to transcribe a phone call, get a live transcript of a call, capture speech-to-text from a phone number, or convert call audio to text. Also triggers on "transcribe the call", "live transcript of incoming calls", "speech-to-text for phone".
when_to_use: The user wants to receive calls and transcribe the caller's speech to text. Runs the VoIP CLI transcribe command, which waits for an inbound call (or dials out with --dial) and prints each transcribed utterance.
argument-hint: "[--dial TARGET] [--stt-model MODEL]"
allowed-tools: Bash(uvx*) Bash(voip sip transcribe*)
---

# Transcribe call audio to text

Start a phone line that transcribes incoming call audio to text using a local
Whisper model. Each transcribed utterance is printed to stdout. Use this skill
when the user wants a live speech-to-text feed from a phone number.

## When to use this skill

Use this skill when the user wants to transcribe phone call audio. Examples:

- "Transcribe my incoming calls."
- "Get a live transcript of the next call to my SIP number."
- "Dial +1234567890 and transcribe what they say."

If the user wants a two-way AI conversation, use the `call` skill instead.

## Prerequisites

The user needs a SIP account. The SIP address-of-record (AOR) looks like:

```
sip:USER:PASSWORD@SIP_SERVER
```

The VoIP CLI is distributed as the `voip[cli]` Python extra and runs via `uvx`.
The first run downloads the model and may take longer.

## Commands

Wait for an inbound call and transcribe it:

```console
uvx 'voip[cli]' sip sip:USER:PASSWORD@SIP_SERVER transcribe
```

Dial out and transcribe the remote party:

```console
uvx 'voip[cli]' sip sip:USER:PASSWORD@SIP_SERVER transcribe --dial sip:target@example.com
```

Common options:

- `--stt-model`: Whisper model size (default `tiny`). Larger models are more
    accurate but slower. Common values: `tiny`, `base`, `small`, `medium`.
- `--dial TARGET`: dial a target instead of waiting for an inbound call.
- `--stun-server HOST[:PORT]`: STUN server for RTP NAT traversal
    (default `stun.cloudflare.com:3478`).
- `--no-verify-tls`: disable TLS certificate verification (testing only).

## Instructions

1. Ask the user for their SIP AOR if it was not already provided. Never invent
    credentials.
1. Confirm whether the user wants to wait for an inbound call or dial out. If
    dialing out, get the target number or SIP URI.
1. Run the `transcribe` CLI command shown above with the user's AOR and any
    requested options. Stream the transcribed utterances back to the user.
1. Stop the command (Ctrl-C / send a termination signal) when the user is done.
