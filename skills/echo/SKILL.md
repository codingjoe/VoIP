---
name: echo
description: Start a phone line that echoes the caller's speech back to them after they finish speaking. Use when the user asks to set up an echo call, test a phone line by replaying speech, or mirror a caller's voice. Also triggers on "echo call", "echo my speech back", "test the phone line".
when_to_use: The user wants to receive calls and replay the caller's speech back to them. Runs the VoIP CLI echo command, which waits for an inbound call (or dials out with --dial) and echoes each utterance back.
argument-hint: "[--dial TARGET]"
allowed-tools: Bash(uvx*) Bash(voip sip echo*)
---

# Echo the caller's speech back

Start a phone line that records the caller's speech and plays it back to them
once they finish speaking. Use this skill for quick line testing or to verify
audio is flowing correctly in both directions.

## When to use this skill

Use this skill when the user wants to echo speech back over a phone line.
Examples:

- "Set up an echo call on my SIP number."
- "Echo my speech back to test the phone line."
- "Dial +1234567890 and echo what they say."

If the user wants a conversation or transcription, use the `call` or `transcribe`
skills instead.

## Prerequisites

The user needs a SIP account. The SIP address-of-record (AOR) looks like:

```
sip:USER:PASSWORD@SIP_SERVER
```

The VoIP CLI is distributed as the `voip[cli]` Python extra and runs via `uvx`.

## Commands

Wait for an inbound call and echo it back:

```console
uvx 'voip[cli]' sip sip:USER:PASSWORD@SIP_SERVER echo
```

Dial out and echo the remote party's speech back:

```console
uvx 'voip[cli]' sip sip:USER:PASSWORD@SIP_SERVER echo --dial sip:target@example.com
```

Common options:

- `--dial TARGET`: dial a target instead of waiting for an inbound call.
- `--stun-server HOST[:PORT]`: STUN server for RTP NAT traversal
    (default `stun.cloudflare.com:3478`).
- `--no-verify-tls`: disable TLS certificate verification (testing only).

## Instructions

1. Ask the user for their SIP AOR if it was not already provided. Never invent
    credentials.
1. Confirm whether the user wants to wait for an inbound call or dial out. If
    dialing out, get the target number or SIP URI.
1. Run the `echo` CLI command shown above with the user's AOR and any requested
    options.
1. Stop the command (Ctrl-C / send a termination signal) when the user is done.
