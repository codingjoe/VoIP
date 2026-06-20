---
name: say
description: Call a phone number and speak a one-way message, then hang up. Use when the user asks to call someone, deliver a voice message, announce something by phone, or send a spoken notification to a phone number. Also triggers on "phone someone", "call and tell", "read this over the phone".
when_to_use: "The user wants to deliver a spoken message to a phone number without holding a conversation. One-way delivery: the message is spoken and the call ends automatically."
argument-hint: "<target> <message>"
arguments: target message
allowed-tools: mcp__VoIP__say
---

# Say a message over the phone

Dial `$target` and speak `$message` using text-to-speech, then hang up once the
message has been delivered. This is a one-way message: the recipient is not
expected to reply and the call ends automatically after the speech finishes.

## When to use this skill

Use this skill when the user wants to deliver a spoken message to a phone
number without holding a two-way conversation. Examples:

- "Call +1234567890 and tell them the meeting is moved to 3pm."
- "Phone Alice and say the server is back up."
- "Read this reminder to my dad: …"

If the user wants a back-and-forth conversation, use the `call` skill instead.

## Target format

`$target` is a phone number or SIP URI:

- `tel:+1234567890`
- `sip:alice@example.com`
- `+1234567890`

## Prerequisites

The VoIP MCP server (bundled with this plugin) must be configured with a valid
`SIP_AOR` environment variable pointing to your SIP address-of-record. The
server connects to your SIP carrier automatically when Claude Code starts it.

## Instructions

1. Call the `say` tool exposed by the VoIP MCP server with:
    - `target`: the phone number or SIP URI from `$target`
    - `prompt`: the message to speak from `$message`
1. The tool dials the target, synthesises the prompt as speech, and hangs up
    automatically once delivery completes.
1. Report to the user whether the message was delivered. If the tool returns an
    error (for example, the number was unreachable), relay the error clearly.
