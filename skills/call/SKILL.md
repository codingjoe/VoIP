---
name: call
description: Call a phone number and hold a two-way AI-driven conversation, returning the full transcript. Use when the user asks to call someone and talk to them, conduct a phone interview, have a conversation with a contact, or have an agent handle an outbound call. Also triggers on "call and talk to", "phone and have a conversation", "interview by phone", "agent call".
when_to_use: The user wants a back-and-forth voice conversation with a phone number. Claude drives the agent side of the call using MCP sampling and returns the transcript once the remote party hangs up.
argument-hint: "<target> [initial-prompt] [system-prompt]"
arguments: target initial_prompt system_prompt
allowed-tools: mcp__VoIP__call
---

# Hold a conversation over the phone

Dial `$target` and drive a two-way conversation. Claude acts as the speaking
agent: it transcribes the remote party's speech, generates replies via MCP
sampling, and speaks them back using text-to-speech. The tool returns once the
remote party hangs up, yielding the full transcript.

## When to use this skill

Use this skill when the user wants a back-and-forth conversation with a phone
number rather than a one-way message. Examples:

- "Call +1234567890 and talk to them about the invoice."
- "Phone the support line and ask about my ticket status."
- "Interview the candidate by phone and send me the transcript."

If the user only wants to deliver a spoken message without replies, use the
`say` skill instead.

## Arguments

- `$target` (required): the phone number or SIP URI to call.
- `$initial_prompt` (optional): opening message spoken when the call connects.
    Pass an empty value to suppress the default greeting.
- `$system_prompt` (optional): system instruction guiding the agent's persona
    and behavior during the call.

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

1. Call the `call` tool exposed by the VoIP MCP server with:
    - `target`: the phone number or SIP URI from `$target`
    - `initial_prompt`: the opening message from `$initial_prompt` (omit or pass
        an empty string to use the default greeting)
    - `system_prompt`: the persona instruction from `$system_prompt` (omit to use
        the agent's default system prompt)
1. The tool dials the target, drives the conversation, and returns the full
    transcript with `Caller:` / `Agent:` prefixes once the remote party hangs up.
1. Summarise or relay the returned transcript to the user. If the tool returns
    an error (for example, the number was unreachable), relay the error clearly.
