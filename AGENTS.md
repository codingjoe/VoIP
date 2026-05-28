# Agents

> [!NOTE]
> Keep this file up to date when adding new modules, changing the project structure, or updating development workflows.

Python VoIP is an async VoIP library for Python. It implements SIP (signalling), SDP (session description), RTP (media transport), SRTP (encrypted media), STUN, and codec handling, with optional AI/transcription and CLI layers on top.

## Repository Structure

| Path               | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| `voip/sip/`        | SIP protocol: messages, transactions, dialog, types, and Pygments lexer |
| `voip/sdp/`        | SDP parsing and generation                                              |
| `voip/rtp.py`      | RTP media transport                                                     |
| `voip/srtp.py`     | SRTP (encrypted RTP)                                                    |
| `voip/stun.py`     | STUN client                                                             |
| `voip/types.py`    | Shared types and parsing helpers                                        |
| `voip/codecs/`     | Audio codec implementations (PCMU, PCMA, G.722, Opus, AV bridge)        |
| `voip/audio.py`    | Audio I/O helpers                                                       |
| `voip/ai.py`       | AI/transcription call helpers                                           |
| `voip/mcp.py`      | MCP (Model Context Protocol) integration                                |
| `voip/__main__.py` | CLI entry point                                                         |
| `docs/`            | MkDocs documentation source                                             |
| `tests/`           | Pytest test suite mirroring the `voip/` package layout                  |

When writing code, you MUST ALWAYS follow the [naming-things](https://github.com/codingjoe/naming-things/blob/main/README.md) guidelines.

<!--
Bots/Agents MUST use:
curl -sSL https://raw.githubusercontent.com/codingjoe/naming-things/refs/heads/main/README.md | head -n 500
-->

Please also read the [CONTRIBUTING.md](CONTRIBUTING.md) before making any changes.

## Setup

Install the pre-commit hooks before your first commit:

```console
uvx prek install
```

## Testing

Run the full test suite:

```console
uv run --group test pytest
```

Run with all optional dependencies:

```console
uv run --group test --extra=cli --extra=pygments --extra=audio pytest
```

Avoid mocking in tests. Only mock transports to avoid network IO or to mimic network counterparts.

## Documentation

Update documentation when changing or adding public APIs. The docs live in the `docs/` directory and are built with [MkDocs](https://www.mkdocs.org/).
