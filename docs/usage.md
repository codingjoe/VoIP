# Usage

The `voip` library provides both a command-line interface (CLI) for quick interactions and a Python API for building custom SIP applications.

## Command Line Interface (CLI)

The CLI tool `voip` is installed with the package. It allows you to register with a SIP provider and answer calls, transcribing audio in real-time.

### Installation

Ensure you have installed the CLI dependencies:

```console
pip install voip[cli]
```

### Transcribe Calls

The `voip sip transcribe` command registers your SIP account and answers incoming calls, printing live transcriptions to the terminal.

```console
voip sip transcribe [OPTIONS] AOR
```

**Arguments:**

- `AOR`: Your Address of Record (e.g., `sips:alice@example.com`).

**Options:**

- `--password TEXT`: SIP password (required).
- `--username TEXT`: SIP username (defaults to user part of AOR).
- `--proxy HOST[:PORT]`: Outbound proxy address.
- `--stun-server HOST[:PORT]`: STUN server for NAT traversal (default: `stun.cloudflare.com:3478`).
- `--model TEXT`: Whisper model size (default: `base`).
- `--no-tls`: Force plain TCP (auto-detected for port 5060).
- `--no-verify-tls`: Disable TLS certificate verification (insecure).
- `-v, --verbose`: Increase verbosity.

**Example:**

```console
voip sip transcribe sips:alice@sip.example.com --password secret
```

## Python API

You can use the `voip` library to build custom SIP applications, such as automated answering services, recording bots, or custom transcription pipelines.

### Basic Example

Here is a basic example of how to answer a call and transcribe audio using the `WhisperCall` class.

```python
import asyncio
import ssl
from voip.audio import WhisperCall
from voip.sip.protocol import SIP


class MyCall(WhisperCall):
    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class MySession(SIP):
    def call_received(self, request) -> None:
        # Answer the incoming call with our custom call handler
        asyncio.create_task(self.answer(request=request, call_class=MyCall))


async def main():
    loop = asyncio.get_running_loop()
    ssl_context = ssl.create_default_context()

    # Create the SIP session and connect
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@example.com",
            username="alice",
            password="secret",
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl_context,
    )

    # Keep the event loop running
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Call Handling

To implement custom logic for audio processing (e.g., saving to disk, streaming to another service), subclass `voip.audio.AudioCall` and override `audio_received`.

See [Calls](calls.md) for more details on the call handling API.
