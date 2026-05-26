# Agents

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
uv run pytest
```

Run with all optional dependencies:

```console
uv run --extra=cli --extra=pygments --extra=audio pytest
```

Avoid mocking in tests. Only mock transports to avoid network IO or to mimic network counterparts.

## Documentation

Update documentation when changing or adding public APIs. The docs live in the `docs/` directory and are built with [MkDocs](https://www.mkdocs.org/).
