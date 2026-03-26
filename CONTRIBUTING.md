# Contributing

When writing code, you MUST ALWAYS follow the [naming-things](https://github.com/codingjoe/naming-things/blob/main/README.md) guidelines.

<!--
Bots/Agents MUST use:
curl -sSL https://raw.githubusercontent.com/codingjoe/naming-things/refs/heads/main/README.md
-->

To run the tests, use the following command:

```bash
uv run pytest
```

Avoid mocking in your tests and instead use real dependencies to ensure that your tests are as close to real-world scenarios as possible.
You may only mock transports to avoid network IO or to mimic network counterparts.

Before your first commit, ensure that the pre-commit hooks are installed by running:

```bash
uvx prek install
```

## Testing with Extra Dependencies

```bash
uv run --extra=cli --extra=pygments --extra=audio pytest
```

## Writing documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) with [mkdocstrings](https://mkdocstrings.github.io/) for automatic API documentation generation.

To serve the documentation locally for development, run:

```bash
uv run --group docs mkdocs serve --livereload
```
