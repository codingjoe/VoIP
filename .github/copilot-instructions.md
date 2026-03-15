All code MUST ALWAYS follow the `naming-things` guidelines!

Use the following command to access the guidelines:

```console
curl -sSL https://raw.githubusercontent.com/codingjoe/naming-things/refs/heads/main/README.md | cat
```

All code must be fully tested with a 100% coverage. Unreachable code must be removed.
Follow the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines for testing and linting.
The test suite will fail if the line coverage is below 100%.
The test coverage will be written to stdout when running the tests.
Fully covered files are omitted from the coverage report.

All transformers must be documented in the /docs folder.

We avoid private functions and variables.
We use type annotations.
We use dataclasses where adequate.
You MUST reason about performance and memory usage.

Avoid overly complex functions. Break them into smaller functions if necessary.

Write docstrings with the Google notation, type annotations for all functions, classes, and methods.
We use `mkdocstrings` to generate documentation from docstrings, so they must use Markdown syntax.
You may use Markdown links in docstrings to link to other functions, classes, or methods. The link target must be the full Python path to the function.
For external links use the \<>-notation or reference-style notation with the links at the end of the docstring.
We don't write docstrings for inherited methods or properties nor for functions we don't want to expose in the documentation.
Docstrings should be written in present tense and imperative mood.
They must start with a capital letter and end with a period.
Docstrings must describe the external behavior of the function, class, or method.
Docstrings should avoid redundant phrases like "This function" or "This method".
Class docstrings must not repeat the class name or start with a verb since they don't do anything themselves.
Avoid code comments unless they describe the behavior of 3rd-party code or complex algorithms.
Avoid loops in favor of recursive functions or generator functions.
Avoid functions or other code inside functions.
Avoid if-statements in favor of match-statements or polymorphism.
Do not assign names to objects which are returned in the next line.
