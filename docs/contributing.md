# Contributing to Hall

Any contributions to *hall* are appreciated!

## Issues

Questions, feature requests and bug reports are all welcome as issues.

When reporting a bug, make sure to include the versions of `hall` and `mpmath`
you are using, as well as the backend (`hall.backend.get_backend()`), and
provide a **reproducable** example of the bug.

## Development

Ensure you have [poetry](https://python-poetry.org/docs/#installation) installed, then

```bash
poetry install
```

Additionally, install [`pre-commit`](https://pre-commit.com/#install), then run:

```bash
pre-commit install
```

This adds git hooks that automatically formats and checks the code before commiting.
