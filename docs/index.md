# Overview

{%
   include-markdown "../README.md"
   start="<!--badges-start-->"
   end="<!--badges-end-->"
%}

`TODO`

# Install

Installation is as simple as:

```bash
pip install hall
```

*Hall* has no required dependencies except python 3.8, 3.9 or 3.10, and
[`mpmath`](https://mpmath.org/doc/current/setup.html#download-and-installation).
If you have `pip` installed, you're good to go.

### Increase performance with [gmpy](https://github.com/aleaxit/gmpy) (optional)

Optionally, the `gmpy2` dependency can be installed to significantly speed up
the multi-precision arithmatical calculations. This adds bindings to the
[`GMP`](http://gmplib.org/) ([`MPIR`](http://www.mpir.org/) on Windows),
[`MPFR`](http://www.mpfr.org/), and [`MPC`](http://mpc.multiprecision.org/)
libraries. On Ubuntu-based systems, these are installed with:

```bash
apt install libgmp-dev libmpfr-dev libmpc-dev
```

Now you can install the `gmpy2` dependency with:

```bash
pip install hall[gmpy2]
```

To verify that `gmpy` is installed, check the backend:

```pycon
>>> from hall.backend import get_backend
>>> get_backend()
'gmpy'
```


# Examples

### Intelligence Quotient (IQ)

{%
   include-markdown "../README.md"
   start="<!--example-iq-start-->"
   end="<!--example-iq-end-->"
%}


# Contributing

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
