<!--badges-start-->
[![CI](https://github.com/jorenham/hall/workflows/CI/badge.svg?event=push)](https://github.com/jorenham/hall/actions?query=event%3Apush+branch%3Amaster+workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/hall.svg)](https://pypi.python.org/pypi/hall)
[![Downloads](https://pepy.tech/badge/hall/month)](https://pepy.tech/project/hall)
[![versions](https://img.shields.io/pypi/pyversions/hall.svg)](https://github.com/jorenham/hall)
[![license](https://img.shields.io/github/license/jorenham/hall.svg)](https://github.com/jorenham/hall/blob/master/LICENSE)
<!--badges-end-->

Probability theory using pythonic and (almost) mathematical notation.

## Help

See [documentation](https://jorenham.github.io/hall/) for more details.

## A simple example: Intelligence quotient

<!--example-iq-start-->
```pycon
>>> from hall import P, E, Std, Normal
>>> IQ = ~Normal(100, 15)
>>> E[IQ]
100.0
>>> Std[IQ]
15.0
>>> P(X >= 130)
0.0227501319481792
```

So the chance of having an IQ (normally distributed with μ=100 and σ=15) of at 
least 130 is approximately 2.3%.
<!--example-iq-end-->

## A simple example: Monty ~~Python~~ Hall 

`TODO`

## Contributing

For guidance on setting up a development environment and how to make a
contribution to *hall*, see
[Contributing to hall](https://jorenham.github.io/hall/#contributing).