Probability theory using pythonic and (almost) mathematical notation.

## Help

See [documentation](https://jorenham.github.io/hall/) for more details.

## A simple example: Intelligence quotient

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

## A simple example: Monty ~~Python~~ Hall 

`TODO`

## Contributing

For guidance on setting up a development environment and how to make a
contribution to *hall*, see
[Contributing to Pydantic](https://jorenham.github.io/hall/#contributing).