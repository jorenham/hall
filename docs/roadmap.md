# Roadmap

Hall is currently in the alpha phase and under development. 
At the moment, only basic random variable algebra and a handfull of 
distributions is implemented.

My goal is to make Hall a reliable library that can be used in production 
environments, as well as a reliable tool to teaching probability theory.

In upcoming releases, hall will add support for:

- Many of the commonly used discrete and continuous probability distributions
- Symbolic [algebra of multiple random variables/vectors](https://en.wikipedia.org/wiki/Algebra_of_random_variables) 
  (without sympy), e.g., the [birthday paradox](https://en.wikipedia.org/wiki/Birthday_problem)
  is `P(B1 == B2)` where `B1` and `B2` are `~Uniform(1, 365)`. 
- Joint distributions and random vectors
- Conditional random variables, covariance, correlation, etc.
- Complex distributions
- Clean interface for user-defined distributions

With lower probability, hall could also be able to:

- [Statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)
  with symbolic parameters (likelihood-based, bayesian, etc.)
- [Confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval), 
  with e.g. t- and z- tests.
- Optional numpy support (random matrices, and linear algebra thereof)
- Optional pandas support
- Optional numba or cupy support (for e.g. convolutions)  to increase performance

And with an even lower probability, hall might include support for:

- [Stochastic processes](https://en.wikipedia.org/wiki/Stochastic_process), 
  maybe even [random fields](https://en.wikipedia.org/wiki/Random_field)
- [Bayesian networks](https://en.wikipedia.org/wiki/Bayesian_network)
- [Mixture models](https://en.wikipedia.org/wiki/Mixture_model)
- [Markov kernels](https://en.wikipedia.org/wiki/Markov_kernel)
- [Fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic)
- [Fuzzy sets](https://en.wikipedia.org/wiki/Fuzzy_set)
- [Possiblity theory](https://en.wikipedia.org/wiki/Possibility_theory)
