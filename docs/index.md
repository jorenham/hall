# Overview

{%
   include-markdown "../README.md"
   start="<!--badges-start-->"
   end="<!--badges-end-->"
%}

!!! note warning "Caution: under development"
    **Hall** is currently under development and is prone to interface changes.
    The documentation is currently incomplete, but will be coming soon.


Hall is a lightweight library with pythonic syntax. Some features include:

 - Clean, pythonic syntax that closely resembles mathematical notation in probability theory.
 - Symbolic algebra and bayesian statistics with random variables.
 - Calculations are numerically precise, with arbitrairy precision.
 - Lightweight; it only requires [mpmath](https://mpmath.org/).
 - Fully type-annotated and mypy friendly 
 - Thoroughly tested. 


## Example

{%
   include-markdown "../README.md"
   start="<!--example-iq-start-->"
   end="<!--example-iq-end-->"
%}


What's going on here:

 - `IQ` is a random variable with the normal distribution, scaled to have a mean of 100 and standard deviation of 15, i.e., \( \mathrm{IQ} \sim \mathcal N(100, 15^2) \).
 - We verify this using the operators for the expectancy \( \operatorname E[\cdot] \), and standard deviation \( \operatorname{Std}[\cdot] \).
 - Next, we obtain the probability of someone having an IQ of at least 130, \( \operatorname P (\mathrm{IQ} \ge 130) \).
 - Finally, we draw a sample from the random variable with the `hall.sample` function.


## Rationale

!!! note info "Coming soon"
