# beam
Project *beam* is a purely python based library that provides a testing environment for finite element analysis code of one-dimensional slender structures.
The goal is to set up a framework, where one can easily code new ideas and test their potential. The main advantage over other software is that it is lightweight and 
therefore easy to manipulate on any level.

Features:
- Finite element framework (assembly of matrices, Newton-Raphson iterative algortihm, postprocessing etc.)
- Beam element based on Simo–Reissner theory for static and dynamic analysis
- Option to implement additional elements
- Experimental contact detection
- Designed to work both in Jupyter notebooks and standalone scripts

## Installation

Clone this repository with *git* and import it as any other module.

Dependencies:
- *NumPy*
- *Matplotlib*

## Usage

See examplary files in the folder *beam_tests*

## Contribution

Contribution is very welcome. With the simple structure of the program in mind if you have any ideas on how to improve the project do not hesitate to 
contact me personally.
