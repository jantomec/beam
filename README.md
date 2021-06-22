# beam
Project *beam* is a purely python based library with research code that provides a testing environment for finite element analysis code of one-dimensional slender structures.
The goal is to set up a framework, where one can easily code new ideas and test their potential. The main advantage over other software is that it is lightweight and 
therefore easy to manipulate on any level.

Features:
- Finite element framework (assembly of matrices, Newton-Raphson iterative algortihm, postprocessing etc.)
- Beam element based on Simo–Reissner theory for static and dynamic analysis
- Option to implement additional elements
- Experimental contact detection
- Designed to work both in Jupyter notebooks and standalone scripts

## Installation

Package installation:
```
pip install -i https://test.pypi.org/simple/ beam-pkg-tomecj==1.0.0
```

## Usage

See examplary files in the folder *beam_tests*

## Contribution

Contribution is very welcome. With the simple structure of the program in mind if you have any ideas on how to improve the project do not hesitate to contact me personally.

Setup of developer environment:
```
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade build
```

Package installation for development (go to root directory of the project and run in terminal):
```
python3 -m build
python3 -m pip install -e "full_path_to_beam_root"
```

To remove the package:
```
pip uninstall beam-pkg-tomecj
```