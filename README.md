# &infin;RETIS
![Tests](https://github.com/infretis/infretis/actions/workflows/test.yaml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/infretis/infretis/badge.svg?branch=main)](https://coveralls.io/github/infretis/infretis?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction
∞RETIS is a Python library designed to facilitate Replica Exchange Transition Interface Sampling (RETIS).
∞RETIS interfaces with molecular dynamics (MD) programs such as
[GROMACS](https://www.gromacs.org/), [LAMMPS](https://www.lammps.org/), and [CP2K](https://www.cp2k.org/), offering researchers an
efficient and flexible tool for advanced simulations.

Our recent publication
[Highly Parallelizable Path Sampling with Minimal Rejections Using Asynchronous Replica Exchange and Infinite Swaps](https://www.pnas.org/doi/10.1073/pnas.2318731121)
showcases the capabilities of ∞RETIS.
We are also in the process of preparing a detailed paper specifically about this software library.

## Features
- **Advanced Sampling Techniques**: Utilizes the [RETIS](https://doi.org/10.1103/PhysRevLett.98.268301) method
  and more specialized subtrajectory moves with high acceptance such as [Wire Fencing](https://doi.org/10.1063/5.0127249) for
  efficient sampling of rare events in molecular simulations.
- **Parallel sampling**: Utilizes a [asynchronous replica exchange scheme for parallelization with infinite swapping](https://doi.org/10.1021/acs.jpca.2c06004).
- **Seamless Integration**: Easy interfacing with popular MD programs like GROMACS, LAMMPS, and CP2K. Please see the [examples](#Examples).


## Installation
∞RETIS can be installed via [pip](https://pypi.org/project/infretis/):

```bash
pip install infretis
```

⚠️ **Note**: ∞RETIS does not manage the installation of external MD programs. Users are responsible for installing and configuring these programs independently.




## Examples

We have provided several examples to help you get started with ∞RETIS:

* [sugar puckering](/examples/gromacs/puckering/), which is an in-depth tutorial
* [chignolin unfolding](https://github.com/infretis/infentory/tree/main/chignolin)
* .. and much more available on the [infentory](https://github.com/infretis/infentory/) repository!

## Contributing


## Citation

If you use ∞RETIS in your research, please cite our paper:
[Highly Parallelizable Path Sampling with Minimal Rejections Using Asynchronous Replica Exchange and Infinite Swaps](https://www.pnas.org/doi/10.1073/pnas.2318731121)

To cite ∞RETIS in your work, please use the following BibTeX entry:

```bibtex
@article{zhang_highly_2024,
	title = {Highly parallelizable path sampling with minimal rejections using asynchronous replica exchange and infinite swaps},
	volume = {121},
	url = {https://www.pnas.org/doi/10.1073/pnas.2318731121},
	doi = {10.1073/pnas.2318731121},
	number = {7},
	journal = {Proceedings of the National Academy of Sciences},
	author = {Zhang, Daniel T. and Baldauf, Lukas and Roet, Sander and Lervik, Anders and van Erp, Titus S.},
	year = {2024},
	publisher = {Proceedings of the National Academy of Sciences},
}
```

## License

∞RETIS is licensed under the MIT License. Please see the file [`LICENSE`](LICENSE)

## Contact

## Acknowledgements

Acknowledge any contributors, funding sources, or institutions that played a significant role in the development of ∞RETIS.
