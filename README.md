# KerPy
[![PyPI version](https://badge.fury.io/py/kerpy.svg)](https://badge.fury.io/py/kerpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

KerPy is a Python package meant for various kernel methods, and in particular Deep Restricted Kernel Machines. You can natively perform SVMs, LS-SVMs, (K)PCA with various kernels, automatic centering, out-of-sample, etc.

The package is built on PyTorch and supports GPU acceleration.

<!-- toc -->

[//]: # (* [Examples]&#40;#examples&#41;)

[//]: # (  - [KPCA]&#40;#kernel-principal-component-analysis&#41;)

[//]: # (  - [Large-scale LS-SVM]&#40;#training-a-large-scale-least-squares-support-vector-machine&#41;)

[//]: # (  - [Deep RKM]&#40;#deep-restricted-kernel-machine&#41;)

[//]: # (  - [Recurrent RKM]&#40;#recurrent-restricted-kernel-machines&#41;)
* [Installation](#installation)
  - [PIP](#pip)
  - [From source](#from-source)
* [Acknowledgements](#acknowledgements)
* [Resources](#resources)
* [License](#license)

[//]: # (## Examples)

[//]: # ()
[//]: # (### Kernel Principal Component Analysis)

[//]: # (Example to come...)

[//]: # (### Training a large-scale Least-Squares Support Vector Machine)

[//]: # (Example to come...)

[//]: # (### Deep Restricted Kernel Machine)

[//]: # (Example to come...)

[//]: # (### Recurrent Restricted Kernel Machines )

[//]: # (Example to come...)

## Installation
As for now, there are two ways to install the package.

### PIP
Using pip, it suffices to run `pip install kerpy`. Just rerun this command with the suffix `--upgrade` to upgrade the package to its newest version.

### From source
You can also install the package directly from the GitHub repository.
```
git clone --recursive https://github.com/hdeplaen/kerpy
cd kerpy
python setup.py install
```

## Resources

* [Documentation](https://hdeplaen.github.io/kerpy/)
* [Bug Tracker](https://github.com/hdeplaen/kerpyissues)
* [E-DUALITY](https://www.esat.kuleuven.be/stadius/E/): ERC Adv. Grant website.
* [ESAT-STADIUS](https://www.esat.kuleuven.be/stadius/): KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for
    Dynamical Systems, Signal Processing and Data Analytics.

## Contributors
The contributors and acknowledgements can be found in the [CONRIBUTORS](CONTRIBUTORS) file.

## License
RKM has a MIT license, as found in the [LICENSE](LICENSE) file.
