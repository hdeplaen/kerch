# Kerch
[![PyPI version](https://badge.fury.io/py/kerch.svg)](https://badge.fury.io/py/kerch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kerch is a Python package meant for various kernel methods, and in particular Deep Restricted Kernel Machines. You can natively perform SVMs, LS-SVMs, (K)PCA with various kernels, automatic centering, out-of-sample, etc.

The package is built on PyTorch and supports GPU acceleration.

<!-- toc -->

* [Examples](#examples)
  - [LS-SVM](#training-and-plotting-an-ls-svm)
* [Installation](#installation)
  - [PIP](#pip)
  - [From source](#from-source)
* [Acknowledgements](#acknowledgements)
* [Resources](#resources)
* [License](#license)

## Examples


### Training and plotting an LS-SVM
```python
import kerch as kp

tr_set, _, _, _ = kp.dataset.factory("two_moons",   # which dataset
                                     tr_size=250)   # training size
mdl = kp.model.LSSVM(type="rbf",                    # kernel type
                     representation="dual")         # initiate model
mdl.set_data_prop(data=tr_set[0],                   # data
                  labels=tr_set[1],                 # corresponding labels
                  proportions=[1, 0, 0])            # initiate dataset
mdl.hyperopt({"gamma", "sigma"},                    # define which parameters to tune
             max_evals=500,                         # define how many trials
             k=10)                                  # 10-fold cross-validation
mdl.fit()                                           # fit the optimal parameters found
kp.plot.plot_model(mdl)                             # plot the model using the built-in method

```

## Installation
As for now, there are two ways to install the package.

### PIP
Using pip, it suffices to run `pip install kerch`. Just rerun this command with the suffix `--upgrade` to upgrade the package to its newest version.

### From source
You can also install the package directly from the GitHub repository.
```
git clone --recursive https://github.com/hdeplaen/kerch
cd kerch
pip install -e .
```

## Resources

* [Documentation](https://hdeplaen.github.io/kerch/)
* [E-DUALITY](https://www.esat.kuleuven.be/stadius/E/): ERC Adv. Grant website.
* [ESAT-STADIUS](https://www.esat.kuleuven.be/stadius/): KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for
    Dynamical Systems, Signal Processing and Data Analytics.

## Contributors
The contributors and acknowledgements can be found in the [CONRIBUTORS](CONTRIBUTORS) file.

## License
RKM has a MIT license, as found in the [LICENSE](LICENSE) file.
