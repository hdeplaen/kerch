# RKM

RKM is a Python package meant for various kernel methods, and in particular Deep Restricted Kernel Machines. You can natively perform SVMs, LS-SVMs, (K)PCA with various kernels, automatic centering, out-of-sample, etc.

The package is built on PyTorch and supports GPU acceleration.

<!-- toc -->

* [Examples](#examples)
  - [KPCA](#kernel-principal-component-analysis)
  - [Large-scale LS-SVM](#training-a-large-scale-least-squares-support-vector-machine)
  - [Deep RKM](#deep-restricted-kernel-machine)
  - [Recurrent RKM](#recurrent-restricted-kernel-machines)
* [Installation](#installation)
  - [PIP](#pip)
  - [From source](#from-source)
* [Acknowledgements](#acknowledgements)
* [Resources](#resources)
* [License](#license)

## Examples

### Kernel Principal Component Analysis
Example to come...
### Training a large-scale Least-Squares Support Vector Machine
Example to come...
### Deep Restricted Kernel Machine
Example to come...
### Recurrent Restricted Kernel Machines 
Example to come...

## Installation
As for now, there are two ways to install the package.

### PIP
Using pip, it suffices to run `pip install git+https://github.com/hdeplaen/rkm.git`. Just rerun this command to update the package to its newest version.

### From source
You can also install the package directly from the GitHub repository.
```
git clone --recursive https://github.com/hdeplaen/rkm
cd rkm
python setup.py install
```

## Acknowledgements

* EU: The research leading to these results has received funding from the European Research Council under the European Union's Horizon 2020 research and innovation program / ERC Advanced Grant E-DUALITY (787960).
* Research Council KUL:
    - Optimization frameworks for deep kernel machines C14/18/068
* Flemish Government:
    - FWO: projects: GOA4917N (Deep Restricted Kernel Machines: Methods and Foundations), PhD/Postdoc grant
    - This research received funding from the Flemish Government (AI Research Program). Johan Suykens and [your name] are affiliated to Leuven.AI - KU Leuven institute for AI, B-3000, Leuven, Belgium.
* Ford KU Leuven Research Alliance Project KUL0076 (Stability analysis and performance improvement of deep reinforcement learning algorithms)

## Resources

* [Documentation]("https://hdeplaen.github.io/rkm/)
* [Bug Tracker](https://github.com/hdeplaen/rkm/issues)
* [E-DUALITY]("https://www.esat.kuleuven.be/stadius/E/): ERC Adv. Grant website.
* [ESAT-STADIUS](https://www.esat.kuleuven.be/stadius/): KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for
    Dynamical Systems, Signal Processing and Data Analytics.


## License
RKM has a MIT license, as found in the [LICENSE](LICENSE) file.