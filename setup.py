import setuptools
import pathlib
import pkg_resources
import kerch

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

with open("LICENSE", "r", encoding="utf-8") as fh:
    license = fh.read()

setuptools.setup(
    name='kerch',
    version=kerch.__version__,
    author=kerch.__author__,
    author_email='henri.deplaen@esat.kuleuven.be',
    description='Kernel Methods with PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hdeplaen/kerch',
    download_url='https://github.com/hdeplaen/kerch/archive/{}.tar.gz'.format(kerch.__version__),
    project_urls = {
        "Documentation": "https://hdeplaen.github.io/kerch",
        "Bug Tracker": "https://github.com/hdeplaen/kerch/issues",
        "E-DUALITY": "https://www.esat.kuleuven.be/stadius/E/",
        "ESAT-STADIUS": "https://www.esat.kuleuven.be/stadius/"
    },
    platforms=['linux', 'macosx', 'windows'],
    license=license,
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ]
)