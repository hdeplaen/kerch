import setuptools
import pathlib
import pkg_resources

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
    version='0.2',
    author='HENRI DE PLAEN',
    author_email='henri.deplaen@esat.kuleuven.be',
    description='Kernel Methods on PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hdeplaen/kerch',
    project_urls = {
        "Documentation": "https://hdeplaen.github.io/kerch",
        "Bug Tracker": "https://github.com/hdeplaen/kerch/issues",
        "E-DUALITY": "https://www.esat.kuleuven.be/stadius/E/",
        "ESAT-STADIUS": "https://www.esat.kuleuven.be/stadius/"
    },
    license=license,
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)