import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license = fh.read()

setuptools.setup(
    name='rkm',
    version='0.1',
    author='HENRI DE PLAEN',
    author_email='henri.deplaen@esat.kuleuven.be',
    description='Restricted Kernel Machines',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hdeplaen/rkm',
    project_urls = {
        "E-DUALITY": "https://www.esat.kuleuven.be/stadius/E/",
        "ESAT-STADIUS": "https://www.esat.kuleuven.be/stadius/",
        "Documentation": None,
        "Bug Tracker": "https://github.com/hdeplaen/rkm/issues"
    },
    license=license,
    install_requires=requirements,
)