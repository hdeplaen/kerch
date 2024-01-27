# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import kerch

# -- Project information -----------------------------------------------------

project = 'kerch'
copyright = kerch.__credits__ + ", " + kerch.__date__
author = kerch.__author__

# The full version, including alpha/beta/rc tags
release = kerch.__version__

# other
source_encoding = "utf-8-sig"
language = 'en'
root_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension level names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax',
              'sphinx_toolbox.more_autodoc.variables',
              'sphinx.ext.autodoc',
              "sphinx.ext.doctest",
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx_rtd_theme',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.graphviz',
              'sphinx.ext.githubpages',
              'sphinx_exec_code',
              'sphinx_new_tab_link',
              'sphinx_codeautolink',
              'sphinx_design',
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
}

# INHERITANCE DIAGRAMS
graphviz_output_format = 'png'
inheritance_graph_attrs = dict(rankdir="TB", fontsize=12, size='"16.0, 20.0"')


def shorten_submodules(submods: list):
    shorts = dict()
    for mod_name in submods:
        mod = getattr(kerch, mod_name)
        for cl_name in mod.__all__:
            cl = getattr(mod, cl_name)
            if hasattr(cl, '__module__'):
                cl_mod_name = cl.__module__
                short_cl_mod_name = '.'.join(cl_mod_name.split('.')[:-1])
                shorts[cl_mod_name + '.' + cl_name] = short_cl_mod_name + '.' + cl_name
    return shorts


inheritance_alias = shorten_submodules(['kernel', 'level', 'feature', 'model'])

# MAPPING
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'torch': ('https://pytorch.org/docs/stable', None)}





# GITHUB
github_username = 'hdeplaen'
github_repository = 'kerch'
