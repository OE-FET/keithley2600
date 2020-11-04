# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../keithley2600"))

autodoc_mock_imports = ["pyvisa"]

# -- Project information -----------------------------------------------------

project = "keithley2600"
copyright = "2019, Sam Schott"
author = "Sam Schott"
version = "2.0.0.post0"
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",  # support numpy style docstrings in config module
    "sphinx.ext.todo",  # parse todo list
    "sphinx.ext.intersphinx",  # support for if-clauses in docs
    "sphinx.ext.ifconfig",  # support for linking between documentations
    "autoapi.extension",  # builds API docs from doc strings without importing module
    "m2r",  # convert markdown to rest
]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Override default css with our own theme
html_context = {
    "css_files": [
        "https://media.readthedocs.org/css/sphinx_rtd_theme.css",
        "https://media.readthedocs.org/css/readthedocs-doc-embed.css",
        "_static/custom.css",
    ],
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "keithley2600doc"

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (
        master_doc,
        "keithley2600.tex",
        "keithley2600 Documentation",
        "Sam Schott",
        "manual",
    ),
]

# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {"https://docs.python.org/": None}
todo_include_todos = True
autoapi_add_toctree_entry = False
autoapi_type = "python"
autoapi_dirs = ["../keithley2600"]
autoapi_options = [
    "members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
