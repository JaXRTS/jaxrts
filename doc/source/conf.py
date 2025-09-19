# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import jaxrts
import pathlib

project = "jaxrts"
copyright = "2024, J. Lütgert and S. Schumacher"
author = "J. Lütgert, S. Schumacher"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_toolbox.collapse",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.viewcode",
]


autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "private-members",
    "show-inheritance",
]
autodoc_default_flags = ["members"]
autosummary_generate = True  # Make _autosummary files and include them

# Napoleon settings
napoleon_google_docstring = False
napoleon_use_rtype = False
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = []


# Sphinx gallery
from sphinx_gallery.scrapers import matplotlib_scraper


class matplotlib_svg_scraper:
    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, format="svg", **kwargs)


sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": "gen_examples",  # path to where to save gallery generated output
    "reference_url": {
        # The module you locally document uses None
        "jaxrts": None,
    },
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("jaxrts"),
    "exclude_implicit_doc": {},
    "prefer_full_module": {r"module\.submodule"},
    "image_scrapers": (matplotlib_svg_scraper(),),
}

# bibtex
bibtex_bibfiles = [str(pathlib.Path(jaxrts.__file__).parent / "literature.bib")]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

import pathlib

# Automatically create an overview page for the models implemented
import sys

sys.path.append(str(pathlib.Path.cwd()))

from available_model_overview import generate_available_model_overview_page

generate_available_model_overview_page()
