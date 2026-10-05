# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import pathlib

sys.path.append(str(pathlib.Path.cwd()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import jaxrts

project = "jaxrts"
copyright = "2024-2025, J. Lütgert, S. Schumacher, and the jaxrts contributors"
author = "J. Lütgert, S. Schumacher, and the jaxrts contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_toolbox.collapse",
    "sphinx_toolbox.sidebar_links",
    "sphinx_toolbox.github",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.viewcode",
]

html_static_path = ["_static"]
html_css_files = ["custom.css"]

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


# sidebar-links settings

github_username = "jaxrts"
github_repository = "jaxrts"

# Sphinx gallery

from gallery_helpers import matplotlib_svg_scraper

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

from pybtex.plugin import register_plugin

from bibtex_helpers import AuthorYearStyle

register_plugin("pybtex.style.formatting", "author_year_bib", AuthorYearStyle)

bibtex_bibfiles = [
    str(pathlib.Path(jaxrts.__file__).parent / "literature.bib")
]
bibtex_reference_style = "label"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Automatically create an overview page for the models implemented
from available_model_overview import generate_available_model_overview_page

generate_available_model_overview_page()
