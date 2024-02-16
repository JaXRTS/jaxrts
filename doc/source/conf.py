# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": "gen_examples",  # path to where to save gallery generated output
    "reference_url": {
        # The module you locally document uses None
        "jaxrts": None,
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("jaxrts"),
    # Regexes to match objects to exclude from implicit backreferences.
    # The default option is an empty set, i.e. exclude nothing.
    # To exclude everything, use: '.*'
    "exclude_implicit_doc": {},
}

# bibtex
bibtex_bibfiles = ["literature.bib"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
