# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import jaxrts
import pathlib

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

html_static_path = ['_static']
html_css_files = ['custom.css']

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

from pybtex.plugin import register_plugin
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting import BaseSortingStyle
from pybtex.style.formatting.unsrt import Style as UnsrtStyle

LATEX_TO_UNICODE = {
    '\\"{a}': "ä",
    '\\"{o}': "ö",
    '\\"{u}': "ü",
    '\\"{A}': "Ä",
    '\\"{O}': "Ö",
    '\\"{U}': "Ü",
    "\\'{a}": "á",
    "\\'{e}": "é",
    "\\'{i}": "í",
    "\\'{o}": "ó",
    "\\'{u}'": "ú",
    "\\'{A}": "Á",
    "\\'{E}": "É",
    "\\'{I}": "Í",
    "\\'{O}": "Ó",
    "\\'{U}": "Ú",
    '\\"a': "ä",
    '\\"o': "ö",
    '\\"u': "ü",
    '\\"A': "Ä",
    '\\"O': "Ö",
    '\\"U': "Ü",
    "\\'a": "á",
    "\\'e": "é",
    "\\'i": "í",
    "\\'o": "ó",
    "\\'u'": "ú",
    "\\'A": "Á",
    "\\'E": "É",
    "\\'I": "Í",
    "\\'O": "Ó",
    "\\'U": "Ú",
    "\\ss{}": "ß",
    "\\ss": "ß",
}


def fix_latex(name: str) -> str:
    """Replace LaTeX escape sequences with proper unicode characters."""
    for latex, uni in LATEX_TO_UNICODE.items():
        name = name.replace(latex, uni)
    # Strip any remaining braces
    name = name.replace("{", "").replace("}", "")
    return name


def get_last(person) -> str:
    last_parts = person.last_names
    last = str(last_parts[0]) if last_parts else "Anon"
    return fix_latex(last)


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        counts = {}  # track raw labels for disambiguation

        # First pass: generate raw labels
        raw_labels = []
        for entry in sorted_entries:
            persons = entry.persons.get("author", [])
            year = entry.fields.get("year", "")

            if len(persons) == 0:
                name_part = "Anon."
            elif len(persons) == 1:
                name_part = get_last(persons[0])
            elif len(persons) == 2:
                name_part = (
                    f"{get_last(persons[0])} and {get_last(persons[1])}"
                )
            else:
                name_part = f"{get_last(persons[0])} et al."

            label = f"{name_part}, {year}"
            raw_labels.append(label)
            counts[label] = counts.get(label, 0) + 1

        # Second pass: disambiguate duplicate labels with a, b, c, ...
        seen = {}
        final_labels = []
        for label in raw_labels:
            if counts[label] > 1:
                idx = seen.get(label, 0)
                seen[label] = idx + 1
                final_labels.append(f"{label}{chr(ord('a') + idx)}")
            else:
                final_labels.append(label)

        yield from final_labels


class AuthorYearSortingStyle(BaseSortingStyle):
    def sort(self, entries):
        def sort_key(entry):
            persons = entry.persons.get("author", [])
            year = entry.fields.get("year", "")
            last = get_last(persons[0]) if persons else "Anon"
            return (last.lower(), year)

        return sorted(entries, key=sort_key)


class AuthorYearStyle(UnsrtStyle):
    default_label_style = AuthorYearLabelStyle
    default_sorting_style = AuthorYearSortingStyle


register_plugin("pybtex.style.formatting", "author_year_bib", AuthorYearStyle)

bibtex_bibfiles = [
    str(pathlib.Path(jaxrts.__file__).parent / "literature.bib")
]
bibtex_reference_style = "label"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

import pathlib

# Automatically create an overview page for the models implemented
import sys

sys.path.append(str(pathlib.Path.cwd()))

from available_model_overview import generate_available_model_overview_page

generate_available_model_overview_page()
