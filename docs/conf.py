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

sys.path.insert(0, os.path.abspath(".."))
import lours

# -- Project information -----------------------------------------------------
project = "Lours"
author = "Lours community"
copyright = "2024, XXII"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"

# Define the json_url for our version switcher.
json_url = "https://lours.readthedocs.io/latest/_static/version_switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION", "")
release = lours.__version__
if version_match.isdigit():
    version_match = ""
elif version_match == "latest":
    version_match = "dev"
elif version_match == "stable":
    version_match = release

print(f"Version is {version_match}")

html_theme_options = {
    "navbar_align": "left",
    "logo": {
        "text": "Lours documentation",
    },
    "external_links": [
        {
            "url": "https://www.xxiiai.com/",
            "name": "XXII",
        }
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/XXII-AI/Lours",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lours/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "check_switcher": False,
    "navbar_start": ["navbar-logo", "version-switcher"],
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinxarg.ext",
]

html_css_files = [
    "css/custom.css",
]

favicons = [
    {"href": "cropped-favicon-192x192.png"},
    {
        "rel": "apple-touch-icon",
        "href": "cropped-favicon-180x180.png",
    },
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autodoc_typehints = "signature"
autoclass_content = "both"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
nbsphinx_allow_errors = True
nbsphinx_execute = "auto"
add_module_names = False
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# copybutton_exclude = ".linenos, .gp, .go"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_js_files = ["pypi.js"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pot": ("https://pythonot.github.io/", None),
    "fiftyone": ("https://docs.voxel51.com/", None),
}
