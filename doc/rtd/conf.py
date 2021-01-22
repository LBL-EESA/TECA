# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'TECA'
copyright = "2019, Burlen Loring, Travis O'Brien & Abdelrahman Elbashandy"
author = "Burlen Loring, Travis O'Brien & Abdelrahman Elbashandy"


# -- General configuration ---------------------------------------------------

# Run Doxygen to generate Doxygen's XML output for autodoc by Breathe
import subprocess, os
from exhale import utils

#read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

#if read_the_docs_build:
#    subprocess.call('cd ../doxygen; doxygen -d Preprocessor', shell=True)

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'exhale'
]

# Setup the breathe extension
breathe_projects = {
    #"TECA": "../doxygen/xml"
    "TECA": "xml"
}
breathe_default_project = "TECA"

#### AR: START ######

def specificationsForKind(kind):
    '''
    For a given input ``kind``, return the list of reStructuredText specifications
    for the associated Breathe directive.
    '''
    # Change the defaults for .. doxygenclass:: and .. doxygenstruct::
    if kind == "class" or kind == "struct":
        return [
          ":members:",
          ":undoc-members:"
        ]
    # Change the defaults for .. doxygenenum::
    elif kind == "enum":
        return [":no-link:"]
    elif kind == "enum":
        return [":no-link:"]
    # An empty list signals to Exhale to use the defaults
    else:
        return []

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "framework_root.rst",
    "rootFileTitle":         "Framework API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    "verboseBuild":          True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    #"exhaleDoxygenStdin":    "INPUT = ../include"
    # AR: set values
    "exhaleUseDoxyfile":     True,
    "unabridgedOrphanKinds": {"dir", "file", "typedef"},
    "customSpecificationsMapping": utils.makeCustomSpecificationsMapping(
        specificationsForKind
    ),
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'text'
#### AR: END ######

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

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
#html_static_path = ['_static']
html_static_path = []

numfig = True
