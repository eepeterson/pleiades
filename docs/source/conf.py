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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sys
import os

project = 'pleiades'
copyright = '2020, Ethan Peterson'
author = 'Ethan Peterson'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = '0.1.0'

# Determine if we're on Read the Docs server
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# On Read the Docs, we need to mock a few third-party modules so we don't get
# ImportErrors when building documentation
from unittest.mock import MagicMock


MOCK_MODULES = ['numpy', 'scipy.special', 'multiprocessing',
                'matplotlib.patches', 'matplotlib.collections',
                'matplotlib.transforms', 'numbers', 'pickle', 'scipy',
                'matplotlib.pyplot', 'os', 'warnings', 'matplotlib',
                'scipy.integrate', 'scipy.interpolate', 'scipy.optimize',
                'scipy.spatial', 'matplotlib.path']
sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../sphinxext'))
sys.path.insert(0, os.path.abspath('../..'))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinxcontrib.katex',
              'sphinx_numfig',
              'notebook_sphinxext']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The name of the Pygments (syntax highlighting) style to use.
#pygments_style = 'sphinx'
#pygments_style = 'friendly'
#pygments_style = 'bw'
#pygments_style = 'fruity'
#pygments_style = 'manni'
pygments_style = 'tango'

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_logo = '_images/pleiades_logo.svg'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Pleiades Documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def setup(app):
    app.add_stylesheet('theme_overrides.css')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'sparcnxdoc'

autosummary_generate = True

napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None)
}
