"""Sphinx configuration file."""

import os
import sys

sys.path.insert(0, os.path.abspath("../topobench"))


project = "TopoBench"
copyright = "2025, Topological-Intelligence Team, Inc."
author = "Topological-Intelligence Team Authors"

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_gallery.load_style",
    "sphinx.ext.autosummary",
    "myst_parser",
]

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"

nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "torch_geometric", "pyg_lib",
    "torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv"
]


master_doc = "index"

language = "en"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_baseurl = "https://geometric-intelligence.github.io/topobench"
htmlhelp_basename = "topobenchdoc"
html_last_updated_fmt = "%c"

html_sidebars = {
    "**": []
}

html_show_sourcelink = False

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
}

latex_elements = {}


latex_documents = [
    (
        master_doc,
        "topobench.tex",
        "TopoBench Documentation",
        "manual",
    ),
]

man_pages = [(master_doc, "topobench", "TopoBench Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "topobench",
        "TopoBench Documentation",
        author,
        "topobench",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]

from pathlib import Path
import shutil

def _copy_thumbnails(app):
    """
    Copy thumbnail png files from <confdir>/_thumbnails to <outdir>/_thumbnails
    once per build (on 'builder-inited').
    """
    src = Path(app.confdir) / "_thumbnails"
    if not src.exists():
        return  # nothing to do

    des = Path(app.outdir) / "_thumbnails"  # app.outdir == docs/_build (html builder)
    # make destination (and parents) if needed
    des.mkdir(parents=True, exist_ok=True)

    # copy all files (keep subfolders)
    for p in src.rglob("*"):
        if p.is_file():
            target = des / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def setup(app):
    app.connect("builder-inited", _copy_thumbnails)

nbsphinx_thumbnails = {
    "notebooks/tutorial_dataset": "_thumbnails/tutorial_dataset.png",
    "notebooks/tutorial_lifting": "_thumbnails/tutorial_lifting.png",
    "notebooks/tutorial_model": "_thumbnails/tutorial_model.png",
}

# configure intersphinx
intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "toponetx": ("https://pyt-team.github.io/toponetx/", None),
}

# configure numpydoc
numpydoc_validation_checks = {"all", "GL01", "ES01", "SA01", "EX01"}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
