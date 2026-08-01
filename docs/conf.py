"""Sphinx configuration for the current TopoBench documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

project = "TopoBench"
copyright = "2025, Topological-Intelligence Team, Inc."
author = "Topological-Intelligence Team Authors"

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "plans/**",
    "**.ipynb_checkpoints",
]

# Runtime and documentation dependencies are installed for documentation jobs.
# Keeping this list empty makes missing current modules fail visibly.
autodoc_mock_imports: list[str] = []
autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__",
    "imported-members": False,
}
autodoc_type_aliases = {}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
autodoc_warningiserror = False
python_use_unqualified_type_names = True
add_module_names = False

autosummary_generate = True
autosummary_imported_members = False

numpydoc_validation_checks = set()
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torch_geometric": (
        "https://pytorch-geometric.readthedocs.io/en/latest/",
        None,
    ),
}

suppress_warnings = ["app.add_directive", "app.add_node"]
nitpicky = False
nitpick_ignore: list[tuple[str, str]] = []

pygments_style = None
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_baseurl = "https://geometric-intelligence.github.io/topobench"
htmlhelp_basename = "topobenchdoc"
html_last_updated_fmt = "%c"
html_sidebars = {"**": []}
html_show_sourcelink = False
html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
}
html_css_files = ["custom.css"]

latex_elements: dict[str, str] = {}
latex_documents = [
    (
        master_doc,
        "topobench.tex",
        "TopoBench Documentation",
        author,
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
        "A benchmark core for graph, heterogeneous graph, and hypergraph learning.",
        "Miscellaneous",
    ),
]
epub_title = project
epub_exclude_files = ["search.html"]
