from __future__ import annotations

import os
import sys
from datetime import datetime

# Add src directory so autodoc can import project modules.
sys.path.insert(0, os.path.abspath("../src"))

project = "markovmodus"
author = "markovmodus developers"
year = datetime.utcnow().year
copyright = f"{year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/subthaumic/markovmodus/",
    "source_branch": "main",
    "source_directory": "docs/",
}

linkcheck_ignore = [
    r"https://github\.com/subthaumic/markovmodus/compare/.*",
    r"https://github\.com/subthaumic/markovmodus/releases/.*",
]
