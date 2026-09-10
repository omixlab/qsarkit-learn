# Local stub shadow for RDKit.
#
# The rdkit-stubs package bundled with the RDKit wheel is auto-generated
# and contains a syntax error (a C++ enum member literally named `None`,
# which is an illegal annotation target in a .pyi). mypy parses stubs
# before anything else, so that single bug aborts the entire run and
# hides every real error in our own code.
#
# These permissive stubs take precedence via `mypy_path` in pyproject.toml.
# RDKit symbols resolve to Any - the same result `ignore_missing_imports`
# would give - while our own annotations stay fully checked.
from typing import Any

def __getattr__(name: str) -> Any: ...
