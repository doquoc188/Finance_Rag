"""Lightweight dependency-boundary checks without architecture-test plugins."""

import ast
import os
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import finrag
import finrag.domain


def test_domain_imports_only_standard_library_or_domain_modules() -> None:
    domain_dir = Path(finrag.domain.__file__).parent
    for path in domain_dir.rglob("*.py"):
        relative_parent = path.parent.relative_to(domain_dir)
        package = ".".join(("finrag", "domain", *relative_parent.parts))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                modules = [resolve_name(module, package) if node.level else module]
            for module in modules:
                assert (
                    module == "finrag.domain"
                    or module.startswith("finrag.domain.")
                    or module.split(".")[0] in sys.stdlib_module_names
                ), f"Domain dependency boundary violated: {path.name} imports {module}"


def test_domain_imports_and_constructs_values_without_site_packages(tmp_path: Path) -> None:
    # -I -S removes environment import paths and site-packages, including Pydantic/SDKs.
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from finrag.domain import DocumentChunk, RetrievedEvidence, RetrievalStrategy, SourceReference
source = SourceReference('document', 'report.pdf', page=1)
chunk = DocumentChunk('chunk', 'document', 'Evidence', source)
result = RetrievedEvidence(chunk, 1, RetrievalStrategy.DENSE)
assert result.chunk.source_reference == source
print('stdlib-only domain import passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(finrag.__file__).parent.parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "stdlib-only domain import passed"


def test_installed_package_and_new_settings_have_no_deprecation_warnings(tmp_path: Path) -> None:
    # Isolated import checks editable packaging outside pytest's root pythonpath.
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("FINRAG_")
    }
    script = """
from finrag.config import Settings
settings = Settings()
assert settings.graphrag.enabled is False
print('installed settings import passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-W", "error::DeprecationWarning", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "installed settings import passed"
