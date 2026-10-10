"""Lightweight dependency-boundary checks without architecture-test plugins."""

import ast
import os
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

import finrag
import finrag.config
import finrag.domain
import finrag.models
import finrag.providers
import finrag.retrieval
import finrag.verification


@pytest.mark.parametrize(
    ("package_name", "package_dir", "allowed_packages"),
    [
        ("finrag.domain", Path(finrag.domain.__file__).parent, ("finrag.domain",)),
        (
            "finrag.verification",
            Path(finrag.verification.__file__).parent,
            ("finrag.verification",),
        ),
        (
            "finrag.config",
            Path(finrag.config.__file__).parent,
            ("finrag.config", "finrag.domain", "pydantic", "pydantic_settings"),
        ),
        (
            "finrag.providers",
            Path(finrag.providers.__file__).parent,
            ("finrag.providers", "finrag.config", "langchain_core"),
        ),
        (
            "finrag.models",
            Path(finrag.models.__file__).parent,
            ("finrag.models", "finrag.config", "finrag.providers.registry", "langchain_core"),
        ),
        (
            "finrag.retrieval",
            Path(finrag.retrieval.__file__).parent,
            ("finrag.retrieval", "finrag.domain", "rank_bm25"),
        ),
    ],
)
def test_package_import_boundaries(
    package_name: str, package_dir: Path, allowed_packages: tuple[str, ...]
) -> None:
    for path in package_dir.rglob("*.py"):
        relative_parent = path.parent.relative_to(package_dir)
        package = ".".join((package_name, *relative_parent.parts))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        adapter_name = path.stem if package_name == "finrag.providers" else ""
        integration = {
            "openai": "langchain_openai",
            "groq": "langchain_groq",
            "anthropic": "langchain_anthropic",
            "google": "langchain_google_genai",
            "ollama": "langchain_ollama",
        }.get(adapter_name)
        permitted = allowed_packages + (("pydantic", integration) if integration else ())
        factory_nodes = {
            child
            for definition in tree.body
            if isinstance(definition, ast.FunctionDef)
            and definition.name == f"create_{adapter_name}_chat_model"
            for child in ast.walk(definition)
        }
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                modules = [resolve_name(module, package) if node.level else module]
            for module in modules:
                if integration and (module == integration or module.startswith(integration + ".")):
                    assert node in factory_nodes, f"{path.name}: integration import must be lazy"
                assert (
                    any(
                        module == allowed or module.startswith(allowed + ".")
                        for allowed in permitted
                    )
                    or module.split(".")[0] in sys.stdlib_module_names
                ), f"{package_name} dependency boundary violated: {path.name} imports {module}"


def test_graph_imports_resolved_models_without_provider_or_settings_dependencies() -> None:
    graph_dir = Path(__file__).resolve().parent.parent / "app" / "graph"
    allowed = (
        "app.graph",
        "app.core.errors",
        "app.core.pdf_parser",
        "app.core.vectorstore",  # LangChain document conversion stays at the app boundary.
        "finrag.models",
        "finrag.retrieval",
        "finrag.verification",
        "langchain_core",
        "langgraph",
        "pydantic",
        "typing_extensions",
        "duckduckgo_search",  # Legacy unregistered helper; no chat provider integration.
    )
    for path in graph_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                assert module.split(".")[0] in sys.stdlib_module_names or any(
                    module == name or module.startswith(name + ".") for name in allowed
                ), f"{path.name}: graph must not import settings/provider construction: {module}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"Settings", "get_llm", "resolve_model_roles"}
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"environ", "getenv"}


def test_numeric_verification_imports_without_site_packages_or_code_execution(
    tmp_path: Path,
) -> None:
    package_dir = Path(finrag.verification.__file__).parent
    for path in package_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from finrag.verification import verify_numeric_claims, VerificationStatus
result = verify_numeric_claims('Revenue was 50 VND.', ['Revenue was 50 VND.'])
assert result[0].status is VerificationStatus.SUPPORTED
print('stdlib-only verification passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(finrag.__file__).parent.parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "stdlib-only verification passed"


def test_provider_imports_need_no_sdks_credentials_network_or_global_registry(
    tmp_path: Path,
) -> None:
    # Cold process avoids already-imported integrations hiding eager imports.
    script = """
import importlib
import importlib.abc
import socket
import sys

attempts = []
class BlockIntegrations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'langchain_openai', 'langchain_groq', 'langchain_anthropic',
            'langchain_google_genai', 'langchain_ollama', 'openai', 'groq',
            'anthropic', 'google', 'ollama',
        }:
            attempts.append(fullname)
            raise AssertionError('Provider SDK imported before factory call')

def block_network(*args, **kwargs):
    attempts.append('network')
    raise AssertionError('Network attempted during import')

sys.meta_path.insert(0, BlockIntegrations())
socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network
modules = [importlib.import_module(name) for name in (
    'finrag', 'finrag.providers', 'finrag.providers.registry',
    'finrag.providers.openai', 'finrag.providers.groq',
    'finrag.providers.anthropic', 'finrag.providers.google', 'finrag.providers.ollama',
    'finrag.providers.builtin', 'finrag.models',
)]
from finrag.providers import ChatProviderRegistry, ProviderNotRegisteredError
for module in modules:
    assert not any(isinstance(value, ChatProviderRegistry) for value in vars(module).values())
for name in ('openai', 'groq', 'anthropic', 'google', 'ollama'):
    try:
        ChatProviderRegistry().get(name)
    except ProviderNotRegisteredError:
        pass
    else:
        raise AssertionError('Provider was automatically registered')
assert not attempts, attempts
from finrag.providers.builtin import register_builtin_chat_providers
registry = ChatProviderRegistry()
register_builtin_chat_providers(registry)
for name in ('openai', 'groq', 'anthropic', 'google', 'ollama'):
    assert callable(registry.get(name))
assert not attempts, attempts
print('provider imports are isolated')
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        not in {
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OLLAMA_API_KEY",
            "OLLAMA_BASE_URL",
            "OLLAMA_HOST",
        }
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "provider imports are isolated"


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


def test_retrieval_imports_do_not_load_ml_runtime_or_access_network(tmp_path: Path) -> None:
    script = """
import importlib
import importlib.abc
import socket
import sys

attempts = []
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sentence_transformers', 'transformers', 'torch'}:
            attempts.append(fullname)
            raise AssertionError('ML runtime imported during generic retrieval import')

def block_network(*args, **kwargs):
    attempts.append('network')
    raise AssertionError('Network attempted during retrieval import')

sys.meta_path.insert(0, BlockML())
socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network
import finrag
import finrag.retrieval
import finrag.retrieval.bge
assert not attempts, attempts
print('retrieval imports are lazy')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "retrieval imports are lazy"


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
