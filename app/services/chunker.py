"""
Code-aware chunking service.

- Python files: parsed with the built-in `ast` module — extracts top-level
  functions and classes as self-contained chunks (full decorators + body).
- All other languages: 60-line sliding window with 10-line overlap plus a
  regex heuristic that labels chunks with the first function/class name found.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c",
    ".cs", ".rb", ".php", ".swift", ".kt",
}

_FUNC_PATTERN = re.compile(
    r"^\s*(public|private|protected|async|static|def|func|fn|function|class|interface)\b",
    re.MULTILINE,
)


@dataclass
class CodeChunk:
    content: str
    file_path: str
    chunk_type: str
    name: str | None
    start_line: int
    end_line: int
    chunk_id: str = field(init=False)

    def __post_init__(self):
        label = self.name or f"L{self.start_line}"
        safe = self.file_path.replace("/", "_").replace("\\", "_")
        self.chunk_id = f"{safe}::{label}::{self.start_line}"


class CodeChunker:

    def chunk_directory(self, root: str | Path) -> list[CodeChunk]:
        root = Path(root)
        chunks: list[CodeChunk] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    chunks.extend(self._chunk_file(path, root))
                except Exception:
                    pass  # skip unreadable / binary-looking files
        return chunks

    def chunk_file(self, path: str | Path, root: str | Path | None = None) -> list[CodeChunk]:
        p = Path(path)
        return self._chunk_file(p, Path(root) if root else p.parent)

    # ------------------------------------------------------------------ #

    def _chunk_file(self, path: Path, root: Path) -> list[CodeChunk]:
        rel_path = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8", errors="replace")
        return self._chunk_python(source, rel_path) if path.suffix == ".py" else self._chunk_generic(source, rel_path)

    def _chunk_python(self, source: str, rel_path: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._chunk_generic(source, rel_path)

        lines = source.splitlines()
        chunks: list[CodeChunk] = []
        covered: set[int] = set()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # skip nodes nested inside another function/class
            if any(
                isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                for parent in ast.walk(tree)
                if hasattr(parent, "body") and node in getattr(parent, "body", [])
            ):
                continue

            start, end = node.lineno - 1, node.end_lineno
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(CodeChunk(
                content="\n".join(lines[start:end]),
                file_path=rel_path,
                chunk_type=chunk_type,
                name=node.name,
                start_line=node.lineno,
                end_line=end,
            ))
            covered.update(range(start, end))

        # capture module-level code not inside any definition
        module_content = "\n".join(
            line for i, line in enumerate(lines) if i not in covered
        ).strip()
        if module_content:
            chunks.append(CodeChunk(
                content=module_content,
                file_path=rel_path,
                chunk_type="module",
                name=None,
                start_line=1,
                end_line=len(lines),
            ))

        if not chunks:
            chunks.append(CodeChunk(
                content=source, file_path=rel_path, chunk_type="module",
                name=None, start_line=1, end_line=len(lines),
            ))
        return chunks

    def _chunk_generic(self, source: str, rel_path: str, window: int = 60, overlap: int = 10) -> list[CodeChunk]:
        lines = source.splitlines()
        if not lines:
            return []
        chunks: list[CodeChunk] = []
        start = 0
        while start < len(lines):
            end = min(start + window, len(lines))
            content = "\n".join(lines[start:end])
            name = self._extract_name_generic(content)
            chunks.append(CodeChunk(
                content=content, file_path=rel_path,
                chunk_type="function" if name else "block",
                name=name, start_line=start + 1, end_line=end,
            ))
            start += window - overlap
        return chunks

    @staticmethod
    def _extract_name_generic(block: str) -> str | None:
        m = _FUNC_PATTERN.search(block)
        if not m:
            return None
        rest = block[m.end():].strip()
        nm = re.match(r"(\w+)", rest)
        return nm.group(1) if nm else None
