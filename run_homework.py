#!/usr/bin/env python3
"""Execute the completed homework notebooks with the active Python environment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

REPOSITORY_DIR = Path(__file__).resolve().parent
NOTEBOOKS = {
    "homework_1": {
        "q4": "ML1_HW1_Q4.ipynb",
        "q5": "ML1_HW1_Q5.ipynb",
    }
}


@contextmanager
def temporary_runtime_environment():
    """Keep runtime caches writable and restore the caller's environment afterward."""
    with TemporaryDirectory(prefix="data37711-notebooks-") as runtime_dir:
        runtime_root = Path(runtime_dir)
        overrides = {
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
            "JUPYTER_RUNTIME_DIR": str(runtime_root / "jupyter"),
            "IPYTHONDIR": str(runtime_root / "ipython"),
            "MPLBACKEND": "Agg",
        }
        previous_values = {name: os.environ.get(name) for name in overrides}
        for name, value in overrides.items():
            if name != "MPLBACKEND":
                Path(value).mkdir(parents=True, exist_ok=True)
            os.environ[name] = value
        try:
            yield
        finally:
            for name, previous in previous_values.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous


def execute_notebook(homework: str, question: str, timeout: int) -> Path:
    source_path = REPOSITORY_DIR / homework / question / NOTEBOOKS[homework][question]
    output_dir = REPOSITORY_DIR / "outputs" / homework / question
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    notebook = nbformat.read(source_path, as_version=4)
    # Partial results must contain only outputs from this run, never stored old results.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    print(f"Running {homework}/{question}: {source_path.name}", flush=True)
    with temporary_runtime_environment():
        manager = KernelManager(kernel_name="python3")
        # Use the interpreter running this script, including an activated .venv.
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(
            notebook,
            timeout=timeout,
            km=manager,
            resources={"metadata": {"path": str(source_path.parent)}},
        )
        try:
            # Notebook-local data paths, figures, and numeric results remain valid.
            client.execute(cwd=str(source_path.parent))
        finally:
            # Preserve completed or partial outputs without rewriting the source notebook.
            nbformat.write(notebook, output_path)
            try:
                if client.kc is not None:
                    client.kc.stop_channels()
                    client.kc = None
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            finally:
                manager.cleanup_resources()
                # Release references while Jupyter modules are still loaded.
                client.km = None
                client = None
                manager = None
                gc.collect()
    print(f"Saved freshly executed notebook: {output_path.relative_to(REPOSITORY_DIR)}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("homework", choices=NOTEBOOKS, help="Homework directory to run")
    parser.add_argument(
        "--question", choices=["q4", "q5", "all"], default="all",
        help="Run one question or both (default: all)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Maximum seconds allowed per notebook cell (default: 600)",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    questions = list(NOTEBOOKS[args.homework]) if args.question == "all" else [args.question]
    for question in questions:
        try:
            execute_notebook(args.homework, question, args.timeout)
        except Exception as error:
            print(
                f"Execution failed for {args.homework}/{question}: "
                f"{type(error).__name__}: {error}\n"
                "Partial execution output, if available, was saved under outputs/.",
                file=sys.stderr,
            )
            return 1
    print("All selected notebooks completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
