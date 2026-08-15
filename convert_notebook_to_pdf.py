"""Convert Group-10.ipynb to a PDF report.

Tries the WebPDF exporter first (Chromium via Playwright, no LaTeX needed),
then falls back to the classic LaTeX-based PDF exporter if a TeX toolchain
(MiKTeX/TeX Live: xelatex/pdflatex) is available on PATH.

Setup (pick whichever exporter you plan to rely on):
    pip install nbconvert nbformat nbconvert[webpdf]
    playwright install chromium
  or, for the LaTeX route, install MiKTeX/TeX Live and ensure xelatex is on PATH.

Usage:
    python convert_notebook_to_pdf.py [notebook_path] [output_pdf_path]

    # defaults to Group-10.ipynb next to this script, output Group-10.pdf
    python convert_notebook_to_pdf.py
"""
import sys
from pathlib import Path

import nbformat
from nbconvert import PDFExporter
from traitlets.config import Config

DEFAULT_NOTEBOOK = Path(__file__).with_name("Group-10.ipynb")


def _make_webpdf_exporter():
    from nbconvert.exporters.webpdf import WebPDFExporter

    config = Config()
    config.WebPDFExporter.allow_chromium_download = True
    return WebPDFExporter(config=config)


def convert_notebook_to_pdf(notebook_path: Path, output_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)

    exporters = [
        ("webpdf", _make_webpdf_exporter),
        ("latex", PDFExporter),
    ]

    last_error = None
    for name, make_exporter in exporters:
        try:
            exporter = make_exporter()
            body, _ = exporter.from_notebook_node(notebook)
            output_path.write_bytes(body)
            print(f"PDF written to {output_path} using the '{name}' exporter.")
            return
        except Exception as exc:
            last_error = exc
            print(f"'{name}' exporter failed: {exc}")

    raise RuntimeError(
        "All PDF export methods failed. Either run "
        "'pip install nbconvert[webpdf] && playwright install chromium' "
        "for the WebPDF route, or install a LaTeX distribution "
        "(MiKTeX/TeX Live) with xelatex on PATH for the LaTeX route."
    ) from last_error


def main() -> None:
    args = sys.argv[1:]
    notebook_path = Path(args[0]) if len(args) >= 1 else DEFAULT_NOTEBOOK
    output_path = Path(args[1]) if len(args) >= 2 else notebook_path.with_suffix(".pdf")

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    convert_notebook_to_pdf(notebook_path, output_path)


if __name__ == "__main__":
    main()
