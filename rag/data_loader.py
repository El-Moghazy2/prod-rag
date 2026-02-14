"""Document loading module - PDF extraction from Covestro Safety Data Sheets."""

import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _extract_full_text(pdf_path: str | Path) -> str:
    """Extract full text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text content of the PDF.
    """
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def _extract_metadata(text: str) -> dict:
    """Extract product metadata from SDS text using regex.

    Args:
        text: Full text of the SDS document.

    Returns:
        Dict with product_name, material_number, chemical_family, and use.
    """
    metadata = {
        "product_name": "",
        "material_number": "",
        "chemical_family": "",
        "use": "",
    }

    # Product Name - appears after "Product Name:" label
    match = re.search(r"Product Name:\s*\n?\s*(.+?)(?:\n|Material Number)", text)
    if match:
        metadata["product_name"] = match.group(1).strip()

    # Material Number
    match = re.search(r"Material Number:\s*\n?\s*(\d+)", text)
    if match:
        metadata["material_number"] = match.group(1).strip()

    # Chemical Family
    match = re.search(r"Chemical Family:\s*\n?\s*(.+?)(?:\n|Use:)", text)
    if match:
        metadata["chemical_family"] = match.group(1).strip()

    # Use
    match = re.search(r"Use:\s*\n?\s*(.+?)(?:\n\s*\n|\n\d+\.)", text, re.DOTALL)
    if match:
        use_text = match.group(1).strip()
        # Clean up multi-line use descriptions
        use_text = re.sub(r"\s+", " ", use_text)
        metadata["use"] = use_text

    return metadata


def load_pdf(pdf_path: str | Path) -> Document:
    """Load a single SDS PDF and return one Document with the full text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A single Document with the full PDF text and metadata.
    """
    pdf_path = Path(pdf_path)
    text = _extract_full_text(pdf_path)
    metadata = _extract_metadata(text)

    # Prepend product context so chunks remain identifiable after splitting
    header = f"Product: {metadata['product_name']} (Material #{metadata['material_number']})\n\n"
    return Document(
        page_content=header + text,
        metadata={
            "product_name": metadata["product_name"],
            "material_number": metadata["material_number"],
            "chemical_family": metadata["chemical_family"],
            "use": metadata["use"],
            "source": pdf_path.name,
        },
    )


def get_documents(data_dir: str | Path | None = None) -> list[Document]:
    """Load all SDS PDFs from the data directory.

    Args:
        data_dir: Directory containing PDF files. Defaults to project data/ folder.

    Returns:
        List of Document objects with content and metadata.
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    documents = []

    pdf_files = sorted(data_path.glob("*.pdf"))
    for pdf_path in pdf_files:
        doc = load_pdf(pdf_path)
        documents.append(doc)
        print(f"  Loaded {pdf_path.name}: {doc.metadata['product_name']}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents
