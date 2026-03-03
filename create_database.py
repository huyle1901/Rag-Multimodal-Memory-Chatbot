import logging
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import get_settings
from embeddings import create_embedding_function
from image_to_text import get_images_to_texts

logging.getLogger().setLevel(logging.INFO)

SETTINGS = get_settings()
CHROMA_PATH = SETTINGS.chroma_path
DATA_PATH = SETTINGS.data_path
GLOB_PDF_PATTERN = "*.pdf"
EXTRACTED_IMG_FOLDER = Path(__file__).resolve().parent / "data" / "extracted_imgs"
EXTRACTED_IMG_FOLDER.mkdir(parents=True, exist_ok=True)


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks_from_text = split_text(documents)

    pdf_paths = list(dict.fromkeys(doc.metadata["source"] for doc in documents))
    chunks_from_images = get_texts_from_images(pdf_paths)

    chunks = chunks_from_text + chunks_from_images
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH, glob=GLOB_PDF_PATTERN)
    documents = loader.load()
    print(f"{len(documents)} documents loaded successfully!")
    return documents


def split_text(documents: list[Document]):
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def get_texts_from_images(pdf_paths: list):
    image_path_list = []
    image_metadata_list = []
    for each_pdf_path in pdf_paths:
        pdf_obj = PdfReader(each_pdf_path)
        pdf_name = Path(each_pdf_path).stem
        for page_index, page in enumerate(pdf_obj.pages, start=1):
            for img_idx, image in enumerate(page.images):
                image_suffix = Path(image.name).suffix or ".bin"
                file_name = f"{pdf_name}_page_{page_index}_img_{img_idx}{image_suffix}"
                extracted_img_path = EXTRACTED_IMG_FOLDER / file_name
                with open(extracted_img_path, "wb") as fp:
                    fp.write(image.data)
                    image_path_list.append(str(extracted_img_path))
                    image_metadata_list.append(
                        f"{each_pdf_path} | PageNum: {page.page_number} | img_idx: {img_idx}"
                    )

    logging.info(f"Number of images {len(image_path_list)} extracted")
    if not image_path_list:
        return []

    img_descriptions = get_images_to_texts(image_path_list)

    chunks = [
        Document(page_content=img_descriptions[i], metadata={"source": image_metadata_list[i]})
        for i in range(len(img_descriptions))
    ]

    return chunks


def save_to_chroma(chunks: list[Document]):
    if not chunks:
        raise RuntimeError("No chunks were generated from the source documents.")

    # Clear the directory contents without deleting the volume mount root.
    clear_directory_contents(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks,
        create_embedding_function(),
        persist_directory=CHROMA_PATH,
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def clear_directory_contents(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return

    for entry_name in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


if __name__ == "__main__":
    main()
