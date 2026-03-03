# Multimodal RAG Chatbot

## Overview

This project uses Multimodal RAG (Retrieval-Augmented Generation) to extract, process, index, and chat with PDF documents. It combines OCR-style image extraction, vector search, and LLM-powered generation in a Streamlit chatbot. The chatbot keeps persistent conversation history in PostgreSQL and maintains a rolling summary so follow-up questions retain useful context without replaying the full transcript every turn.

## Features

- **PDF and Image Processing**: Automated extraction of text and imagery from PDF files, making all accessible information machine-readable.
- **Adaptive Image-to-Text Conversion**: Using OpenAI API for converting images to descriptive text, enabling deep analysis and indexing.
- **Advanced Text and Image Indexing**: Leverages the Chroma vector database for embedding and efficiently searching through text and image-derived content.
- **Interactive Chatbot Interface**: A Streamlit chatbot with persistent sessions, stored history, and summary memory.
- **Persistent Memory Layer**: Chat sessions and message history are stored in PostgreSQL, with an LLM-generated summary used as compact long-term context.
- **Explicit Model Configuration**: Chat, summary, embedding, and vision models are configured with environment variables.

## Technical Components

- **LangChain**: Manages document loading, text extraction, splitting, and the integration of complex NLP workflows.
- **OpenAI API**: Provides high-accuracy image-to-text conversion through OpenAI's models, capable of understanding and generating descriptions for complex visual content.
- **Chroma**: Utilizes a vector-based approach for storing and retrieving text embeddings, ensuring quick and relevant similarity searches.
- **Streamlit**: Provides the chat interface for multi-turn conversations.
- **PostgreSQL**: Stores sessions, messages, and conversation summaries.

## Machine Learning Concepts

### Optical Character Recognition (OCR)
Facilitates the conversion of textual content from images into machine-encoded text for further digital processing.

### Natural Language Processing (NLP)
Employs algorithms to analyze, understand, and generate human language from the text extracted from documents and images.

### Embeddings and Vector Databases
Converts text into high-dimensional vectors or embeddings, enabling semantic similarity comparisons.

### Retrieval-Augmented Generation (RAG)
Improves the quality and relevance of text responses by basing them on content extracted from multimodal data sources.

### Multimodal Learning
Processes and analyzes information from varied data types, such as text and images, to improve overall understanding and contextually relevant responses.

## Setup

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`. If you want LLaVA support, also install `requirements.llava.txt`.
3. Copy `.env.example` to `.env` and set the required environment variables, especially `OPENAI_API_KEY`.
4. Start PostgreSQL locally, or use Docker Compose.


## Environment Variables

- `CHAT_MODEL`: chat completion model used to answer the user.
- `SUMMARY_MODEL`: model used to compress older turns into summary memory.
- `EMBEDDING_MODEL`: embedding model used for both indexing and retrieval.
- `OPENAI_VISION_MODEL`: vision-capable model used when `IMAGE_TO_TEXT_PROVIDER=openai`.
- `IMAGE_TO_TEXT_PROVIDER`: `openai` or `llava`.
- `DATABASE_URL`: PostgreSQL connection string for chat history and summary storage.

## Docker

- Start the app and PostgreSQL: `docker compose up --build`
- Build or refresh the vector database: `docker compose run --rm app python create_database.py`
- Open the chatbot at `http://localhost:8501`
