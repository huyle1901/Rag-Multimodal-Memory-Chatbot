from langchain_openai import OpenAIEmbeddings

from config import get_settings


def create_embedding_function() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model)
