from __future__ import annotations

from collections.abc import Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from chat_store import ChatMessage
from config import get_settings

ANSWER_SYSTEM_PROMPT = """
You are a document-grounded assistant for a multimodal RAG application.
Use the retrieved document context as the primary source of truth.
Use the conversation summary and recent history only to resolve references and maintain continuity.
If the retrieved context is missing or insufficient, say that clearly instead of inventing facts.
Answer completely. Do not default to one short sentence when the context supports a fuller explanation.
When the user asks an informational question, give a direct answer first, then expand with the most relevant details,
key evidence, implications, or examples from the retrieved context.
Use bullet points when they improve clarity.
""".strip()

ANSWER_USER_PROMPT = """
Conversation summary:
{conversation_summary}

Recent conversation:
{recent_history}

Retrieved document context:
{context}

User question:
{question}

Write a response that is complete and specific enough to stand on its own.
""".strip()

SUMMARY_SYSTEM_PROMPT = """
You maintain a compressed memory of a conversation.
Write a factual summary that preserves user intent, unresolved questions, named entities, constraints,
and decisions already made. Do not include filler. Keep it compact and useful for future turns.
""".strip()

SUMMARY_USER_PROMPT = """
Existing summary:
{existing_summary}

New conversation messages:
{new_messages}

Update the summary so it includes the important information from the new messages.
Limit the response to roughly {max_summary_tokens} tokens.
""".strip()


def create_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.chat_model, temperature=0.2)


def create_summary_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.summary_model)


def answer_question(
    question: str,
    context: str,
    conversation_summary: str,
    recent_messages: list[ChatMessage],
) -> str:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM_PROMPT), ("human", ANSWER_USER_PROMPT)]
    )
    model = create_chat_model()
    response = model.invoke(
        prompt_template.format_messages(
            conversation_summary=conversation_summary or "No conversation summary yet.",
            recent_history=_format_messages(recent_messages),
            context=context or "No relevant context retrieved from the vector database.",
            question=question,
        )
    )
    return response.content.strip()


def stream_answer(
    question: str,
    context: str,
    conversation_summary: str,
    recent_messages: list[ChatMessage],
) -> Iterator[str]:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM_PROMPT), ("human", ANSWER_USER_PROMPT)]
    )
    model = create_chat_model()
    for chunk in model.stream(
        prompt_template.format_messages(
            conversation_summary=conversation_summary or "No conversation summary yet.",
            recent_history=_format_messages(recent_messages),
            context=context or "No relevant context retrieved from the vector database.",
            question=question,
        )
    ):
        text = _chunk_to_text(chunk.content)
        if text:
            yield text


def summarize_messages(
    existing_summary: str,
    new_messages: list[ChatMessage],
    max_summary_tokens: int,
) -> str:
    if not new_messages:
        return existing_summary

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_SYSTEM_PROMPT), ("human", SUMMARY_USER_PROMPT)]
    )
    model = create_summary_model()
    response = model.invoke(
        prompt_template.format_messages(
            existing_summary=existing_summary or "No summary yet.",
            new_messages=_format_messages(new_messages),
            max_summary_tokens=max_summary_tokens,
        )
    )
    return response.content.strip()


def _format_messages(messages: list[ChatMessage]) -> str:
    if not messages:
        return "No recent messages."
    return "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)


def _chunk_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "".join(parts)
    return ""
