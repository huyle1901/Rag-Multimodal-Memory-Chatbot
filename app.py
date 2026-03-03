import base64
import queue
import threading
import time
from pathlib import Path

import streamlit as st

from chat_store import ChatMessage, ChatStore
from config import Settings, get_settings
from llm import stream_answer, summarize_messages
from retrieval import RetrievalResult, retrieve_context

APP_TITLE = "Multimodal RAG Chatbot"
SVG_ICON_PATH = Path("resources/static/1538298822.svg")

st.set_page_config(page_title=APP_TITLE, layout="wide")


@st.cache_resource(show_spinner=False)
def get_store(database_url: str) -> ChatStore:
    store = ChatStore(database_url)
    store.ensure_schema()
    return store


def load_css(file_name: str) -> None:
    with open(file_name, encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    with open(SVG_ICON_PATH, "rb") as file:
        data = file.read()

    b64 = base64.b64encode(data).decode("utf-8")
    image_html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:56px; height:auto;" />'
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-header">
                {image_html}
                <div>
                    <h1>{APP_TITLE}</h1>
                    <p>Chat with your indexed PDFs, persistent history, and summary memory.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(store: ChatStore, settings: Settings):
    st.sidebar.title("Conversations")

    if st.sidebar.button("New chat", use_container_width=True):
        session = store.create_session()
        st.session_state["active_session_id"] = session.id
        st.rerun()

    sessions = store.list_sessions(limit=50)
    if not sessions:
        session = store.create_session()
        sessions = [session]
        st.session_state["active_session_id"] = session.id

    session_lookup = {session.id: session for session in sessions}
    active_session_id = st.session_state.get("active_session_id", sessions[0].id)
    if active_session_id not in session_lookup:
        active_session_id = sessions[0].id

    session_ids = [session.id for session in sessions]
    selected_session_id = st.sidebar.selectbox(
        "Session",
        options=session_ids,
        index=session_ids.index(active_session_id),
        format_func=lambda session_id: session_lookup[session_id].title,
    )
    st.session_state["active_session_id"] = selected_session_id

    current_session = store.get_session(selected_session_id)

    with st.sidebar.expander("Models", expanded=False):
        st.code(
            "\n".join(
                [
                    f"Chat: {settings.chat_model}",
                    f"Summary: {settings.summary_model}",
                    f"Embedding: {settings.embedding_model}",
                    f"Vision: {settings.openai_vision_model or 'not set'}",
                    f"Image provider: {settings.image_to_text_provider}",
                ]
            )
        )

    if current_session and current_session.summary:
        with st.sidebar.expander("Conversation summary", expanded=False):
            st.write(current_session.summary)

    return current_session


def render_messages(messages: list[ChatMessage]) -> None:
    if not messages:
        st.info("Start a new conversation by asking a question about your indexed PDFs.")
        return

    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def handle_user_turn(store: ChatStore, session_id: str, prompt: str, settings: Settings) -> None:
    session = store.get_session(session_id)
    if session is None:
        raise RuntimeError(f"Session '{session_id}' was not found.")

    if session.title == "New chat":
        store.rename_session(session_id, build_session_title(prompt))

    store.add_message(session_id, "user", prompt)

    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()

        retrieval_result = load_retrieval_result(prompt, assistant_placeholder)
        history = store.get_messages(session_id)
        recent_messages = history[-(settings.recent_history_messages + 1):-1]
        session = store.get_session(session_id)
        answer = stream_answer_to_placeholder(
            assistant_placeholder=assistant_placeholder,
            question=prompt,
            context=retrieval_result.context_text,
            conversation_summary=session.summary if session else "",
            recent_messages=recent_messages,
        )

    store.add_message(session_id, "assistant", answer)

    refresh_summary(store, session_id, settings)


def refresh_summary(store: ChatStore, session_id: str, settings: Settings) -> None:
    session = store.get_session(session_id)
    if session is None:
        return

    messages = store.get_messages(session_id)
    summarizable_count = max(0, len(messages) - settings.recent_history_messages)
    pending_count = summarizable_count - session.summary_message_count
    if pending_count < settings.summary_trigger_messages:
        return

    new_messages = messages[session.summary_message_count:summarizable_count]
    updated_summary = summarize_messages(
        existing_summary=session.summary,
        new_messages=new_messages,
        max_summary_tokens=settings.max_summary_tokens,
    )
    store.update_summary(session_id, updated_summary, summarizable_count)


def build_session_title(prompt: str) -> str:
    trimmed = " ".join(prompt.split())
    return trimmed[:60] or "New chat"


def load_retrieval_result(prompt: str, assistant_placeholder) -> RetrievalResult:
    result_holder: dict[str, RetrievalResult | Exception] = {}

    def worker() -> None:
        try:
            result_holder["result"] = retrieve_context(prompt)
        except FileNotFoundError as exc:
            result_holder["result"] = RetrievalResult(
                context_text=f"Vector database warning: {exc}",
                sources=[],
                scores=[],
            )
        except Exception as exc:
            result_holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    frame = 0
    while thread.is_alive():
        assistant_placeholder.markdown(_typing_frame(frame))
        time.sleep(0.25)
        frame += 1

    thread.join()

    error = result_holder.get("error")
    if error:
        raise error

    result = result_holder.get("result")
    if isinstance(result, RetrievalResult):
        return result

    return RetrievalResult(context_text="", sources=[], scores=[])


def stream_answer_to_placeholder(
    assistant_placeholder,
    question: str,
    context: str,
    conversation_summary: str,
    recent_messages: list[ChatMessage],
) -> str:
    chunk_queue: queue.Queue[tuple[str, str | Exception | None]] = queue.Queue()

    def worker() -> None:
        try:
            for chunk in stream_answer(
                question=question,
                context=context,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
            ):
                chunk_queue.put(("chunk", chunk))
            chunk_queue.put(("done", None))
        except Exception as exc:
            chunk_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    answer_parts: list[str] = []
    frame = 0
    while True:
        try:
            event, payload = chunk_queue.get(timeout=0.25)
        except queue.Empty:
            current_answer = "".join(answer_parts)
            assistant_placeholder.markdown(
                _stream_frame(current_answer, frame),
            )
            frame += 1
            continue

        if event == "chunk" and isinstance(payload, str):
            answer_parts.append(payload)
            assistant_placeholder.markdown("".join(answer_parts) + "▌")
            continue

        if event == "error" and isinstance(payload, Exception):
            raise payload

        if event == "done":
            break

    answer = "".join(answer_parts).strip()
    if not answer:
        answer = "I could not generate a response from the available context."
    assistant_placeholder.markdown(answer)
    return answer


def _typing_frame(frame: int) -> str:
    return "Thinking" + "." * ((frame % 3) + 1)


def _stream_frame(current_answer: str, frame: int) -> str:
    if current_answer:
        return current_answer + "▌"
    return _typing_frame(frame)


def main() -> None:
    settings = get_settings()
    load_css("style.css")
    render_header()

    try:
        store = get_store(settings.database_url)
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        st.stop()

    session = render_sidebar(store, settings)
    if session is None:
        st.error("Unable to initialize a chat session.")
        st.stop()

    messages = store.get_messages(session.id)
    render_messages(messages)

    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_user_turn(store, session.id, prompt, settings)
        st.rerun()


if __name__ == "__main__":
    main()
