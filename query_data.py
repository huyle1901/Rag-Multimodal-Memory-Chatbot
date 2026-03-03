import argparse

from llm import answer_question
from retrieval import retrieve_context


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        result = retrieve_context(query_text)
    except FileNotFoundError as exc:
        print(exc)
        return

    response_text = answer_question(
        question=query_text,
        context=result.context_text,
        conversation_summary="",
        recent_messages=[],
    )
    formatted_response = f"Response:\n{response_text}\n\nSources:\n" + "\n".join(result.sources)
    print(formatted_response)


if __name__ == "__main__":
    main()
