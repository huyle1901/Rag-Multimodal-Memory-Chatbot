import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
api_key = os.getenv("OPENAI_API_KEY")

if not VISION_MODEL:
    raise RuntimeError("Missing OPENAI_VISION_MODEL in environment or .env file.")

if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")

client = OpenAI(api_key=api_key)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _get_image_to_text(image_path):
    base64_image = encode_image(image_path)
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    try:
        response = client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this image for PDF retrieval. Focus on document-relevant "
                                "content such as figures, tables, charts, diagrams, equations, and captions."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            max_output_tokens=300,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI image-to-text request failed for '{image_path}': {exc}") from exc

    if response.output_text:
        return response.output_text.strip()

    raise RuntimeError(
        f"OpenAI image-to-text returned no text for '{image_path}'. Response: {response.model_dump()}"
    )


def get_images_to_texts(image_path_list: list):
    assert all([os.path.exists(_) for _ in image_path_list])
    return [_get_image_to_text(_) for _ in image_path_list]
