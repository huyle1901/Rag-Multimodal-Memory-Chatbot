from config import get_settings


def get_images_to_texts(image_path_list: list[str]) -> list[str]:
    settings = get_settings()

    if settings.image_to_text_provider == "llava":
        from img2txt.llava_local_img import get_images_to_texts as image_to_text_impl
    else:
        from img2txt.openai_local_img import get_images_to_texts as image_to_text_impl

    return image_to_text_impl(image_path_list)
