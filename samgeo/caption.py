"""Image captioning and feature extraction module using BLIP and spaCy.

This module provides functionality to generate captions for images using the BLIP
model and extract relevant features from the captions using spaCy NLP.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.utils import logging as hf_logging
from PIL import Image
import torch
import spacy
import importlib
from spacy.cli import download
import requests
from io import BytesIO
from typing import List, Optional, Union, Tuple
from .common import get_device, show_image

# ---------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------

AERIAL_FEATURES_URL = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/aerial_features.json"

hf_logging.set_verbosity_error()  # silence HF load reports

# Default model names
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# ---------------------------------------------------------------------
# 1. Ensure spaCy model
# ---------------------------------------------------------------------


def ensure_spacy_model(model_name: str = DEFAULT_SPACY_MODEL) -> None:
    """Download spaCy model only if it's missing.

    Args:
        model_name: Name of the spaCy model to ensure is installed.
            Defaults to "en_core_web_sm".
    """
    try:
        importlib.import_module(model_name)
    except ImportError:
        print(f"↓ Model '{model_name}' not found. Installing...")
        try:
            download(model_name)
            print(f"✓ Model '{model_name}' installed.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download spaCy model '{model_name}'. "
                f"You may need to install it manually with: python -m spacy download {model_name}. "
                f"Error: {e}"
            ) from e


# ---------------------------------------------------------------------
# 2. Load aerial feature vocabulary from Hugging Face
# ---------------------------------------------------------------------


def load_aerial_feature_vocab(url: str = AERIAL_FEATURES_URL) -> List[str]:
    """Load the nested aerial_features.json and flatten to a list of feature keys.

    Args:
        url: URL to the aerial features JSON file. Defaults to the
            Hugging Face hosted version.

    Returns:
        Sorted list of feature keys extracted from the JSON file.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    features = set()

    # data is two-level nested: {category: list OR {subcat: list}}
    for _, val in data.items():
        if isinstance(val, list):
            features.update(val)
        elif isinstance(val, dict):
            for _, sublist in val.items():
                if isinstance(sublist, list):
                    features.update(sublist)

    return sorted(features)


RAW_FEATURES = load_aerial_feature_vocab()

# Canonical mapping: phrase ("parking lot") -> key ("parking_lot")
FEATURE_PHRASE_TO_CANON = {feat.replace("_", " "): feat for feat in RAW_FEATURES}

# Split into single- and multi-word phrases for matching
SINGLE_WORD_FEATURES = {
    phrase: canon
    for phrase, canon in FEATURE_PHRASE_TO_CANON.items()
    if " " not in phrase
}
MULTIWORD_FEATURES = {
    phrase: canon for phrase, canon in FEATURE_PHRASE_TO_CANON.items() if " " in phrase
}

# ---------------------------------------------------------------------
# 3. Large-scale features to ignore (lemmas)
# ---------------------------------------------------------------------

LARGE_SCALE = {
    "city",
    "town",
    "village",
    "country",
    "continent",
    "region",
    "landscape",
    "area",
    "scene",
    "view",
    "field",
    "terrain",
    "forest",
    "mountain",
    "valley",
    "coast",
    "shore",
    "horizon",
    "suburb",
    "district",
    "neighborhood",
}

# ---------------------------------------------------------------------
# 4. Helper: load image from path or URL
# ---------------------------------------------------------------------


def load_image(source: Union[str, Image.Image]) -> Image.Image:
    """Load a PIL image from various sources.

    Supports loading from local file paths, HTTP(S) URLs, or returns
    the image directly if it's already a PIL.Image.Image.

    Args:
        source: The image source. Can be a local file path (str),
            an HTTP(S) URL (str), or an existing PIL Image object.

    Returns:
        PIL Image object converted to RGB mode.

    Raises:
        TypeError: If the source type is not supported.
        requests.HTTPError: If downloading from URL fails.
        FileNotFoundError: If local file path doesn't exist.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, stream=True)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            return Image.open(source).convert("RGB")

    raise TypeError(f"Unsupported image source type: {type(source)}")


# ---------------------------------------------------------------------
# 5. ImageCaptioner class
# ---------------------------------------------------------------------


class ImageCaptioner:
    """Image captioning and feature extraction using BLIP and spaCy.

    This class provides functionality to generate captions for images using
    the BLIP model and extract relevant features from the captions using
    spaCy NLP processing.

    Args:
        blip_model_name: Name or path of the BLIP model to use for captioning.
            Defaults to "Salesforce/blip-image-captioning-base".
        spacy_model_name: Name of the spaCy model to use for NLP processing.
            Defaults to "en_core_web_sm".
        device: Device to run the BLIP model on. If None, automatically
            detects the best available device (CUDA, MPS, or CPU).

    Attributes:
        blip_model_name: The name of the loaded BLIP model.
        spacy_model_name: The name of the loaded spaCy model.
        device: The device the model is running on.
        processor: The BLIP processor for image preprocessing.
        blip_model: The BLIP model for caption generation.
        nlp: The spaCy NLP pipeline.

    Example:
        >>> captioner = ImageCaptioner()
        >>> caption, features = captioner.analyze("path/to/image.jpg")
        >>> print(caption)
        "an aerial view of a parking lot with cars"
        >>> print(features)
        ["parking_lot", "car"]
    """

    def __init__(
        self,
        blip_model_name: str = DEFAULT_BLIP_MODEL,
        spacy_model_name: str = DEFAULT_SPACY_MODEL,
        device: Optional[str] = None,
    ):
        """Initialize the ImageCaptioner with specified models.

        Args:
            blip_model_name: Name or path of the BLIP model to use.
                Defaults to "Salesforce/blip-image-captioning-base".
            spacy_model_name: Name of the spaCy model to use.
                Defaults to "en_core_web_sm".
            device: Device to run the model on ('cuda', 'mps', 'cpu').
                If None, automatically detects the best available device.
        """
        self.blip_model_name = blip_model_name
        self.spacy_model_name = spacy_model_name
        self.device = device if device else get_device()

        # Load spaCy model
        ensure_spacy_model(spacy_model_name)
        self.nlp = spacy.load(spacy_model_name)

        # Load BLIP model
        self.processor = BlipProcessor.from_pretrained(
            blip_model_name,
            use_fast=True,
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name
        ).to(self.device)
        self.blip_model.eval()

    @torch.inference_mode()
    def generate_caption(self, image_source: Union[str, Image.Image]) -> str:
        """Generate a caption for the given image.

        Args:
            image_source: The image to caption. Can be a local file path,
                an HTTP(S) URL, or a PIL Image object.

        Returns:
            Generated caption string describing the image content.

        Example:
            >>> captioner = ImageCaptioner()
            >>> caption = captioner.generate_caption("path/to/aerial.jpg")
            >>> print(caption)
            "an aerial view of a building with a parking lot"
        """
        img = load_image(image_source)
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def extract_features(
        self,
        caption: str,
        include_features: Optional[Union[str, List[str]]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> List[str]:
        """Extract features from a caption using NLP processing.

        Uses spaCy to parse the caption and extract relevant noun features
        based on the provided inclusion/exclusion criteria.

        Args:
            caption: The caption text to extract features from.
            include_features: Controls which features to extract:
                - None: Extract any noun (excluding large-scale terms
                  and custom excludes).
                - "default" or ["default"]: Use the aerial_features.json
                  vocabulary for matching.
                - List of strings: Custom allowed features (with or without
                  underscores).
            exclude_features: List of noun lemmas to exclude in addition
                to the built-in large-scale terms.

        Returns:
            Sorted list of extracted feature names (canonical keys or
            noun lemmas).

        Example:
            >>> captioner = ImageCaptioner()
            >>> features = captioner.extract_features(
            ...     "a parking lot with several cars",
            ...     include_features=["default"]
            ... )
            >>> print(features)
            ["car", "parking_lot"]
        """
        doc = self.nlp(caption)
        detected = set()

        # ----------------------- exclusions -----------------------
        active_exclude = set(LARGE_SCALE)
        if exclude_features:
            active_exclude.update(
                ex.lower().replace("_", " ") for ex in exclude_features
            )

        # Normalize include_features semantics
        use_default_vocab = False
        user_include_list: Optional[List[str]] = None

        if include_features is None:
            use_default_vocab = False
        else:
            # allow include_features="default" or ["default"]
            if isinstance(include_features, str):
                include_features = [include_features]

            if any(f.lower() == "default" for f in include_features):
                use_default_vocab = True
            else:
                user_include_list = include_features

        # Build maps for matching
        if use_default_vocab:
            single_map = SINGLE_WORD_FEATURES
            multi_map = MULTIWORD_FEATURES
        elif user_include_list:
            norm = [f.lower().replace("_", " ") for f in user_include_list]
            single_map = {
                phrase: phrase.replace(" ", "_") for phrase in norm if " " not in phrase
            }
            multi_map = {
                phrase: phrase.replace(" ", "_") for phrase in norm if " " in phrase
            }
        else:
            single_map = None
            multi_map = None

        # ----------------------- multi-word (noun chunks) -----------------------
        if use_default_vocab or user_include_list:
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                chunk_lemma = " ".join(tok.lemma_.lower() for tok in chunk)

                for candidate in {chunk_text, chunk_lemma}:
                    if multi_map and candidate in multi_map:
                        detected.add(multi_map[candidate])

        # ----------------------- single-word nouns -----------------------
        original_include = include_features
        for token in doc:
            if token.pos_ != "NOUN":
                continue

            lemma = token.lemma_.lower()
            if lemma in active_exclude:
                continue

            # Case: no include list → accept any noun lemma
            if original_include is None:
                detected.add(lemma)
            else:
                if single_map and lemma in single_map:
                    detected.add(single_map[lemma])

        # ----------------------- fallback if using include list -----------------------
        if (use_default_vocab or user_include_list) and not detected:
            fallback = {
                token.lemma_.lower()
                for token in doc
                if token.pos_ == "NOUN" and token.lemma_.lower() not in active_exclude
            }
            return sorted(fallback)

        return sorted(detected)

    @torch.inference_mode()
    def analyze(
        self,
        image_source: Union[str, Image.Image],
        include_features: Optional[Union[str, List[str]]] = None,
        exclude_features: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Analyze an image by generating a caption and extracting features.

        This is the main entry point that combines caption generation and
        feature extraction into a single call.

        Args:
            image_source: The image to analyze. Can be a local file path,
                an HTTP(S) URL, or a PIL Image object.
            include_features: Controls which features to extract:
                - None: Extract any noun (excluding large-scale terms
                  and custom excludes).
                - "default" or ["default"]: Use the aerial_features.json
                  vocabulary for matching.
                - List of strings: Custom allowed features (with or without
                  underscores).
            exclude_features: List of noun lemmas to exclude in addition
                to the built-in large-scale terms.

        Returns:
            A tuple containing:
                - caption: The BLIP-generated caption string.
                - features: Sorted list of extracted feature names.

        Example:
            >>> captioner = ImageCaptioner()
            >>> caption, features = captioner.analyze(
            ...     "https://example.com/aerial.jpg",
            ...     include_features=["default"],
            ...     exclude_features=["building"]
            ... )
            >>> print(caption)
            "an aerial view of a residential area"
            >>> print(features)
            ["house", "road", "tree"]
        """
        caption = self.generate_caption(image_source)
        features = self.extract_features(
            caption,
            include_features=include_features,
            exclude_features=exclude_features,
        )
        return caption, features


# ---------------------------------------------------------------------
# 6. Module-level convenience functions (backward compatibility)
# ---------------------------------------------------------------------

# Lazy-loaded default captioner instance
_default_captioner: Optional[ImageCaptioner] = None


def _get_default_captioner() -> ImageCaptioner:
    """Get or create the default ImageCaptioner instance.

    Returns:
        The default ImageCaptioner instance with default model settings.
    """
    global _default_captioner
    if _default_captioner is None:
        _default_captioner = ImageCaptioner()
    return _default_captioner


def extract_features_from_caption(
    caption: str,
    include_features: Optional[Union[str, List[str]]] = None,
    exclude_features: Optional[List[str]] = None,
) -> List[str]:
    """Extract features from a caption using NLP processing.

    This is a convenience function that uses the default ImageCaptioner
    instance. For more control over models, create an ImageCaptioner
    instance directly.

    Args:
        caption: The caption text to extract features from.
        include_features: Controls which features to extract:
            - None: Extract any noun (excluding large-scale terms
              and custom excludes).
            - "default" or ["default"]: Use the aerial_features.json
              vocabulary for matching.
            - List of strings: Custom allowed features (with or without
              underscores).
        exclude_features: List of noun lemmas to exclude in addition
            to the built-in large-scale terms.

    Returns:
        Sorted list of extracted feature names (canonical keys or
        noun lemmas).

    Example:
        >>> features = extract_features_from_caption(
        ...     "a parking lot with several cars",
        ...     include_features=["default"]
        ... )
        >>> print(features)
        ["car", "parking_lot"]
    """
    captioner = _get_default_captioner()
    return captioner.extract_features(
        caption,
        include_features=include_features,
        exclude_features=exclude_features,
    )


@torch.inference_mode()
def blip_analyze_image(
    image_source: Union[str, Image.Image],
    include_features: Optional[Union[str, List[str]]] = None,
    exclude_features: Optional[List[str]] = None,
    blip_model_name: Optional[str] = None,
    spacy_model_name: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Analyze an image by generating a caption and extracting features.

    This is a convenience function that provides the full pipeline for
    image analysis. For repeated use or custom model configurations,
    consider creating an ImageCaptioner instance directly.

    Args:
        image_source: The image to analyze. Can be a local file path,
            an HTTP(S) URL, or a PIL Image object.
        include_features: Controls which features to extract:
            - None: Extract any noun (excluding large-scale terms
              and custom excludes).
            - "default" or ["default"]: Use the aerial_features.json
              vocabulary for matching.
            - List of strings: Custom allowed features (with or without
              underscores).
        exclude_features: List of noun lemmas to exclude in addition
            to the built-in large-scale terms.
        blip_model_name: Name or path of the BLIP model to use.
            If None, uses the default "Salesforce/blip-image-captioning-base".
        spacy_model_name: Name of the spaCy model to use.
            If None, uses the default "en_core_web_sm".

    Returns:
        A tuple containing:
            - caption: The BLIP-generated caption string.
            - features: Sorted list of extracted feature names.

    Example:
        >>> caption, features = blip_analyze_image(
        ...     "path/to/image.jpg",
        ...     include_features=["default"],
        ...     blip_model_name="Salesforce/blip-image-captioning-large"
        ... )
        >>> print(caption)
        "an aerial view of a parking lot with cars"
        >>> print(features)
        ["car", "parking_lot"]
    """
    # Use custom models if specified, otherwise use default captioner
    if blip_model_name is not None or spacy_model_name is not None:
        captioner = ImageCaptioner(
            blip_model_name=blip_model_name or DEFAULT_BLIP_MODEL,
            spacy_model_name=spacy_model_name or DEFAULT_SPACY_MODEL,
        )
    else:
        captioner = _get_default_captioner()

    return captioner.analyze(
        image_source,
        include_features=include_features,
        exclude_features=exclude_features,
    )
