from __future__ import annotations

import logging
import random
import re
from typing import List

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# Download NLTK stopwords if not already available
try:
    STOPWORDS = set(stopwords.words("czech"))
except (LookupError, OSError):
    logger.warning("Czech stopwords not found. Downloading...")
    try:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("czech"))
    except Exception as e:
        logger.error(f"Failed to download Czech stopwords: {e}")
        logger.warning("Using empty stopwords set. Install Czech stopwords with: nltk.download('stopwords')")
        STOPWORDS = set()


class CzechPreprocessor:
    """Czech text preprocessing with UDPipe lemmatization, PyStemmer stemming, or basic tokenization only."""

    def __init__(self, mode: str = "udpipe", stopwords: set[str] | None = None):
        """
        Initialize Czech preprocessor.

        Args:
            mode: "udpipe" (default) for lemmatization, "stem" for stemming, or "none" for basic tokenization only
            stopwords: Set of stopwords to remove. If None, uses NLTK Czech stopwords.
        """
        if mode not in {"udpipe", "stem", "none"}:
            raise ValueError(f"mode must be 'udpipe', 'stem', or 'none', got {mode}")

        self.mode = mode
        self.stopwords = stopwords if stopwords is not None else STOPWORDS

        if mode == "udpipe":
            try:
                from ufal.udpipe import Model, Pipeline
                import os

                # Try to load the Czech UDPipe model
                # Common paths: czech-pdt-ud-2.5-191206.udpipe or newer versions
                model_filenames = [
                    "czech-pdt-ud-2.5-191206.udpipe",
                    "czech-pdt-ud-2.12-230717.udpipe",  # newer version
                ]
                
                # Common locations to search for UDPipe models
                search_paths = [
                    ".",  # Current directory
                    os.path.dirname(__file__),  # Directory where this file is located (bm25/)
                    os.path.expanduser("~/udpipe-models"),  # Common user directory
                    os.path.expanduser("~/.local/share/udpipe"),  # Another common location
                    "/usr/local/share/udpipe",  # System-wide location
                    os.path.join(os.path.dirname(__file__), "models"),  # Relative to this file
                ]
                
                self.model = None
                loaded_path = None
                
                for filename in model_filenames:
                    # First try just the filename (in case it's in the current working directory)
                    try:
                        self.model = Model.load(filename)
                        logger.info(f"Loaded UDPipe model: {filename}")
                        loaded_path = filename
                        break
                    except Exception:
                        pass
                    
                    # Then try in common search paths
                    for search_path in search_paths:
                        if not os.path.exists(search_path):
                            continue
                        model_path = os.path.join(search_path, filename)
                        try:
                            if os.path.exists(model_path):
                                self.model = Model.load(model_path)
                                logger.info(f"Loaded UDPipe model: {model_path}")
                                loaded_path = model_path
                                break
                        except Exception:
                            continue
                    
                    if self.model is not None:
                        break

                if self.model is None:
                    error_msg = (
                        f"Could not load UDPipe model. Tried filenames: {model_filenames}\n"
                        f"Searched in: {search_paths}\n\n"
                        "Please download the Czech UDPipe model from:\n"
                        "  https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131\n"
                        "or\n"
                        "  https://github.com/ufal/udpipe/tree/master/src/models\n\n"
                        "Place the .udpipe file in one of the search paths above, "
                        "or in the current working directory."
                    )
                    raise FileNotFoundError(error_msg)

                self.pipeline = Pipeline(
                    self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
                )
                self.Pipeline = Pipeline
            except ImportError:
                raise ImportError(
                    "ufal.udpipe is required for UDPipe mode. Install with: pip install ufal.udpipe"
                )

        elif mode == "stem":
            try:
                import PyStemmer

                self.stemmer = PyStemmer.Stemmer("czech")
            except ImportError:
                raise ImportError(
                    "PyStemmer is required for stem mode. Install with: pip install PyStemmer"
                )
        elif mode == "none":
            # No additional initialization needed for "none" mode
            # Just basic tokenization and lowercasing
            pass

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess Czech text according to the configured mode.

        Args:
            text: Input text string

        Returns:
            List of normalized tokens
        """
        if self.mode == "udpipe":
            return self._preprocess_udpipe(text)
        elif self.mode == "stem":
            return self._preprocess_stem(text)
        else:  # mode == "none"
            return self._preprocess_none(text)

    def _preprocess_udpipe(self, text: str) -> List[str]:
        """Preprocess using UDPipe tokenization and lemmatization."""
        # Process with UDPipe
        processed = self.pipeline.process(text)
        tokens = []

        # Parse CONLLU format to extract lemmas
        for line in processed.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                # CONLLU format: ID, FORM, LEMMA, ...
                lemma = parts[2]  # LEMMA column
                if lemma != "_" and lemma:
                    tokens.append(lemma)

        # Apply text processing rules
        return self._apply_text_rules(tokens)

    def _preprocess_stem(self, text: str) -> List[str]:
        """Preprocess using PyStemmer stemming."""
        # Simple tokenization: split on whitespace and punctuation
        # Keep hyphens inside tokens
        words = re.findall(r"\b[\w-]+\b", text)
        tokens = []

        for word in words:
            # Stem the word
            stemmed = self.stemmer.stem(word)
            tokens.append(stemmed)

        # Apply text processing rules
        return self._apply_text_rules(tokens)

    def _preprocess_none(self, text: str) -> List[str]:
        """
        Preprocess with no stemming or lemmatization.
        Only performs basic tokenization and lowercasing.
        """
        # Simple tokenization: split on whitespace and punctuation
        # Keep hyphens inside tokens
        words = re.findall(r"\b[\w-]+\b", text)
        tokens = []

        for word in words:
            # No stemming or lemmatization - use word as-is
            tokens.append(word)

        # Apply text processing rules (lowercasing, stopword removal, etc.)
        return self._apply_text_rules(tokens)

    def _apply_text_rules(self, tokens: List[str]) -> List[str]:
        """
        Apply text processing rules to tokens:
        - Keep digits
        - Strip punctuation except hyphens inside tokens
        - Keep acronyms as-is (all-caps words)
        - Lowercase all text
        - Remove Czech stopwords
        - Preserve diacritics
        """
        processed = []

        for token in tokens:
            # Keep digits as-is
            if token.isdigit():
                processed.append(token)
                continue

            # Detect acronyms (all caps, at least 2 characters)
            if len(token) >= 2 and token.isupper() and token.isalpha():
                # Keep acronyms as-is (but lowercase them per rules)
                processed.append(token.lower())
                continue

            # Strip punctuation except hyphens inside tokens
            # Hyphens are already preserved by tokenization regex
            # Remove other punctuation
            cleaned = re.sub(r"[^\w-]", "", token)

            # Skip empty tokens
            if not cleaned:
                continue

            # Lowercase
            cleaned = cleaned.lower()

            # Remove stopwords
            if cleaned in self.stopwords:
                continue

            # Skip if token becomes empty after cleaning
            if cleaned:
                processed.append(cleaned)

        return processed


def preprocess_czech_text(text: str, mode: str = "udpipe") -> List[str]:
    """
    Helper function to preprocess Czech text.

    Args:
        text: Input text string
        mode: "udpipe" (default) for lemmatization, "stem" for stemming, or "none" for basic tokenization only

    Returns:
        List of normalized tokens
    """
    preprocessor = CzechPreprocessor(mode=mode)
    return preprocessor.preprocess(text)

