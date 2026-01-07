from __future__ import annotations

import logging
from typing import Callable, Literal, Mapping, Sequence

import torch
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TEXT_INSTRUCTION = "Na základě níže uvedeného dotazu najdi texty, které nejlépe odpovídají a poskytují relevantní informace."
PromptMode = Literal["none", "custom", "predefined"]
_PROMPT_MODES = {"none", "custom", "predefined"}


def _resolve_device(device: str) -> str:
    """Determine the device string to pass to SentenceTransformer."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _format_query_prompt(instruction: str, query: str) -> str:
    """Standardized instruction/query formatting for query embeddings."""
    return f"Instruct: {instruction}\nQuery: {query}"


def _format_document_prompt(instruction: str, document: str) -> str:
    """Standardized instruction/document formatting for document embeddings."""
    return f"Instruct: {instruction}\nDocument: {document}"


class InstructionalSentenceTransformer(EmbeddingFunction):
    """SentenceTransformer wrapper that can prepend instructions for queries and documents."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        query_instruction: str | None = TEXT_INSTRUCTION,
        query_prompt_mode: PromptMode = "custom",
        document_instruction: str | None = None,
        document_prompt_mode: PromptMode = "none",
        normalize_embeddings: bool = True,
        model_kwargs: Mapping[str, object] | None = None,
        tokenizer_kwargs: Mapping[str, object] | None = None,
        trust_remote_code: bool = False,
        allow_predefined_query: bool = False,
    ) -> None:
        if query_prompt_mode not in _PROMPT_MODES:
            raise ValueError(f"query_prompt_mode must be one of {_PROMPT_MODES}")
        if document_prompt_mode not in _PROMPT_MODES:
            raise ValueError(f"document_prompt_mode must be one of {_PROMPT_MODES}")

        self.device = _resolve_device(device)
        self.normalize_embeddings = normalize_embeddings
        self._query_instruction = query_instruction
        self._document_instruction = document_instruction
        self._query_prompt_mode = self._normalize_prompt_mode(
            query_prompt_mode,
            allow_predefined_query,
            role="query",
        )
        self._document_prompt_mode = self._normalize_prompt_mode(
            document_prompt_mode,
            False,
            role="document",
        )
        self._prompt_type = self._query_prompt_mode

        model_kwargs = dict(model_kwargs or {})
        tokenizer_kwargs = dict(tokenizer_kwargs or {})

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _normalize_prompt_mode(
        prompt_mode: PromptMode,
        allow_predefined: bool,
        *,
        role: str,
    ) -> PromptMode:
        if prompt_mode == "predefined" and not allow_predefined:
            logger.warning(
                "Prompt mode 'predefined' is not supported for %s embeddings. Falling back to 'custom'.",
                role,
            )
            return "custom"
        return prompt_mode

    def _prepare_texts(
        self,
        texts: Sequence[str],
        instruction: str | None,
        prompt_mode: PromptMode,
        formatter: Callable[[str, str], str],
    ) -> list[str]:
        buffer = list(texts)
        if prompt_mode != "custom" or not instruction:
            return buffer
        return [formatter(instruction, text) for text in buffer]

    def _encode_queries(self, texts: Sequence[str]):
        return self.model.encode(texts, normalize_embeddings=self.normalize_embeddings)

    def _encode_documents(self, texts: Sequence[str]):
        return self.model.encode(texts, normalize_embeddings=self.normalize_embeddings)

    def embed_documents(self, texts: Sequence[str]):
        prepared = self._prepare_texts(
            texts,
            self._document_instruction,
            self._document_prompt_mode,
            _format_document_prompt,
        )
        return self._encode_documents(prepared)

    def embed_query(self, text: str):
        prepared = self._prepare_texts(
            [text],
            self._query_instruction,
            self._query_prompt_mode,
            _format_query_prompt,
        )
        return self._encode_queries(prepared)[0]

    def __call__(self, input):
        return self.embed_documents(input)

    @property
    def query_prompt_mode(self) -> PromptMode:
        return self._query_prompt_mode

    @property
    def document_prompt_mode(self) -> PromptMode:
        return self._document_prompt_mode

    def get_prompt_type(self) -> str:
        return self._prompt_type


class Qwen3EmbeddingFunction(InstructionalSentenceTransformer):
    """Embedding function for Qwen3 models with optional built-in query prompts."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "auto",
        prompt_mode: PromptMode = "predefined",
        instruction_text: str = TEXT_INSTRUCTION,
        document_instruction: str | None = None,
        document_prompt_mode: PromptMode = "none",
        use_flash_attention: bool = False,
    ):
        resolved_device = _resolve_device(device)
        model_kwargs = {
            "torch_dtype": torch.float16 if resolved_device == "cuda" else torch.float32,
        }

        if use_flash_attention and resolved_device == "cuda":
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2 for Qwen3 model")
            except ImportError:
                logger.warning("flash_attn not available, falling back to default attention")

        tokenizer_kwargs = {"padding_side": "left"}

        super().__init__(
            model_name,
            device=resolved_device,
            query_instruction=instruction_text,
            query_prompt_mode=prompt_mode,
            document_instruction=document_instruction,
            document_prompt_mode=document_prompt_mode,
            normalize_embeddings=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            allow_predefined_query=True,
        )

        self.use_builtin_prompt = prompt_mode == "predefined"
        self._has_builtin_prompt = (
            hasattr(self.model, "prompts")
            and isinstance(self.model.prompts, dict)
            and "query" in self.model.prompts
        )
        self.prompt_type = "none"
        if self.query_prompt_mode == "none":
            self.prompt_type = "none"
        elif self.use_builtin_prompt and self._has_builtin_prompt:
            self.prompt_type = "builtin"
        else:
            if self.use_builtin_prompt and not self._has_builtin_prompt:
                logger.warning(
                    "Predefined prompt requested but %s has no built-in prompts, falling back to custom instruction.",
                    model_name,
                )
                self.use_builtin_prompt = False
            self.prompt_type = "custom"

        logger.info(
            "Qwen3 prompt configuration: prompt_mode=%s, builtin=%s, has_builtin=%s, prompt_type=%s",
            prompt_mode,
            self.use_builtin_prompt,
            self._has_builtin_prompt,
            self.prompt_type,
        )

    def _encode_queries(self, texts: Sequence[str]):
        if self.prompt_type == "builtin" and self.use_builtin_prompt:
            return self.model.encode(
                texts, prompt_name="query", normalize_embeddings=self.normalize_embeddings
            )
        return super()._encode_queries(texts)

    def get_prompt_type(self) -> str:
        return self.prompt_type


class EmbeddingGemmaEmbeddingFunction(InstructionalSentenceTransformer):
    """Embedding function for Google EmbeddingGemma models with built-in retrieval prompts."""

    def __init__(
        self,
        *,
        model_name: str = "google/embeddinggemma-300m",
        device: str = "auto",
        prompt_mode: PromptMode = "predefined",
        instruction_text: str = TEXT_INSTRUCTION,
        document_instruction: str | None = None,
        document_prompt_mode: PromptMode = "none",
    ):
        resolved_device = _resolve_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32

        super().__init__(
            model_name,
            device=resolved_device,
            query_instruction=instruction_text,
            query_prompt_mode=prompt_mode,
            document_instruction=document_instruction,
            document_prompt_mode=document_prompt_mode,
            normalize_embeddings=True,
            model_kwargs={"torch_dtype": dtype},
            allow_predefined_query=True,
            trust_remote_code=True,  # REQUIRED for EmbeddingGemma
        )

        self.prompt_type = self.get_prompt_type()
        self._has_query_method = hasattr(self.model, "encode_query")
        self._has_document_method = hasattr(self.model, "encode_document")
        logger.info(
            "EmbeddingGemma initialized: has_encode_query=%s, has_encode_document=%s",
            self._has_query_method,
            self._has_document_method,
        )

    @staticmethod
    def _extract_title_and_content(text: str) -> tuple[str, str] | None:
        """Extract title and content from document text format: 'title: {title}\\nurl: {url}\\ntext: {content}'
        
        Returns (title, content) if format matches, None otherwise.
        """
        title = "none"
        content = None
        
        # Try to extract title from "title: {title}\n" pattern
        if text.startswith("title: "):
            lines = text.split("\n", 2)
            if len(lines) >= 1:
                title_line = lines[0]
                if title_line.startswith("title: "):
                    title = title_line[7:].strip()  # Remove "title: " prefix
        
        # Try to extract content from "text: {content}" pattern
        if "\ntext: " in text:
            parts = text.split("\ntext: ", 1)
            if len(parts) == 2:
                content = parts[1]
        elif text.startswith("text: "):
            content = text[6:]  # Remove "text: " prefix
        
        # If we couldn't extract content, the format doesn't match
        if content is None:
            return None
        
        return title, content

    def embed_documents(self, texts: Sequence[str]):
        """Format documents as 'title: {title | "none"} | text: {content}' for Gemma retrieval."""
        formatted_texts = texts #[]
        # for text in texts:
        #     extracted = self._extract_title_and_content(text)
        #     if extracted is not None:
        #         title, content = extracted
        #         formatted = f"title: {title} | text: {content}"
        #     else:
        #         # If format doesn't match, use original text as-is
        #         formatted = text
        #     formatted_texts.append(formatted)
        
        if self._has_document_method:
            return self.model.encode_document(formatted_texts)
        return self.model.encode(formatted_texts, normalize_embeddings=self.normalize_embeddings)

    def embed_query(self, text: str):
        """Format query as 'task: search result | query: {content}' for Gemma retrieval."""
        formatted = f"task: search result | query: {text}"
        
        if self._has_query_method:
            return self.model.encode_query([formatted])[0]
        return self.model.encode([formatted], normalize_embeddings=self.normalize_embeddings)[0]

    def _encode_queries(self, texts: Sequence[str]):
        if self._has_query_method:
            return self.model.encode_query(texts)
        return super()._encode_queries(texts)

    def _encode_documents(self, texts: Sequence[str]):
        if self._has_document_method:
            return self.model.encode_document(texts)
        return super()._encode_documents(texts)


class GeneralEmbeddingFunction(InstructionalSentenceTransformer):
    """General embedder for SBERT-style models with query + document instructions."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-multilingual-gemma2",
        device: str = "auto",
        prompt_mode: PromptMode = "custom",
        instruction_text: str = TEXT_INSTRUCTION,
        document_instruction: str | None = None,
        document_prompt_mode: PromptMode = "none",
    ) -> None:
        resolved_device = _resolve_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32

        super().__init__(
            model_name,
            device=resolved_device,
            query_instruction=instruction_text,
            query_prompt_mode=prompt_mode,
            document_instruction=document_instruction,
            document_prompt_mode=document_prompt_mode,
            normalize_embeddings=True,
            model_kwargs={"torch_dtype": dtype},
        )


class LlamaEmbedNemotronEmbeddingFunction(InstructionalSentenceTransformer):
    """Embedding function for NVIDIA llama-embed-nemotron models."""

    def __init__(
        self,
        *,
        model_name: str = "nvidia/llama-embed-nemotron-8b",
        device: str = "auto",
        prompt_mode: PromptMode = "custom",
        task_instruction: str = TEXT_INSTRUCTION,
        document_instruction: str | None = None,
        document_prompt_mode: PromptMode = "none",
        use_flash_attention: bool = True,
    ):
        resolved_device = _resolve_device(device)
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
        model_kwargs = {"torch_dtype": dtype}
        attn_implementation = "eager"
        if use_flash_attention and resolved_device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_implementation = "flash_attention_2"
                model_kwargs["attn_implementation"] = attn_implementation
                logger.info("Using flash_attention_2 for llama-embed-nemotron-8b")
            except ImportError:
                logger.warning(
                    "flash_attn not available, falling back to eager attention"
                )

        tokenizer_kwargs = {"padding_side": "left"}

        super().__init__(
            model_name,
            device=resolved_device,
            query_instruction=task_instruction,
            query_prompt_mode=prompt_mode,
            document_instruction=document_instruction,
            document_prompt_mode=document_prompt_mode,
            normalize_embeddings=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=True,
        )

        self.prompt_type = self.get_prompt_type()
        self._has_query_method = hasattr(self.model, "encode_query")
        self._has_document_method = hasattr(self.model, "encode_document")
        logger.info(
            "LlamaEmbedNemotron initialized: has_encode_query=%s, has_encode_document=%s, attn=%s",
            self._has_query_method,
            self._has_document_method,
            attn_implementation,
        )

    def _encode_queries(self, texts: Sequence[str]):
        if self._has_query_method:
            return self.model.encode_query(texts)
        return super()._encode_queries(texts)

    def _encode_documents(self, texts: Sequence[str]):
        if self._has_document_method:
            return self.model.encode_document(texts)
        return super()._encode_documents(texts)
