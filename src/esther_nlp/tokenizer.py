"""Tokenization wrapper using Hugging Face Byte Pair Encoding (BPE)."""

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class EstherBpeTokenizer:
    """Wrapper around Hugging Face Tokenizer for Esther NLP pipeline."""

    def __init__(self, vocab_size: int = 1000, unk_token: str = "[UNK]"):
        """Initializes a new BpeTokenizer wrapper."""
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        # Create core HF tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()
        self._is_trained = False

    def train(self, texts: Iterable[str], special_tokens: List[str] = None) -> None:
        """Trains the BPE tokenizer from an iterator of texts.

        Args:
            texts: Iterable of string sentences/verses.
            special_tokens: Optional custom list of special tokens.
        """
        if special_tokens is None:
            special_tokens = [self.unk_token]
            
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens
        )
        self.tokenizer.train_from_iterator(texts, trainer)
        self._is_trained = True

    def encode_to_tokens(self, text: str) -> List[str]:
        """Encodes a string into a list of BPE string tokens.

        Args:
            text: Raw input string.

        Returns:
            List of string tokens.
        """
        return self.tokenizer.encode(text).tokens

    def encode_to_ids(self, text: str) -> List[int]:
        """Encodes a string into a list of BPE numerical token IDs.

        Args:
            text: Raw input string.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token IDs back into a reconstructed string.

        Args:
            ids: List of token IDs.

        Returns:
            Reconstructed string.
        """
        return self.tokenizer.decode(ids)

    def get_vocab(self) -> dict:
        """Returns the vocabulary dictionary mapping string tokens -> token IDs."""
        return self.tokenizer.get_vocab()

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves the trained tokenizer model to a JSON file.

        Args:
            filepath: Target file path to write.
        """
        self.tokenizer.save(str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EstherBpeTokenizer":
        """Loads a pre-trained BPE tokenizer model from a JSON file.

        Args:
            filepath: Source JSON tokenizer file path.

        Returns:
            An initialized EstherBpeTokenizer instance.
        """
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(str(filepath))
        instance._is_trained = True
        return instance


def compute_vocabulary_counts(tokenized_corpus: List[List[str]]) -> Counter:
    """Helper to compute token frequency counts over the tokenized corpus.

    Args:
        tokenized_corpus: Nested list of tokenized sentences.

    Returns:
        Counter mapping string tokens to total corpus occurrences.
    """
    return Counter(token for sentence in tokenized_corpus for token in sentence)
