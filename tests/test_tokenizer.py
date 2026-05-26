"""Unit tests for the BPE tokenizer module."""

from esther_nlp.data import load_chapters_and_verses
from esther_nlp.tokenizer import EstherBpeTokenizer, compute_vocabulary_counts


def test_tokenizer_training_and_encoding():
    """Asserts BPE tokenizer trains correctly and encodes text."""
    _, all_verses = load_chapters_and_verses()
    
    tokenizer = EstherBpeTokenizer(vocab_size=1000)
    assert not tokenizer._is_trained
    
    tokenizer.train(all_verses)
    assert tokenizer._is_trained
    
    # Assert vocabulary properties
    vocab = tokenizer.get_vocab()
    assert len(vocab) <= 1000
    assert "[UNK]" in vocab
    
    # Test encoding/decoding loops
    test_verse = "ויהי בימי אחשורוש הוא אחשורוש המלך המלך"
    tokens = tokenizer.encode_to_tokens(test_verse)
    assert len(tokens) > 0
    assert isinstance(tokens[0], str)
    
    ids = tokenizer.encode_to_ids(test_verse)
    assert len(ids) == len(tokens)
    assert isinstance(ids[0], int)
    
    # Reconstructed text match
    decoded = tokenizer.decode(ids)
    assert len(decoded) > 0


def test_compute_vocabulary_counts():
    """Asserts token frequency counters sum up correctly."""
    tokenized = [["המלך", "אשר"], ["המלך", "המן", "במלך"]]
    counts = compute_vocabulary_counts(tokenized)
    
    assert counts["המלך"] == 2
    assert counts["המן"] == 1
    assert counts["אשר"] == 1
    assert counts["במלך"] == 1
    assert "אחשורוש" not in counts
