"""Unit tests for the data access layer."""

from esther_nlp.data import load_chapters_and_verses, get_verse_location_map


def test_load_dataset():
    """Asserts that Megillat Esther parses exactly 10 chapters and 167 verses."""
    chapters, all_verses = load_chapters_and_verses()
    
    assert len(chapters) == 10, f"Expected 10 chapters, found {len(chapters)}"
    assert len(all_verses) == 167, f"Expected 167 verses, found {len(all_verses)}"
    
    # Check simple content sanity
    assert "ויהי בימי אחשורוש" in all_verses[0], "First verse did not match expectations."
    assert "כי מרדכי היהודי משנה" in all_verses[-1], "Last verse did not match expectations."


def test_verse_location_mapping():
    """Asserts verse location mapping converts correctly."""
    chapters, _ = load_chapters_and_verses()
    loc_map = get_verse_location_map(chapters)
    
    # Assert size
    assert len(loc_map) == 167
    
    # Assert coordinates bounds
    first_ch, first_v = loc_map[0]
    assert first_ch == 1
    assert first_v == 1
    
    last_ch, last_v = loc_map[166]
    assert last_ch == 10
    assert last_v == 3
