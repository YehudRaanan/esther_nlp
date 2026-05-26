"""Data access layer for Megillat Esther corpus."""

import ast
from pathlib import Path
from typing import List, Tuple

# Default data path relative to this source file
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "esther_dataset.txt"


def load_raw_dataset(data_path: Path = DEFAULT_DATA_PATH) -> List[str]:
    """Loads the raw string list from the dataset file.

    Args:
        data_path: Path to the dataset text file.

    Returns:
        List of raw strings, including '[CHAPTER]' markers.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Esther dataset not found at: {data_path.resolve()}")
    
    content = data_path.read_text(encoding="utf-8").strip()
    # Wrap in square brackets to load as python list literal
    return ast.literal_eval(f"[{content}]")


def load_chapters_and_verses(data_path: Path = DEFAULT_DATA_PATH) -> Tuple[List[List[str]], List[str]]:
    """Loads the dataset, separating it into chapters and a flat list of all verses.

    Args:
        data_path: Path to the dataset text file.

    Returns:
        A tuple of (chapters, all_verses), where:
        - chapters: List of chapters, each chapter being a List of verse strings.
        - all_verses: Flat list of all 176 verse strings (with [CHAPTER] markers filtered out).
    """
    raw_strings = load_raw_dataset(data_path)
    
    chapters: List[List[str]] = [[]]
    for item in raw_strings:
        if item == "[CHAPTER]":
            chapters.append([])
        else:
            chapters[-1].append(item)
            
    # Filter out empty chapters (e.g., initial or trailing separators)
    chapters = [c for c in chapters if c]
    all_verses = [verse for chapter in chapters for verse in chapter]
    
    return chapters, all_verses


def get_verse_location_map(chapters: List[List[str]]) -> dict:
    """Returns a dictionary mapping global verse index (0-175) to (chapter_num, verse_num) 1-based tuples.

    Args:
        chapters: Nested list of chapter verses.

    Returns:
        Dict mapping global index -> (chapter_1_based, verse_in_chapter_1_based).
    """
    verse_loc = {}
    global_idx = 0
    for ch_idx, chapter in enumerate(chapters, start=1):
        for verse_idx, _ in enumerate(chapter, start=1):
            verse_loc[global_idx] = (ch_idx, verse_idx)
            global_idx += 1
    return verse_loc
