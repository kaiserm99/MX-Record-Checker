from writing_assistant_mcp.text import normalize, segment, split_paragraphs, split_sentences


def test_split_sentences_handles_abbreviations_and_quotes():
    text = 'Dr. Smith arrived at 5 p.m. with J. Doe. "It works!" she said. Then e.g. this stays.'
    assert split_sentences(text) == [
        "Dr. Smith arrived at 5 p.m. with J. Doe.",
        '"It works!" she said.',
        "Then e.g. this stays.",
    ]


def test_list_items_are_separate_paragraphs():
    text = "Intro line.\n- first item\n- second item\n\nNext paragraph."
    assert split_paragraphs(text) == ["Intro line.", "- first item", "- second item", "Next paragraph."]


def test_segment_indexes_sentences_and_paragraphs():
    sentences = segment("One. Two.\n\nThree.")
    assert [(s.index, s.paragraph, s.text) for s in sentences] == [
        (0, 0, "One."),
        (1, 0, "Two."),
        (2, 1, "Three."),
    ]


def test_normalize_strips_hidden_characters_and_mixed_script_homoglyphs():
    # Cyrillic "е" and "о" inside Latin words, plus a zero-width space
    result = normalize("Thе mоdel​ works. Αθήνα stays Greek.")
    assert result.text == "The model works. Αθήνα stays Greek."
    assert result.zero_width_removed == 1
    assert result.homoglyphs_replaced == 2
