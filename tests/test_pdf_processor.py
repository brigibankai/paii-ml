from paii.pdf import PdfProcessor


def test_chunk_page_paragraphs_and_metadata():
    text = """
    Paragraph one line one.

    Paragraph two has more content and should be grouped together.

    Paragraph three is short.
    """

    proc = PdfProcessor(chunk_size=200)
    chunks = proc.chunk_page(text, page_num=1)

    assert len(chunks) >= 2
    for c in chunks:
        assert "text" in c
        assert "page" in c and c["page"] == 1
        assert "start_char" in c and "end_char" in c
