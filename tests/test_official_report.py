# ruff: noqa: E501  -- the fixture below is the publisher's markup, kept on its own lines
"""Reading the Official Report's markup, on a fragment shaped like the real page."""

from __future__ import annotations

from stm_pipeline.official_report import Contribution, parse_official_report

PAGE = """
<div class="rich-text"> <h2 class="h3 u--mtop0">First Minister’s Question Time</h2> <hr />
<p id="225813" class="lead">Party Leaders</p>
<p id="orscontributions_C3100812"> <div> <strong> <a name="0"></a> </strong> </div> </p>
<p>14:01</p>
<p id="orscontributions_M2110E489P990C3100813"> <div> <strong> <a name="2110" href="/msps/x">The Presiding Officer (Kenneth Gibson)</a> </strong> </div> </p>
<p>We now move to First Minister&#x2019;s question time for party leaders. [Applause.]</p>
<hr />
<p id="225814" class="lead">Government Priorities</p>
<p id="orscontributions_M21005E400P974C3100814"> <div> <strong> <a name="21005" href="/msps/y">1. Malcolm Offord (West Scotland) (Reform)</a> </strong> <div class="share-float-right">share</div> </div> </p>
<p>When John Swinney became First Minister in May 2024, he promised to lead.</p>
<p>Second paragraph of the same speaker.</p>
<p id="orscontributions_M2110E489P990C3100815"> <div> <strong> <a name="2110" href="/msps/x">The First Minister (John Swinney)</a> </strong> </div> </p>
<p>One of the many elements&nbsp;of the Government.</p>
</div> <!-- Previous/Next navigation --> <hr> <div class="row"> <div class="col-sm-6"> <p class="h5">Previous</p>
<a href="?meeting=20233&amp;iob=225803"><span>General Question Time</span></a> </div>
<div class="col-sm-6"> <p class="h5">Next</p> <a href="?iob=225819"><span>Programme for International Student Assessment 2025</span></a> </div> </div>
<p>Back to Official Report: search what was said in Parliament</p>
<p>&copy; Scottish Parliament 2026</p>
</div>
"""


def test_contributions_speakers_paragraphs_and_clock() -> None:
    cs = parse_official_report(PAGE)
    assert [c.speaker for c in cs] == [
        "The Presiding Officer (Kenneth Gibson)",
        "1. Malcolm Offord (West Scotland) (Reform)",
        "The First Minister (John Swinney)",
    ]
    assert cs[0].clock == "14:01" and cs[1].clock is None
    assert cs[0].paragraphs == [
        "We now move to First Minister’s question time for party leaders. [Applause.]"
    ]
    assert cs[1].paragraphs == [
        "When John Swinney became First Minister in May 2024, he promised to lead.",
        "Second paragraph of the same speaker.",
    ]
    assert cs[2].paragraphs == ["One of the many elements of the Government."]


def test_headings_and_footer_are_not_speech() -> None:
    cs = parse_official_report(PAGE)
    joined = " ".join(p for c in cs for p in c.paragraphs)
    assert "Party Leaders" not in joined
    assert "Government Priorities" not in joined
    assert "Back to Official Report" not in joined
    assert "©" not in joined
    assert "Previous" not in joined and "Next" not in joined and "Assessment" not in joined


def test_display_names_drop_numbers_and_parentheses() -> None:
    assert (
        Contribution("1. Malcolm Offord (West Scotland) (Reform)").display_name == "Malcolm Offord"
    )
    assert Contribution("The First Minister (John Swinney)").display_name == "The First Minister"
    assert Contribution("The Presiding Officer").display_name == "The Presiding Officer"


def test_word_count_and_empty_page() -> None:
    cs = parse_official_report(PAGE)
    assert cs[1].word_count == 19
    assert parse_official_report("<p>nothing here</p>") == []
