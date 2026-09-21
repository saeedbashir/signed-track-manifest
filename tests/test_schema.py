from __future__ import annotations

import copy
from typing import Any

import pytest

from stm.cli import main as validate_main
from stm.schema import is_valid, validate_catalogue, validate_title
from tests.conftest import EXAMPLES, load_example


@pytest.mark.parametrize("name", ["corner-inset.json", "side-panel.json"])
def test_valid_examples(name: str) -> None:
    issues = validate_title(load_example(name))
    assert is_valid(issues), [str(i) for i in issues]


def test_valid_catalogue_example() -> None:
    issues = validate_catalogue(load_example("catalogue.json"))
    assert is_valid(issues), [str(i) for i in issues]


@pytest.mark.parametrize(
    "name",
    [
        "invalid-missing-provenance.json",
        "invalid-reviewed-without-reviewer.json",
        "invalid-synthesised-with-extraction.json",
    ],
)
def test_invalid_examples_fail(name: str) -> None:
    assert not is_valid(validate_title(load_example(name)))


def _base() -> dict[str, Any]:
    return copy.deepcopy(load_example("corner-inset.json"))


def test_provenance_has_no_default() -> None:
    data = _base()
    del data["signLanguage"][0]["provenance"]
    issues = validate_title(data)
    assert any("provenance" in i.message for i in issues)


def test_provenance_enum_is_closed() -> None:
    data = _base()
    data["signLanguage"][0]["provenance"] = "probably-human"
    assert not is_valid(validate_title(data))


def test_generated_caption_requires_reviewed() -> None:
    data = _base()
    del data["captions"][1]["reviewed"]
    issues = validate_title(data)
    assert any("reviewed" in i.message for i in issues)


def test_reviewed_true_requires_reviewer() -> None:
    data = _base()
    data["captions"][1]["reviewed"] = True
    assert not is_valid(validate_title(data))
    data["captions"][1]["reviewedBy"] = "A. Reviewer"
    assert is_valid(validate_title(data))


def test_reviewer_without_reviewed_true_is_rejected() -> None:
    data = _base()
    data["captions"][0]["reviewedBy"] = "A. Reviewer"
    assert not is_valid(validate_title(data))


def test_iso_639_3_language_codes() -> None:
    data = _base()
    data["signLanguage"][0]["language"] = "en"
    assert not is_valid(validate_title(data))
    data["signLanguage"][0]["language"] = "bfi"
    assert is_valid(validate_title(data))


def test_additional_properties_rejected() -> None:
    data = _base()
    data["signLanguage"][0]["extraction"]["inpainting"] = "generative"
    assert not is_valid(validate_title(data))


def test_low_confidence_is_a_warning_not_an_error() -> None:
    data = _base()
    data["signLanguage"][0]["extraction"]["confidence"] = 0.5
    issues = validate_title(data)
    assert is_valid(issues)
    assert not is_valid(issues, warnings_as_errors=True)
    assert any(i.severity == "warning" for i in issues)


def test_duplicate_sign_language_is_an_error() -> None:
    data = _base()
    data["signLanguage"].append(copy.deepcopy(data["signLanguage"][0]))
    issues = validate_title(data)
    assert any("duplicate" in i.message for i in issues)


def test_duplicate_caption_kind_is_an_error() -> None:
    data = _base()
    data["captions"].append(copy.deepcopy(data["captions"][0]))
    assert not is_valid(validate_title(data))


def test_duplicate_catalogue_ids() -> None:
    data = load_example("catalogue.json")
    data["titles"][1]["id"] = data["titles"][0]["id"]
    assert not is_valid(validate_catalogue(data))


def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert validate_main([str(EXAMPLES / "corner-inset.json")]) == 0
    assert validate_main([str(EXAMPLES / "invalid-missing-provenance.json")]) == 1
    assert validate_main(["--catalogue", str(EXAMPLES / "catalogue.json")]) == 0
    assert validate_main([str(EXAMPLES / "does-not-exist.json")]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "provenance" in out
