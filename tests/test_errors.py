"""Tests for backend.errors."""

from backend import errors
from backend.errors import EmptyDocumentError, ExtractionFailedError, LensError


def _all_error_classes() -> list[type[LensError]]:
    return [
        obj
        for obj in vars(errors).values()
        if isinstance(obj, type) and issubclass(obj, LensError) and obj is not LensError
    ]


def test_every_error_is_a_lens_error():
    assert issubclass(ExtractionFailedError, LensError)
    assert issubclass(EmptyDocumentError, LensError)


def test_codes_are_unique():
    codes = [cls.code for cls in _all_error_classes()]
    assert len(codes) == len(set(codes)), f"duplicate error codes: {codes}"


def test_no_error_inherits_the_base_code():
    for cls in _all_error_classes():
        assert cls.code != LensError.code, f"{cls.__name__} did not set its own code"


def test_detail_is_carried_and_readable():
    error = EmptyDocumentError("0 pages")
    assert error.detail == "0 pages"
    assert str(error) == "0 pages"


def test_code_survives_being_raised():
    try:
        raise ExtractionFailedError("docling crashed")
    except LensError as error:
        assert error.code == "extraction_failed"
        assert error.detail == "docling crashed"
