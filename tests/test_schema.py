from __future__ import annotations

import pandas as pd
import pytest

from argus.data.load import DataLoadError, load_ibm_aml
from argus.data.validate import DataValidationError, validate_raw_schema

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)
ROW = "2022/09/01 00:00,010,AAA,020,BBB,10,US Dollar,10,US Dollar,ACH,0\n"


def test_loader_preserves_identifier_strings_and_duplicate_header_position(tmp_path) -> None:
    source = tmp_path / "HI-Small_Trans.csv"
    source.write_text(HEADER + ROW, encoding="utf-8")

    frame = load_ibm_aml(source)

    assert frame.loc[0, "From Bank"] == "010"
    assert frame.loc[0, "Account"] == "AAA"
    assert frame.loc[0, "Account.1"] == "BBB"
    assert frame.loc[0, "source_row_number"] == 2
    assert validate_raw_schema(frame).valid


def test_loader_rejects_schema_drift(tmp_path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("Timestamp,Wrong\n2024-01-01,x\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="Unexpected IBM AML transaction schema"):
        load_ibm_aml(source)


def test_schema_report_raises_for_missing_required_column() -> None:
    report = validate_raw_schema(pd.DataFrame({"Timestamp": ["2024-01-01"]}))

    assert not report.valid
    with pytest.raises(DataValidationError, match="missing_required_columns"):
        report.raise_for_errors()
