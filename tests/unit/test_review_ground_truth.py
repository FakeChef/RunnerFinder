from review_ground_truth import parse_human_bibs


def test_parse_human_bibs_supports_all_review_separators():
    assert parse_human_bibs("00123, A18866，B18866、31087; 13568；16235\n2026") == [
        "00123", "A18866", "B18866", "31087", "13568", "16235", "2026"
    ]


def test_parse_human_bibs_keeps_31616_and_68896_separate():
    assert parse_human_bibs("31616，68896") == ["31616", "68896"]


def test_parse_human_bibs_ignores_empty_values():
    assert parse_human_bibs(" ,，、;；\n  ") == []
