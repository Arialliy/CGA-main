from pathlib import Path
import pytest
from tools.official.check_cga_v2_claim_guard import main
from tools.official.check_cga_v2_claim_guard import find_rejected_phrases


def test_claim_guard_rejects_bad_phrase(tmp_path, monkeypatch):
    p = tmp_path / "paper.md"
    p.write_text("CGA-v2 is externally validated.", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", "--paths", str(p), "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        main()


def test_claim_guard_allows_rejected_claims_section():
    text = "## Rejected claims\n\n- CGA-v2 is externally validated.\n"
    assert find_rejected_phrases(text) == []
