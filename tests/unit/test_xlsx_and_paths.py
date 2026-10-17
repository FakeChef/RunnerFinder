import os
import subprocess
import sys
import zipfile
from pathlib import Path

from utils.xlsx_writer import read_xlsx_text, write_xlsx

ROOT = Path(__file__).resolve().parents[2]


def test_xlsx_keeps_leading_zeros_and_text_format(tmp_path):
    p = write_xlsx(tmp_path / "r.xlsx", ["photo", "bibs"],
                   [["1", "00123;UT001"], ["2", "007"], ["3", "A&B <x> 太湖"]])
    assert read_xlsx_text(p) == [["photo", "bibs"], ["1", "00123;UT001"], ["2", "007"],
                                 ["3", "A&B <x> 太湖"]]
    with zipfile.ZipFile(p) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
        styles = z.read("xl/styles.xml").decode()
        assert {"[Content_Types].xml", "xl/workbook.xml", "_rels/.rels"} <= set(z.namelist())
    assert 'numFmtId="49"' in styles                  # 49 = "@" text format
    assert 'style="1"' in sheet                       # whole columns formatted as text
    assert 't="inlineStr"' in sheet and "<v>" not in sheet   # no numeric cells


def _cache_from(cwd, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k != "PADDLE_PDX_CACHE_HOME"}
    env["PYTHONPATH"] = str(ROOT)
    env.update(extra_env or {})
    out = subprocess.run([sys.executable, "-c",
                          "import config, os; print(os.environ['PADDLE_PDX_CACHE_HOME'])"],
                         cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_paddle_cache_is_absolute_and_independent_of_working_directory(tmp_path):
    a = _cache_from(ROOT)
    b = _cache_from(tmp_path)
    assert a == b == str(ROOT / "models" / "paddlex")
    assert os.path.isabs(a)


def test_paddle_cache_environment_override_is_respected(tmp_path):
    assert _cache_from(tmp_path, {"PADDLE_PDX_CACHE_HOME": str(tmp_path / "x")}) == str(tmp_path / "x")
