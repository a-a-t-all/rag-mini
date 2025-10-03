import subprocess, sys, json, os, pathlib, shutil

def test_cli(tmp_path):
    # Copy repo minimal docs to temp and ingest
    here = pathlib.Path(__file__).resolve().parents[1]
    docs_src = here / "docs" / "sample.txt"
    work = tmp_path
    (work/"docs").mkdir()
    shutil.copy(docs_src, work/"docs"/"sample.txt")

    # Install is assumed. Run ingest
    cmd = [sys.executable, "-m", "src.ingest", "--docs_dir", str(work/"docs"), "--outdir", str(work/"artifacts")]
    r = subprocess.run(cmd, cwd=str(here), capture_output=True, text=True)
    assert r.returncode == 0

    # Ask a simple Q
    cmd = [sys.executable, "-m", "src.rag", "--artifacts", str(work/"artifacts"),
           "--question", "What is RAG?"]
    r = subprocess.run(cmd, cwd=str(here), capture_output=True, text=True)
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "answer" in data and "sources" in data
