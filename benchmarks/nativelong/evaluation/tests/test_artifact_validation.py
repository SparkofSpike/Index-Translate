#!/usr/bin/env python3
"""Review fixes for the nativelong evaluation suite.

Each test asserts the *fixed* behavior and, where the defect was behavioural,
also asserts the baseline behaviour so a regression cannot silently return.

Run directly (python tests/test_artifact_validation.py) or under pytest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent
SCRIPTS = EVALUATION / "scripts"
ROOT = EVALUATION.parents[1]
# The revision this review fixed; baseline assertions must not track HEAD.
BASELINE_REV = "9cc5ee7"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EVALUATION))


def stub_heavy_dependencies() -> None:
    """evaluate_comet imports torch/comet at module scope; the fixes do not."""
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-stub"
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: False
    backends.mps = mps
    torch.backends = backends
    comet = types.ModuleType("comet")
    comet.download_model = lambda *a, **k: None
    comet.load_from_checkpoint = lambda *a, **k: None
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.backends", backends)
    sys.modules.setdefault("torch.backends.mps", mps)
    sys.modules.setdefault("comet", comet)


stub_heavy_dependencies()
import evaluate_comet as ec  # noqa: E402
import summarize_document_segale as sds  # noqa: E402
import run as runner  # noqa: E402


def baseline_module(relative_path: str, name: str):
    """Load the pre-fix revision of a module straight out of the reviewed baseline."""
    source = subprocess.run(
        ["git", "show", f"{BASELINE_REV}:{relative_path}"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    path = EVALUATION / "tests" / f"_baseline_{name}.py"
    path.write_text(source, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(f"baseline_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        path.unlink()
    return module


# Original source lines are "Aaa"/"Bbb"/"Ccc"/"Ddd" (12 original characters).
# Both windows join two sentences, so summing len(row['src']) gives 14 - the
# mixed basis the review flagged.
MAPPED_ROWS = [
    {"doc_id": "d", "seg_id": 1, "src": "Aaa Bbb", "ref": "r1", "tgt": "t1",
     "alignment_type": "aligned", "source_char_start": 0, "source_char_end": 6,
     "source_sentence_start": 0, "source_sentence_end": 2},
    {"doc_id": "d", "seg_id": 2, "src": "Ccc Ddd", "ref": "r2", "tgt": "t2",
     "alignment_type": "aligned", "source_char_start": 6, "source_char_end": 12,
     "source_sentence_start": 2, "source_sentence_end": 4},
]


def test_unclassified_window_is_counted_not_fatal():
    baseline = baseline_module(
        "benchmarks/nativelong/evaluation/scripts/evaluate_comet.py", "evaluate_comet")
    row = {"doc_id": "d", "seg_id": 1, "src": "", "ref": "", "tgt": ""}
    assert ec.classify_window(row) == "unclassified"
    try:
        baseline.classify_window(row)
    except ValueError:
        pass
    else:  # pragma: no cover - guards the regression
        raise AssertionError("baseline was expected to abort on an unknown pattern")


def test_probe_without_windows_reports_status():
    baseline = baseline_module(
        "benchmarks/nativelong/evaluation/scripts/evaluate_comet.py", "evaluate_comet")
    probe = {"probe_id": "p1", "source_char_start": 0, "source_char_end": 3}
    result = ec.summarize_probe(MAPPED_ROWS, probe)
    assert result["status"] == "no_windows_selected"
    assert result["windows"] == 0 and result["comet"] is None
    try:
        baseline.summarize_probe(MAPPED_ROWS, probe)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("baseline was expected to abort on an empty probe")


def test_probe_character_basis_matches_row_coordinates():
    baseline = baseline_module(
        "benchmarks/nativelong/evaluation/scripts/evaluate_comet.py", "evaluate_comet")
    probe = {"probe_id": "p2", "source_char_start": 11, "source_char_end": 13}
    try:
        ec.summarize_probe(MAPPED_ROWS, probe)
    except ValueError as error:
        # 12 original characters, not the 14 of the space-joined window text.
        assert "source_total=12" in str(error), error
    else:  # pragma: no cover
        raise AssertionError("probe span beyond the original source must be rejected")
    try:
        baseline.summarize_probe(MAPPED_ROWS, probe)
    except ValueError as error:
        assert "source_total=14" not in str(error) and "source_total=12" not in str(error), error
    else:  # pragma: no cover
        raise AssertionError("baseline accepted the mixed coordinate basis")


def test_checkpoint_sha_is_observed_and_verified(tmp_path: Path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    observed = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert ec.resolve_checkpoint_sha256(checkpoint, None) == (None, observed)
    assert ec.resolve_checkpoint_sha256(checkpoint, observed) == (observed, observed)
    try:
        ec.resolve_checkpoint_sha256(checkpoint, "0" * 64)
    except ValueError as error:
        assert "mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a wrong declared checkpoint digest must fail the run")


def write_comet_run(directory: Path, summary: dict, *, completed: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "per_window.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                 for name in ("summary.json", "per_window.jsonl")}
    (directory / "artifact-manifest.json").write_text(json.dumps(artifacts), encoding="utf-8")
    if completed:
        (directory / "COMPLETED.json").write_text(json.dumps({
            "status": "completed",
            "artifact_manifest_sha256": hashlib.sha256(
                (directory / "artifact-manifest.json").read_bytes()).hexdigest(),
        }), encoding="utf-8")
    return directory / "summary.json"


def comet_summary(inputs: dict, cases: list[dict]) -> dict:
    return {"schema_version": "document-comet-v1", "suite_id": "s", "system_key": "k",
            "inputs": inputs, "cases": cases}


def test_comet_sidecar_validation(tmp_path: Path):
    baseline = baseline_module(
        "benchmarks/nativelong/evaluation/scripts/summarize_document_segale.py",
        "summarize_document_segale")
    assert not hasattr(baseline, "verify_comet_artifacts")
    source = tmp_path / "aligned.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    inputs = {"aligned_input": {"path": str(source),
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    run_dir = tmp_path / "comet"
    summary_path = write_comet_run(run_dir, comet_summary(inputs, []))
    sds.verify_comet_artifacts(summary_path, "s", "k")
    # Missing completed marker: a partial run must not be consumed.
    (run_dir / "COMPLETED.json").unlink()
    try:
        sds.verify_comet_artifacts(summary_path, "s", "k")
    except ValueError as error:
        assert "not completed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an incomplete COMET run must be rejected")
    # Tampered artifact.
    write_comet_run(run_dir, comet_summary(inputs, []))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    try:
        sds.verify_comet_artifacts(summary_path, "s", "k")
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a stale COMET artifact must be rejected")


def test_duplicate_comet_case_id_is_rejected(tmp_path: Path):
    source = tmp_path / "aligned.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    inputs = {"aligned_input": {"path": str(source),
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case_id": "a", "reference": "r", "metadata": {},
                                 "alignment_unit_count": 0}) + "\n", encoding="utf-8")
    mt = "t"
    generations = tmp_path / "generations.jsonl"
    generations.write_text(json.dumps({
        "case_id": "a", "mt": mt, "status": "ok",
        "output_sha256": hashlib.sha256(mt.encode()).hexdigest()}) + "\n", encoding="utf-8")
    summary_path = write_comet_run(tmp_path / "comet", comet_summary(inputs, [
        {"case_id": "a", "comet": 0.5}, {"case_id": "a", "comet": 0.9}]))
    argv = sys.argv
    sys.argv = ["summarize_document_segale.py", "--cases", str(cases),
                "--generations", str(generations), "--comet-summary", str(summary_path),
                "--output", str(tmp_path / "summary.json"), "--suite-id", "s",
                "--system-key", "k"]
    try:
        sds.main()
    except ValueError as error:
        assert "Duplicate case_id in COMET summary" in str(error), error
    else:  # pragma: no cover
        raise AssertionError("duplicate COMET case_id silently accepted")
    finally:
        sys.argv = argv


def test_empty_selection_comet_run_passes_validation(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("", encoding="utf-8")
    inputs = runner.file_inputs({"cases": cases})
    comet_dir = tmp_path / "comet"
    runner.write_empty_comet_run(comet_dir, inputs)
    sds.verify_comet_artifacts(comet_dir / "summary.json", runner.SUITE_ID, runner.SYSTEM_KEY)


def test_five_band_macro_exposes_scored_with_comet():
    macro = runner.five_band_macro({"groups": {"length_band": {
        "4k": {"segale_comet": 0.5, "scored_cases": 3, "scored_with_comet_cases": 2, "cases": 4},
        "8k": {"segale_comet": 0.4, "scored_cases": 2, "scored_with_comet_cases": 2, "cases": 2},
        "16k": {"segale_comet": 0.3, "scored_cases": 1, "scored_with_comet_cases": 1, "cases": 1},
        "32k": {"segale_comet": 0.2, "scored_cases": 1, "scored_with_comet_cases": 1, "cases": 1},
        "64k": {"segale_comet": 0.1, "scored_cases": 1, "scored_with_comet_cases": 0, "cases": 1}}}})
    assert macro["coverage"]["4k"] == {"scored": 3, "scored_with_comet": 2, "total": 4}
    assert macro["value"] is not None


def test_empty_selection_branch_no_longer_writes_bare_summary():
    baseline = baseline_module("benchmarks/nativelong/evaluation/run.py", "run")
    assert hasattr(baseline, "five_band_macro")
    assert not hasattr(baseline, "write_empty_comet_run")
    assert not hasattr(baseline, "file_inputs")


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        if "tmp_path" in function.__code__.co_varnames[:function.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as directory:
                try:
                    function(Path(directory))
                except Exception as error:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL {name}: {type(error).__name__}: {error}")
                else:
                    print(f"PASS {name}")
        else:
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"PASS {name}")
    print(f"{failures} failing test(s)")
    sys.exit(1 if failures else 0)
