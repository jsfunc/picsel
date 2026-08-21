import subprocess
import sys
from pathlib import Path

from tamis import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_flag_prints_version_and_exits_without_opening_a_window():
    result = subprocess.run(
        [sys.executable, "main.py", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    # The version is the first line; the optional-extra report follows it.
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"Tamis {__version__}"
    assert any(line.startswith("Quality scoring:") for line in lines[1:])


def test_version_flag_reports_which_optional_extras_are_present(capsys):
    """Both optional features hide themselves when their dependencies are
    absent, which is right for the user and useless for telling a
    correctly-lean build apart from a packaged one that was meant to include
    them. Two releases shipped without quality scoring and looked fine.
    """
    from tamis.features import summary

    lines = summary()
    assert len(lines) == 2
    assert lines[0].startswith("Face recognition: ")
    assert lines[1].startswith("Quality scoring:")
    assert all(line.endswith(("enabled", "not installed")) for line in lines)


def test_feature_report_agrees_with_the_guards_the_app_actually_uses():
    # find_spec is a proxy for MainWindow's guarded imports; if the two ever
    # disagree, --version would report a feature the app does not offer.
    import tamis.main_window as mw_module
    from tamis.features import quality_available, recognition_available

    assert recognition_available() == mw_module.RECOGNITION_AVAILABLE
    assert quality_available() == mw_module.QUALITY_AVAILABLE
