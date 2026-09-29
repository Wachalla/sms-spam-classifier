import json
import os
import shutil

import run_all
import spam_classifier


def test_run_all_end_to_end(tmp_path):
    readme = tmp_path / "README.md"
    shutil.copy("README.md", readme)
    metrics = run_all.main(readme=str(readme))

    for name in ["audit.md", "metrics.json", "report.md", "casebook.csv"]:
        assert os.path.getsize(os.path.join("results", name)) > 0

    with open("results/metrics.json") as f:
        on_disk = json.load(f)
    for key in ["audit", "comparison", "v2_holdout", "v2_cross_validation", "original_split_seen_vs_unseen", "casebook"]:
        assert key in on_disk
    assert metrics["v2_holdout"]["train_test_overlap"] == 0
    assert "<!-- RESULTS:START -->" in readme.read_text()


def test_spam_classifier_script_runs():
    spam_classifier.main(save=False)
