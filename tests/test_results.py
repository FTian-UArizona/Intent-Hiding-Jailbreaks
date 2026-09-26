"""CPU-only regressions against the archived paper tables.

Run from the repository root: python -m unittest discover -s tests -v.
These tests need only requirements-analysis.txt; they run no models or API calls.
"""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_results", ROOT / "scripts/plot_results.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Cannot load scripts/plot_results.py")
plot_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_results)

REFERENCE_FILES = (
    "table_cumulative_asr_32b.csv",
    "table_incremental_gains_32b.csv",
    "table_cumulative_asr_14b_with_32b_k1.csv",
)


class PublishedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outcomes = plot_results.load_outcomes(ROOT / "data/processed/target_outcomes.csv")
        with redirect_stdout(io.StringIO()):
            cls.summaries = plot_results.build_summaries(cls.outcomes)

    def test_exported_tables_match_archived_results(self):
        """Recompute from outcome counts, then compare all exported rows/columns."""
        with tempfile.TemporaryDirectory() as directory:
            exported = plot_results.export_summary_tables(self.summaries, directory)
            self.assertEqual(set(exported), set(REFERENCE_FILES))
            for filename in REFERENCE_FILES:
                with self.subTest(table=filename):
                    actual = pd.read_csv(Path(directory) / filename, float_precision="round_trip")
                    expected = pd.read_csv(
                        ROOT / "data/reference_tables" / filename,
                        float_precision="round_trip",
                    )
                    # Row/column order, labels, dtypes, and integer counts must
                    # agree. Only binary-float serialization roundoff is allowed
                    # (the archived incremental table contains e.g. 0.93 - 1e-16).
                    pd.testing.assert_frame_equal(
                        actual, expected, check_exact=False, rtol=0, atol=1e-12,
                    )

    def test_figure_d_starts_at_each_models_32b_baseline(self):
        """Both generator curves must share the downstream model's direct result."""
        for model, corrected in self.summaries["cumulative_14b"].items():
            with self.subTest(model=model):
                baseline = self.summaries["cumulative_32b"][model]
                pd.testing.assert_frame_equal(
                    corrected.loc[corrected["K"].eq(1)].reset_index(drop=True),
                    baseline.loc[baseline["K"].eq(1)].reset_index(drop=True),
                    check_exact=True,
                )

    def test_raw_14b_direct_results_do_not_change_corrected_curve(self):
        """Baseline replacement must affect cumulative unions at every K."""
        changed = self.outcomes.copy()
        raw_direct = changed["query_generator"].eq("Qwen3-14B") & changed["total_bundle_size"].eq(1)
        self.assertTrue(raw_direct.any())
        changed.loc[raw_direct, "n_success"] = 1 - changed.loc[raw_direct, "n_success"]
        with redirect_stdout(io.StringIO()):
            recomputed = plot_results.build_summaries(changed)
        for model, expected in self.summaries["cumulative_14b"].items():
            with self.subTest(model=model):
                pd.testing.assert_frame_equal(
                    recomputed["cumulative_14b"][model], expected, check_exact=True,
                )


if __name__ == "__main__":
    unittest.main()
