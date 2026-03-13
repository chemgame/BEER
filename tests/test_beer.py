"""Unit tests for BEER helper functions (pure-Python; no Qt/matplotlib/BioPython needed)."""
import sys, os, math, unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavyweight dependencies before importing beer
# ---------------------------------------------------------------------------
_numpy_mock = MagicMock()
_numpy_mock.zeros = lambda shape: [[0.0]*shape[1] for _ in range(shape[0])]
_numpy_mock.linspace = lambda a, b, n: [a + (b-a)*i/(n-1) for i in range(n)]
sys.modules.setdefault('numpy', _numpy_mock)
sys.modules.setdefault('numpy.core', MagicMock())

for _mod in [
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.figure',
    'matplotlib.backends', 'matplotlib.backends.backend_qt5agg',
    'matplotlib.patches', 'matplotlib.cm', 'matplotlib.colors',
    'mplcursors',
    'Bio', 'Bio.SeqUtils', 'Bio.SeqUtils.ProtParam',
    'Bio.SeqIO', 'Bio.PDB', 'Bio.PDB.Polypeptide', 'Bio.SeqUtils.seq1',
    'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtGui', 'PyQt5.QtCore',
    'PyQt5.QtPrintSupport',
]:
    sys.modules.setdefault(_mod, MagicMock())

# Make Bio.SeqUtils.ProtParam.ProteinAnalysis import-able
from unittest.mock import MagicMock as _MM
sys.modules['Bio.SeqUtils.ProtParam'] = _MM()
sys.modules['Bio.SeqUtils.ProtParam'].ProteinAnalysis = _MM

# Now import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib
beer_spec = importlib.util.spec_from_file_location(
    'beer', os.path.join(os.path.dirname(__file__), '..', 'beer.py')
)
beer_mod = importlib.util.module_from_spec(beer_spec)
# Inject stubs into the module namespace before exec
beer_mod.np = _numpy_mock
try:
    beer_spec.loader.exec_module(beer_mod)
except Exception:
    pass  # GUI classes may fail without display; helpers are already loaded
import beer as B

# ---------------------------------------------------------------------------
# Sequences for testing
# ---------------------------------------------------------------------------
SEQ_ALL20  = "ACDEFGHIKLMNPQRSTVWY"
SEQ_POLY_K = "KKKKKKKKKK"
SEQ_POLY_D = "DDDDDDDDDD"
SEQ_HELIX  = "AAAAAAAAAA"          # Ala → strongest helix former
SEQ_SHEET  = "VVVVVVVVVV"          # Val → strong sheet former
SEQ_REPEAT = "AAAAAAAAAAAAA"        # pure repeat → max LC


# ---------------------------------------------------------------------------
class TestCleanSequence(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(B.clean_sequence("  ACD  "), "ACD")

    def test_uppercases(self):
        self.assertEqual(B.clean_sequence("acd"), "ACD")

    def test_removes_internal_spaces(self):
        self.assertEqual(B.clean_sequence("A C D"), "ACD")


class TestIsValidProtein(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(B.is_valid_protein(SEQ_ALL20))

    def test_invalid_chars(self):
        self.assertFalse(B.is_valid_protein("ABCXZ"))

    def test_empty_is_vacuously_true(self):
        self.assertTrue(B.is_valid_protein(""))


class TestCalcNetCharge(unittest.TestCase):
    def test_polyK_positive_at_7(self):
        self.assertGreater(B.calc_net_charge(SEQ_POLY_K, 7.0), 0)

    def test_polyD_negative_at_7(self):
        self.assertLess(B.calc_net_charge(SEQ_POLY_D, 7.0), 0)

    def test_neutral_peptide_mixed(self):
        # A peptide with equal K and D should be near-neutral at pH near pI
        charge = B.calc_net_charge("KD", 7.0)
        self.assertAlmostEqual(charge, 0.0, delta=0.5)

    def test_custom_pka_changes_result(self):
        pka = dict(B.DEFAULT_PKA)
        pka['K'] = 8.0
        charge_default = B.calc_net_charge(SEQ_POLY_K, 7.0)
        charge_custom  = B.calc_net_charge(SEQ_POLY_K, 7.0, pka)
        self.assertLess(charge_custom, charge_default)


class TestSlidingWindowHydrophobicity(unittest.TestCase):
    def test_short_seq_returns_one_value(self):
        result = B.sliding_window_hydrophobicity("ACDE", window_size=9)
        self.assertEqual(len(result), 1)

    def test_normal_length(self):
        result = B.sliding_window_hydrophobicity(SEQ_ALL20, window_size=5)
        self.assertEqual(len(result), len(SEQ_ALL20) - 5 + 1)

    def test_hydrophobic_positive(self):
        result = B.sliding_window_hydrophobicity("IIIIIIIII", window_size=9)
        self.assertGreater(result[0], 0)

    def test_hydrophilic_negative(self):
        result = B.sliding_window_hydrophobicity("DDDDDDDDD", window_size=9)
        self.assertLess(result[0], 0)


class TestCalcShannonEntropy(unittest.TestCase):
    def test_single_aa_zero_entropy(self):
        self.assertAlmostEqual(B.calc_shannon_entropy("AAAAAAA"), 0.0)

    def test_all20_max_entropy(self):
        e = B.calc_shannon_entropy(SEQ_ALL20)
        self.assertAlmostEqual(e, math.log2(20), delta=0.01)

    def test_two_equal_halves(self):
        e = B.calc_shannon_entropy("ADADADADAD")
        self.assertAlmostEqual(e, 1.0, delta=0.01)


class TestCalcKappa(unittest.TestCase):
    def test_well_mixed_low(self):
        seq = "KDKDKDKDKDKDKDKDKDKD"
        self.assertLess(B.calc_kappa(seq), 0.35)

    def test_segregated_high(self):
        seq = "K" * 10 + "D" * 10
        self.assertGreater(B.calc_kappa(seq), 0.45)

    def test_no_charge_zero(self):
        self.assertAlmostEqual(B.calc_kappa("GGGGGGGG"), 0.0)


class TestCalcOmega(unittest.TestCase):
    def test_clustered_stickers(self):
        seq = "F" * 10 + "G" * 10
        self.assertGreater(B.calc_omega(seq), 0.4)

    def test_no_stickers(self):
        self.assertAlmostEqual(B.calc_omega("GGGGGGGG"), 0.0)


class TestFractionLowComplexity(unittest.TestCase):
    def test_pure_repeat_is_one(self):
        self.assertAlmostEqual(B.fraction_low_complexity(SEQ_REPEAT), 1.0)

    def test_high_diversity_low(self):
        seq  = SEQ_ALL20 * 4
        frac = B.fraction_low_complexity(seq, threshold=2.0)
        self.assertLess(frac, 0.4)


class TestStickerSpacingStats(unittest.TestCase):
    def test_no_stickers_none(self):
        stats = B.sticker_spacing_stats("AAAAAAAAA")
        self.assertIsNone(stats["mean"])

    def test_regular_spacing(self):
        # F at positions 0,2,4,6 → gaps of 2
        stats = B.sticker_spacing_stats("AGFAFAFAG")
        self.assertIsNotNone(stats["mean"])
        self.assertGreater(stats["mean"], 0)


class TestFormatSequenceBlock(unittest.TestCase):
    def test_header_present(self):
        out = B.format_sequence_block(SEQ_ALL20, name="test")
        self.assertIn(">test", out)

    def test_position_1(self):
        out = B.format_sequence_block(SEQ_ALL20)
        self.assertIn("1", out)

    def test_groups_10_aa(self):
        out = B.format_sequence_block("A" * 10)
        self.assertIn("AAAAAAAAAA", out)

    def test_60_wide_two_lines(self):
        out = B.format_sequence_block("A" * 70)
        lines = [l for l in out.split("\n") if l.strip()]
        self.assertGreaterEqual(len(lines), 2)


class TestChouFasmanProfile(unittest.TestCase):
    def test_lengths(self):
        h, s = B.calc_chou_fasman_profile(SEQ_ALL20)
        self.assertEqual(len(h), len(SEQ_ALL20))
        self.assertEqual(len(s), len(SEQ_ALL20))

    def test_values_positive(self):
        h, s = B.calc_chou_fasman_profile("AAAVVV")
        for v in h + s:
            self.assertGreater(v, 0)

    def test_ala_helix_value(self):
        h, _ = B.calc_chou_fasman_profile("A")
        self.assertAlmostEqual(h[0], 1.42)

    def test_val_sheet_value(self):
        _, s = B.calc_chou_fasman_profile("V")
        self.assertAlmostEqual(s[0], 1.70)

    def test_helix_dominant_in_poly_ala(self):
        h, s = B.calc_chou_fasman_profile(SEQ_HELIX)
        self.assertGreater(sum(h), sum(s))


class TestDisorderProfile(unittest.TestCase):
    def test_length(self):
        d = B.calc_disorder_profile(SEQ_ALL20)
        self.assertEqual(len(d), len(SEQ_ALL20))

    def test_range_0_to_1(self):
        d = B.calc_disorder_profile(SEQ_ALL20)
        for v in d:
            self.assertGreaterEqual(v, 0.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_proline_more_disordered_than_trp(self):
        # Mix P (highest disorder propensity) and W (lowest) so normalization works
        d = B.calc_disorder_profile("PWPWPWPW", window=1)
        # P positions (even indices) should score higher than W (odd indices)
        self.assertGreater(d[0], d[1])

    def test_short_seq_no_crash(self):
        d = B.calc_disorder_profile("A", window=9)
        self.assertEqual(len(d), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
