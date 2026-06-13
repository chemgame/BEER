"""MolProbity Top8000 Ramachandran: grid loading, class detection, scoring."""
import matplotlib
matplotlib.use("Agg")
from beer.graphs.structural import _load_rama_grids, _rama_eval, _rama_classes


def test_grids_load_with_all_classes():
    g = _load_rama_grids()
    assert g is not None
    classes = {k.split("__")[0] for k in g if "__grid" in k}
    for c in ("General", "Glycine", "Trans-proline", "Pre-proline", "Ile-Val"):
        assert c in classes


def test_favored_and_outlier_regions():
    g = _load_rama_grids()
    # Canonical right-handed alpha-helix is favored.
    assert _rama_eval(-63, -42, "General", g) == "favored"
    # Beta-sheet basin is favored.
    assert _rama_eval(-120, 130, "General", g) == "favored"
    # Sterically forbidden region is an outlier.
    assert _rama_eval(0, 0, "General", g) == "outlier"
    # Glycine allows the left-handed region.
    assert _rama_eval(60, 40, "Glycine", g) == "favored"


def test_residue_class_detection():
    data = [
        {"resname": "ALA", "chain_id": "A"},
        {"resname": "GLY", "chain_id": "A"},
        {"resname": "ALA", "chain_id": "A"},   # pre-proline (next is PRO)
        {"resname": "PRO", "chain_id": "A"},
        {"resname": "VAL", "chain_id": "A"},
    ]
    cls = _rama_classes(data)
    assert cls == ["General", "Glycine", "Pre-proline", "Trans-proline", "Ile-Val"]
