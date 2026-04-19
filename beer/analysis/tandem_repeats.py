"""Tandem repeat, direct repeat, and low-complexity region detection."""
from __future__ import annotations
from collections import Counter

from beer.reports.css import make_style_tag


# ---------------------------------------------------------------------------
# Tandem repeat detection
# ---------------------------------------------------------------------------

def find_tandem_repeats(
    seq: str,
    min_unit: int = 2,
    max_unit: int = 8,
    min_copies: int = 2,
) -> list[dict]:
    """Find exact tandem repeats by k-mer extension.

    For every starting position and every unit length in [*min_unit*, *max_unit*],
    the repeat unit ``seq[i:i+k]`` is extended greedily as long as the next
    k-mer matches exactly.  Overlapping candidates are resolved by keeping the
    longest repeat (by total covered length).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    min_unit:
        Minimum repeat unit length (default 2).
    max_unit:
        Maximum repeat unit length (default 8).
    min_copies:
        Minimum number of consecutive copies (default 2).

    Returns
    -------
    list[dict]
        Each dict:

        ``start_1based``
            1-based start position of the repeat.
        ``end_1based``
            1-based end position (inclusive).
        ``unit``
            Repeat unit sequence.
        ``unit_length``
            Length of the repeat unit.
        ``n_copies``
            Number of exact consecutive copies.
        ``total_length``
            Total length covered (unit_length * n_copies).

    Sorted by *total_length* descending.  Overlapping repeats (same positions)
    are deduplicated; only the longest non-overlapping set is returned.
    """
    n = len(seq)
    candidates: list[dict] = []

    for k in range(min_unit, max_unit + 1):
        i = 0
        while i <= n - k * min_copies:
            unit = seq[i:i + k]
            copies = 1
            j = i + k
            while j + k <= n and seq[j:j + k] == unit:
                copies += 1
                j += k
            if copies >= min_copies:
                candidates.append({
                    'start_1based': i + 1,
                    'end_1based': i + k * copies,
                    'unit': unit,
                    'unit_length': k,
                    'n_copies': copies,
                    'total_length': k * copies,
                })
                # advance past this repeat to avoid partial overlaps at same k
                i += k * copies
            else:
                i += 1

    # Sort by total length descending
    candidates.sort(key=lambda d: d['total_length'], reverse=True)

    # Deduplicate: remove intervals that are fully contained in already-kept repeats
    kept: list[dict] = []
    used_positions: set[int] = set()
    for cand in candidates:
        s0 = cand['start_1based'] - 1  # 0-based
        e0 = cand['end_1based']        # 0-based exclusive
        pos_set = set(range(s0, e0))
        # keep only if it has zero positional overlap with already-kept intervals
        # (strict non-overlapping interval selection)
        if not pos_set.intersection(used_positions):
            kept.append(cand)
            used_positions.update(pos_set)

    kept.sort(key=lambda d: d['total_length'], reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Direct repeat detection
# ---------------------------------------------------------------------------

def find_direct_repeats(
    seq: str,
    min_length: int = 4,
    max_gap: int = 30,
) -> list[dict]:
    """Find pairs of identical subsequences separated by <= *max_gap* residues.

    Uses an O(n^2) approach: for each position *i*, look ahead to positions
    *j* within ``[i + min_length, i + min_length + max_gap]`` and check
    whether ``seq[i:i+L]`` == ``seq[j:j+L]`` for increasing L.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    min_length:
        Minimum repeat length to report (default 4).
    max_gap:
        Maximum gap (in residues) between the two copies (default 30).

    Returns
    -------
    list[dict]
        Each dict:

        ``seq1_start_1based``
            1-based start of the first copy.
        ``seq2_start_1based``
            1-based start of the second copy.
        ``repeat_seq``
            Shared repeat sequence.
        ``length``
            Length of the shared sequence.
        ``gap``
            Number of residues between the two copies.
    """
    n = len(seq)
    results: list[dict] = []

    for i in range(n - min_length):
        for j in range(i + min_length, min(i + min_length + max_gap + 1, n - min_length + 1)):
            gap = j - i - min_length  # gap after end of first copy
            # Actually gap = j - (i + L); we'll compute for the actual matched length
            # Extend match
            L = 0
            max_L = min(n - i, n - j)
            while L < max_L and seq[i + L] == seq[j + L]:
                L += 1
            if L >= min_length:
                actual_gap = j - (i + L)
                if actual_gap < 0:
                    # copies overlap — skip
                    continue
                if actual_gap > max_gap:
                    continue
                results.append({
                    'seq1_start_1based': i + 1,
                    'seq2_start_1based': j + 1,
                    'repeat_seq': seq[i:i + L],
                    'length': L,
                    'gap': actual_gap,
                })

    # Remove duplicates (same seq2_start, same repeat)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in results:
        key = (r['seq1_start_1based'], r['seq2_start_1based'], r['repeat_seq'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda d: d['length'], reverse=True)
    return unique


# ---------------------------------------------------------------------------
# Compositional (low-complexity) repeat detection
# ---------------------------------------------------------------------------

def find_compositional_repeats(
    seq: str,
    window: int = 10,
    step: int = 5,
    n_top: int = 3,
) -> list[dict]:
    """Find windows with unusually low compositional diversity.

    A window is flagged as low-complexity (LC) if the fraction contributed by
    the top-*n_top* most frequent amino acids exceeds 0.70.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window length (default 10).
    step:
        Step size between window starts (default 5).
    n_top:
        Number of most-frequent amino acids to consider (default 3).

    Returns
    -------
    list[dict]
        Each dict:

        ``start``
            0-based start of window.
        ``end``
            0-based exclusive end.
        ``dominant_aa``
            The single most frequent amino acid in the window.
        ``fraction``
            Combined fraction of the top-*n_top* amino acids.
        ``seq_window``
            The window sub-sequence.
    """
    n = len(seq)
    results: list[dict] = []

    for i in range(0, n - window + 1, step):
        sub = seq[i:i + window]
        counts = Counter(sub)
        top_counts = sorted(counts.values(), reverse=True)[:n_top]
        top_fraction = sum(top_counts) / window
        if top_fraction > 0.70:
            dominant_aa = counts.most_common(1)[0][0]
            results.append({
                'start': i,
                'end': i + window,
                'dominant_aa': dominant_aa,
                'fraction': round(top_fraction, 3),
                'seq_window': sub,
            })

    # Merge overlapping LC windows
    if not results:
        return results

    merged: list[dict] = [results[0]]
    for r in results[1:]:
        prev = merged[-1]
        if r['start'] <= prev['end']:
            # Extend previous window
            merged[-1] = {
                'start': prev['start'],
                'end': r['end'],
                'dominant_aa': prev['dominant_aa'],
                'fraction': max(prev['fraction'], r['fraction']),
                'seq_window': seq[prev['start']:r['end']],
            }
        else:
            merged.append(r)

    return merged


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def calc_repeat_stats(seq: str) -> dict:
    """Compute comprehensive repeat statistics for a protein sequence.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``n_tandem_repeats``
            Total number of non-overlapping tandem repeats.
        ``total_tandem_coverage``
            Fraction of sequence covered by tandem repeats.
        ``n_direct_repeats``
            Number of direct-repeat pairs (capped at reporting first 100).
        ``largest_tandem_unit``
            Repeat unit of the longest tandem repeat.
        ``largest_tandem_copies``
            Copy number of the longest tandem repeat.
        ``tandem_repeats``
            Full list from :func:`find_tandem_repeats`.
        ``direct_repeats``
            First 10 entries from :func:`find_direct_repeats`.
    """
    n = len(seq)
    tandem = find_tandem_repeats(seq)
    direct = find_direct_repeats(seq)

    # Coverage
    covered: set[int] = set()
    for tr in tandem:
        covered.update(range(tr['start_1based'] - 1, tr['end_1based']))
    coverage = len(covered) / n if n > 0 else 0.0

    largest = tandem[0] if tandem else None

    return {
        'n_tandem_repeats': len(tandem),
        'total_tandem_coverage': round(coverage, 4),
        'n_direct_repeats': len(direct),
        'largest_tandem_unit': largest['unit'] if largest else '',
        'largest_tandem_copies': largest['n_copies'] if largest else 0,
        'tandem_repeats': tandem,
        'direct_repeats': direct[:10],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_repeats_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for tandem and direct repeat analysis.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``).

    Returns
    -------
    str
        Self-contained HTML fragment (includes ``<style>`` block).
    """
    accent = style_tag if style_tag else "#4361ee"
    _s = make_style_tag(accent)

    stats = calc_repeat_stats(seq)
    lc = find_compositional_repeats(seq)

    summary_rows = (
        f"<tr><td>Tandem repeats</td><td>{stats['n_tandem_repeats']}</td></tr>"
        f"<tr><td>Tandem repeat coverage</td>"
        f"<td>{stats['total_tandem_coverage']:.1%}</td></tr>"
        f"<tr><td>Largest tandem unit</td>"
        f"<td><code>{stats['largest_tandem_unit'] or 'N/A'}</code> "
        f"({stats['largest_tandem_copies']} copies)</td></tr>"
        f"<tr><td>Direct repeats (&ge;4 aa, gap &le;30)</td>"
        f"<td>{stats['n_direct_repeats']}</td></tr>"
        f"<tr><td>Low-complexity windows</td><td>{len(lc)}</td></tr>"
    )

    summary_html = (
        "<h2>Sequence Repeats</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
    )

    # Tandem repeat table
    if stats['tandem_repeats']:
        tr_header = (
            "<tr><th>Start</th><th>End</th><th>Unit</th>"
            "<th>Unit length</th><th>Copies</th><th>Total length</th></tr>"
        )
        tr_rows = "".join(
            f"<tr>"
            f"<td>{tr['start_1based']}</td>"
            f"<td>{tr['end_1based']}</td>"
            f"<td><code>{tr['unit']}</code></td>"
            f"<td>{tr['unit_length']}</td>"
            f"<td>{tr['n_copies']}</td>"
            f"<td>{tr['total_length']}</td>"
            f"</tr>"
            for tr in stats['tandem_repeats']
        )
        tandem_html = (
            "<h2>Tandem Repeats</h2>"
            "<table>"
            f"{tr_header}{tr_rows}"
            "</table>"
            "<p class='note'>"
            "Exact k-mer extension algorithm; unit lengths 2&ndash;8; "
            "minimum 2 copies; non-overlapping."
            "</p>"
        )
    else:
        tandem_html = (
            "<h2>Tandem Repeats</h2>"
            "<p>No exact tandem repeats found "
            "(unit 2&ndash;8 aa, &ge;2 copies).</p>"
        )

    # Direct repeat table (first 10)
    if stats['direct_repeats']:
        dr_header = (
            "<tr><th>Copy 1 start</th><th>Copy 2 start</th>"
            "<th>Sequence</th><th>Length</th><th>Gap</th></tr>"
        )
        dr_rows = "".join(
            f"<tr>"
            f"<td>{dr['seq1_start_1based']}</td>"
            f"<td>{dr['seq2_start_1based']}</td>"
            f"<td><code>{dr['repeat_seq']}</code></td>"
            f"<td>{dr['length']}</td>"
            f"<td>{dr['gap']}</td>"
            f"</tr>"
            for dr in stats['direct_repeats']
        )
        direct_html = (
            "<h2>Direct Repeats (top 10 by length)</h2>"
            "<table>"
            f"{dr_header}{dr_rows}"
            "</table>"
            "<p class='note'>"
            "Direct repeats: identical sequence pairs separated by &le;30 residues, "
            "minimum match length 4 aa."
            "</p>"
        )
    else:
        direct_html = (
            "<h2>Direct Repeats</h2>"
            "<p>No direct repeats found (&ge;4 aa, gap &le;30).</p>"
        )

    # LC table
    if lc:
        lc_header = (
            "<tr><th>Start</th><th>End</th>"
            "<th>Dominant AA</th><th>Top-3 fraction</th><th>Sequence</th></tr>"
        )
        lc_rows = "".join(
            f"<tr>"
            f"<td>{w['start'] + 1}</td>"
            f"<td>{w['end']}</td>"
            f"<td>{w['dominant_aa']}</td>"
            f"<td>{w['fraction']:.1%}</td>"
            f"<td><code>{w['seq_window']}</code></td>"
            f"</tr>"
            for w in lc
        )
        lc_html = (
            "<h2>Low-Complexity Windows</h2>"
            "<table>"
            f"{lc_header}{lc_rows}"
            "</table>"
            "<p class='note'>"
            "Heuristic: window = 10 aa, step = 5; flagged when the three most "
            "frequent residues account for &gt;70% of the window. "
            "Low-complexity detection uses a compositional heuristic "
            "(top-3 residue frequency &gt; 70% in a 10-residue window). "
            "For rigorous LC analysis use Shannon entropy or SEG "
            "(Wootton &amp; Federhen 1993)."
            "</p>"
        )
    else:
        lc_html = (
            "<h2>Low-Complexity Windows</h2>"
            "<p>No low-complexity windows detected.</p>"
            "<p class='note'>"
            "Low-complexity detection uses a compositional heuristic "
            "(top-3 residue frequency &gt; 70% in a 10-residue window). "
            "For rigorous LC analysis use Shannon entropy or SEG "
            "(Wootton &amp; Federhen 1993)."
            "</p>"
        )

    return _s + summary_html + tandem_html + direct_html + lc_html


# Alias expected by beer.py
format_tandem_repeats_report = format_repeats_report
