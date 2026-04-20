"""Local TMHMM 2.0 TM-helix predictor using bundled model parameters.

Runs entirely offline — no internet, no external binary.

Model parameters
----------------
Bundled from the TMHMM 2.0 distribution (Krogh et al. 2001, J. Mol. Biol.
305:567-580). The TMHMM2.0.model file is redistributed here from the
tmhmm.py package (MIT licence, Dan Søndergaard,
https://github.com/dansondergaard/tmhmm.py).  The original TMHMM software
is available under a DTU academic non-commercial licence from
https://services.healthtech.dtu.dk/services/TMHMM-2.0/.

Algorithm
---------
Viterbi decoding (log-space, vectorised with NumPy) of the 395-state
profile HMM. States are labelled i (inside/cytoplasm), o (outside/
extracellular), M (membrane/TM helix).  Topology is inferred from the
positive-inside rule after decoding.
"""
from __future__ import annotations

import collections
import re
import importlib.resources
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# TMHMM 2.0 model parser  (adapted from tmhmm.py, MIT licence)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r'([A-Za-z0-9.\-_]+|[:;{}])', text)


def _strip_comments(file_like) -> str:
    return ''.join(line for line in file_like if not line.startswith('#'))


def _parse_map(tokens: collections.deque):
    d: dict = collections.OrderedDict()
    while True:
        tok = tokens.popleft()
        if tok == ';':
            tokens.appendleft(tok)
            return tokens, d
        nxt = tokens.popleft()
        if nxt != ':':
            tokens.appendleft(nxt)
            tokens.appendleft(tok)
            return tokens, None
        d[tok] = float(tokens.popleft())


def _parse_list(tokens: collections.deque):
    lst = []
    while True:
        tok = tokens.popleft()
        if tok == ';':
            tokens.appendleft(tok)
            return tokens, lst
        lst.append(tok)


def _parse_state(tokens: collections.deque):
    name = tokens.popleft()
    tokens.popleft()  # "{"
    state: dict = {}
    while True:
        tok = tokens.popleft()
        if tok == '}':
            return tokens, (name, state)
        if tok in ('trans', 'only'):
            tokens, val = _parse_map(tokens)
            if val is None:
                tokens, val = _parse_list(tokens)
        elif tok in ('type', 'end'):
            val = int(tokens.popleft())
        else:
            val = tokens.popleft()
        state[tok] = val
        tokens.popleft()  # ";"


def _parse_header(tokens: collections.deque):
    tokens.popleft()  # "header"
    tokens.popleft()  # "{"
    header: dict = {}
    while True:
        tok = tokens.popleft()
        if tok == '}':
            break
        header[tok] = tokens.popleft()
        tokens.popleft()  # ";"
    return tokens, header


def _normalize_states(states: dict) -> dict:
    for name, state in states.items():
        if 'tied_trans' in state:
            parent = states[state['tied_trans']]
            state['trans'] = dict(zip(state['trans'], parent['trans'].values()))
        if 'tied_letter' in state:
            state['only'] = dict(states[state['tied_letter']]['only'])
    return states


def _to_matrices(alphabet: str, states: dict):
    begin = dict(states.pop('begin'))
    state_map = {v: k for k, v in enumerate(states)}
    char_map  = {v: k for k, v in enumerate(alphabet)}
    S, C = len(states), len(alphabet)
    initial    = np.zeros(S)
    trans_mat  = np.zeros((S, S))
    emit_mat   = np.zeros((S, C))
    label_map: dict[int, str] = {}
    for sname, p in begin['trans'].items():
        initial[state_map[sname]] = p
    for sname, sdict in states.items():
        si = state_map[sname]
        if 'label' in sdict:
            label_map[si] = sdict['label']
        for oname, p in sdict.get('trans', {}).items():
            trans_mat[si, state_map[oname]] = p
        for ch, p in sdict.get('only', {}).items():
            emit_mat[si, char_map[ch]] = p
    return initial, trans_mat, emit_mat, char_map, label_map


def _parse_model(file_like):
    contents = _strip_comments(file_like)
    tokens   = collections.deque(_tokenize(contents))
    tokens, header = _parse_header(tokens)
    states: dict = {}
    while tokens:
        tokens, (n, s) = _parse_state(tokens)
        states[n] = s
    return _to_matrices(header['alphabet'], _normalize_states(states))


# ---------------------------------------------------------------------------
# Cached model (loaded once per process)
# ---------------------------------------------------------------------------

_MODEL = None  # (initial, trans, emit, char_map, label_map)


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        ref = importlib.resources.files("beer.models").joinpath("TMHMM2.0.model")
        with importlib.resources.as_file(ref) as p:
            with open(p) as fh:
                _MODEL = _parse_model(fh)
    except Exception:
        p = pathlib.Path(__file__).parent.parent / "models" / "TMHMM2.0.model"
        with open(p) as fh:
            _MODEL = _parse_model(fh)
    return _MODEL


# ---------------------------------------------------------------------------
# Viterbi (log-space, NumPy-vectorised over states)
# ---------------------------------------------------------------------------

def _viterbi(seq: str, initial, trans, emit, char_map: dict, label_map: dict) -> str:
    seq = seq.upper()
    valid = set(char_map)
    seq = ''.join(c if c in valid else 'A' for c in seq)  # map unknowns to Ala

    N, S = len(seq), len(initial)

    # Use -inf for zero probabilities so forbidden transitions stay impossible
    # regardless of sequence length (avoids epsilon accumulation).
    # Suppress divide-by-zero warning: np.where evaluates both branches eagerly.
    with np.errstate(divide='ignore'):
        l_init  = np.where(initial > 0, np.log(initial), -np.inf)
        l_trans = np.where(trans   > 0, np.log(trans),   -np.inf)  # (S, S)
        l_emit  = np.where(emit    > 0, np.log(emit),    -np.inf)  # (S, C)

    M = l_init + l_emit[:, char_map[seq[0]]]   # (S,) log-prob of best path to each state
    P = np.zeros((N, S), dtype=np.int32)        # backpointers

    for i in range(1, N):
        # scores[k, j] = M[k] + l_trans[k, j]  →  best predecessor k for every j
        scores = M[:, np.newaxis] + l_trans     # (S, S)
        P[i]   = np.argmax(scores, axis=0)      # (S,)
        M      = np.max(scores, axis=0) + l_emit[:, char_map[seq[i]]]

    # Backtrack
    path  = []
    state = int(np.argmax(M))
    for i in range(N - 1, -1, -1):
        path.append(label_map.get(state, 'o'))
        state = P[i, state]
    return ''.join(reversed(path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_tm_helices(sequence: str) -> list[dict]:
    """Run TMHMM 2.0 Viterbi on *sequence*; return a list of TM helix dicts.

    Each dict: start (int, 0-based), end (int, 0-based inclusive),
               score (float), orientation (str), source (str).
    """
    initial, trans, emit, char_map, label_map = _load_model()
    path = _viterbi(sequence, initial, trans, emit, char_map, label_map)

    helices: list[dict] = []
    in_h = False
    h_start = 0
    for i, lbl in enumerate(path):
        if lbl == 'M' and not in_h:
            in_h, h_start = True, i
        elif lbl != 'M' and in_h:
            in_h = False
            helices.append(_helix_dict(sequence, path, h_start, i - 1))
    if in_h:
        helices.append(_helix_dict(sequence, path, h_start, len(sequence) - 1))
    return helices


def _helix_dict(seq: str, path: str, start: int, end: int) -> dict:
    """Build a TM helix dict and assign orientation via inside-positive rule."""
    flank = 15
    n = len(seq)
    n_flank = seq[max(0, start - flank): start]
    c_flank = seq[end + 1: min(n, end + 1 + flank)]
    n_kr = sum(1 for aa in n_flank if aa in 'KR')
    c_kr = sum(1 for aa in c_flank if aa in 'KR')
    # More K/R on N-terminal side → that side is cytoplasmic (inside)
    orientation = "N-in" if n_kr >= c_kr else "N-out"
    seg = seq[start: end + 1]
    score = round(sum(seg.count(aa) for aa in 'AILMFVWY') / max(len(seg), 1), 3)
    return {
        "start": start,
        "end": end,
        "score": score,
        "orientation": orientation,
        "source": "TMHMM 2.0",
    }
