"""Structure prediction utilities."""
from __future__ import annotations
import math
import re
from collections import Counter

from beer.constants import (
    KYTE_DOOLITTLE,
    DISORDER_PROPENSITY,
    COILS_MTIDK,
    COILS_BG_LOG,
    LINEAR_MOTIFS,
    LARKS_AROMATIC,
    LARKS_LC,
    STICKER_ALL,
)


_DPROP_MIN = min(DISORDER_PROPENSITY.values())   # W = -0.884
_DPROP_MAX = max(DISORDER_PROPENSITY.values())   # P =  0.987
_DPROP_SPAN = _DPROP_MAX - _DPROP_MIN


def _classical_disorder_profile(seq: str, window: int = 9) -> list:
    """Classical sliding-window disorder propensity, normalised to 0-1.

    Normalisation uses the global min/max of the DISORDER_PROPENSITY scale
    (W = -0.884, P = 0.987) so that absolute scores are comparable across
    sequences.  Per-sequence normalisation would make a poly-W sequence look
    as disordered as a poly-P sequence, which is incorrect.
    """
    raw = [DISORDER_PROPENSITY.get(aa, 0.0) for aa in seq]
    norm = [(v - _DPROP_MIN) / _DPROP_SPAN for v in raw]
    n = len(norm)
    if n < window:
        return norm
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(norm[lo:hi]) / (hi - lo))
    return smoothed


def _lstm_step(x, h, c, W_ih, W_hh, b_ih, b_hh):
    """Single LSTM cell step — numpy implementation."""
    import numpy as np
    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    H = len(h)
    def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    i_g = sigmoid(gates[:H]);        f_g = sigmoid(gates[H:2*H])
    g_g = np.tanh(gates[2*H:3*H]);   o_g = sigmoid(gates[3*H:])
    c_new = f_g * c + i_g * g_g
    h_new = o_g * np.tanh(c_new)
    return h_new, c_new


def _bilstm_forward(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """Full BiLSTM forward pass for one sequence using exported PyTorch weights.

    Supports arbitrary num_layers and hidden size as saved by train_all_heads.py.
    """
    import numpy as np
    n_layers = int(head.get("lstm_layers", np.array(2)).item()
                   if hasattr(head.get("lstm_layers", 2), "item")
                   else head.get("lstm_layers", 2))

    x = emb.astype(np.float32)   # (L, D)
    L = len(x)

    for layer in range(n_layers):
        # --- forward direction ---
        W_ih_f = head[f"lstm.weight_ih_l{layer}"]
        W_hh_f = head[f"lstm.weight_hh_l{layer}"]
        # Derive hidden from actual weight shape — robust against wrong metadata.
        hidden = W_hh_f.shape[1]
        b_ih_f = head[f"lstm.bias_ih_l{layer}"]
        b_hh_f = head[f"lstm.bias_hh_l{layer}"]
        W_ih_b = head[f"lstm.weight_ih_l{layer}_reverse"]
        W_hh_b = head[f"lstm.weight_hh_l{layer}_reverse"]
        b_ih_b = head[f"lstm.bias_ih_l{layer}_reverse"]
        b_hh_b = head[f"lstm.bias_hh_l{layer}_reverse"]

        h_f = np.zeros(hidden, dtype=np.float32)
        c_f = np.zeros(hidden, dtype=np.float32)
        fwd = []
        for t in range(L):
            h_f, c_f = _lstm_step(x[t], h_f, c_f, W_ih_f, W_hh_f, b_ih_f, b_hh_f)
            fwd.append(h_f)

        h_b = np.zeros(hidden, dtype=np.float32)
        c_b = np.zeros(hidden, dtype=np.float32)
        bwd = [None] * L
        for t in range(L - 1, -1, -1):
            h_b, c_b = _lstm_step(x[t], h_b, c_b, W_ih_b, W_hh_b, b_ih_b, b_hh_b)
            bwd[t] = h_b

        x = np.concatenate([np.stack(fwd), np.stack(bwd)], axis=1)  # (L, 2*hidden)

    W_out = head["W_out"]   # (1, 2*hidden)
    b_out = head["b_out"]   # (1,)
    logits = x @ W_out.T + b_out
    return 1.0 / (1.0 + np.exp(-logits.ravel()))


def _bilstm_forward_window(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """BiLSTM + sliding average-pool + linear classifier (aggregation head).

    Identical to _bilstm_forward but applies a sliding average window over
    the hidden states before the output layer. The window size is stored in
    head['window_size'].
    """
    import numpy as np
    window = int(head.get("window_size", np.array(7)).item()
                 if hasattr(head.get("window_size", 7), "item")
                 else head.get("window_size", 7))

    # Run BiLSTM layers to get hidden states (L, 2*hidden)
    n_layers = int(head.get("lstm_layers", np.array(2)).item()
                   if hasattr(head.get("lstm_layers", 2), "item")
                   else head.get("lstm_layers", 2))
    hidden   = int(head["lstm_hidden"].item()
                   if hasattr(head["lstm_hidden"], "item")
                   else head["lstm_hidden"])
    x = emb.astype(np.float32)
    L = len(x)

    for layer in range(n_layers):
        W_ih_f = head[f"lstm.weight_ih_l{layer}"]
        W_hh_f = head[f"lstm.weight_hh_l{layer}"]
        b_ih_f = head[f"lstm.bias_ih_l{layer}"]
        b_hh_f = head[f"lstm.bias_hh_l{layer}"]
        W_ih_b = head[f"lstm.weight_ih_l{layer}_reverse"]
        W_hh_b = head[f"lstm.weight_hh_l{layer}_reverse"]
        b_ih_b = head[f"lstm.bias_ih_l{layer}_reverse"]
        b_hh_b = head[f"lstm.bias_hh_l{layer}_reverse"]

        h_f = np.zeros(hidden, np.float32);  c_f = np.zeros(hidden, np.float32)
        fwd = []
        for t in range(L):
            h_f, c_f = _lstm_step(x[t], h_f, c_f, W_ih_f, W_hh_f, b_ih_f, b_hh_f)
            fwd.append(h_f)

        h_b = np.zeros(hidden, np.float32);  c_b = np.zeros(hidden, np.float32)
        bwd = [None] * L
        for t in range(L - 1, -1, -1):
            h_b, c_b = _lstm_step(x[t], h_b, c_b, W_ih_b, W_hh_b, b_ih_b, b_hh_b)
            bwd[t] = h_b
        x = np.concatenate([np.stack(fwd), np.stack(bwd)], axis=1)

    # Sliding average window pool over hidden states
    half  = window // 2
    x_pad = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    x_win = np.stack([x_pad[i:i + window].mean(axis=0) for i in range(L)])

    W_out  = head["W_out"]   # (1, 2*hidden)
    b_out  = head["b_out"]   # (1,)
    logits = x_win @ W_out.T + b_out
    return 1.0 / (1.0 + np.exp(-logits.ravel()))


def _bilstm_crf_emit(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """BiLSTM + linear emission layer for CRF head. Returns (L, N_states) logits."""
    import numpy as np
    n_layers = int(head.get("lstm_layers", np.array(2)).item()
                   if hasattr(head.get("lstm_layers", 2), "item")
                   else head.get("lstm_layers", 2))
    hidden   = int(head["lstm_hidden"].item()
                   if hasattr(head["lstm_hidden"], "item")
                   else head["lstm_hidden"])

    x = emb.astype(np.float32)
    L = len(x)

    for layer in range(n_layers):
        W_ih_f = head[f"lstm.weight_ih_l{layer}"]
        W_hh_f = head[f"lstm.weight_hh_l{layer}"]
        b_ih_f = head[f"lstm.bias_ih_l{layer}"]
        b_hh_f = head[f"lstm.bias_hh_l{layer}"]
        W_ih_b = head[f"lstm.weight_ih_l{layer}_reverse"]
        W_hh_b = head[f"lstm.weight_hh_l{layer}_reverse"]
        b_ih_b = head[f"lstm.bias_ih_l{layer}_reverse"]
        b_hh_b = head[f"lstm.bias_hh_l{layer}_reverse"]

        h_f = np.zeros(hidden, np.float32);  c_f = np.zeros(hidden, np.float32)
        fwd = []
        for t in range(L):
            h_f, c_f = _lstm_step(x[t], h_f, c_f, W_ih_f, W_hh_f, b_ih_f, b_hh_f)
            fwd.append(h_f)

        h_b = np.zeros(hidden, np.float32);  c_b = np.zeros(hidden, np.float32)
        bwd = [None] * L
        for t in range(L - 1, -1, -1):
            h_b, c_b = _lstm_step(x[t], h_b, c_b, W_ih_b, W_hh_b, b_ih_b, b_hh_b)
            bwd[t] = h_b
        x = np.concatenate([np.stack(fwd), np.stack(bwd)], axis=1)

    W_out = head["W_out"]   # (N_states, 2*hidden)
    b_out = head["b_out"]   # (N_states,)
    return x @ W_out.T + b_out   # (L, N_states)


def _viterbi_decode(emissions: "np.ndarray", transitions: "np.ndarray") -> "np.ndarray":
    """Viterbi decoding for linear-chain CRF.

    emissions:   (L, S) — per-position per-state logit scores
    transitions: (S, S) — learned transition matrix T[from, to]
    Returns:     (L,) int8 array of state assignments
    """
    import numpy as np
    L, S     = emissions.shape
    viterbi  = emissions[0].copy()      # (S,)
    backptr  = np.zeros((L, S), dtype=np.int32)

    for t in range(1, L):
        scores = viterbi[:, None] + transitions    # (S, S): [prev, curr]
        best_prev = scores.argmax(axis=0)          # (S,)
        viterbi   = scores[best_prev, np.arange(S)] + emissions[t]
        backptr[t] = best_prev

    # Backtrack
    seq      = np.zeros(L, dtype=np.int8)
    seq[L-1] = int(viterbi.argmax())
    for t in range(L - 2, -1, -1):
        seq[t] = backptr[t + 1, seq[t + 1]]
    return seq


def _conv1d_relu_numpy(x: "np.ndarray", weight: "np.ndarray",
                       bias: "np.ndarray", pad: int) -> "np.ndarray":
    """1-D convolution + ReLU in numpy.  x: (L, D_in), weight: (D_out, D_in, K)."""
    import numpy as np
    L = x.shape[0]
    K = weight.shape[2]
    x_pad = np.pad(x, ((pad, pad), (0, 0)))   # (L+2*pad, D_in)
    out = np.zeros((L, weight.shape[0]), dtype=np.float32)
    for k in range(K):
        out += x_pad[k: k + L] @ weight[:, :, k].T
    return np.maximum(0.0, out + bias)         # ReLU


def _apply_cnn_branch(x: "np.ndarray", head: dict) -> "np.ndarray":
    """Apply the multi-scale CNN branch and return augmented (L, D+cnn_out) array."""
    import numpy as np
    filter_widths = head.get("cnn_filters", [3, 7, 15])
    if hasattr(filter_widths, "tolist"):
        filter_widths = filter_widths.tolist()
    cnn_outs = []
    for idx, kw in enumerate(filter_widths):
        kw  = int(kw)
        W   = head[f"convs.{idx}.weight"]    # (C, D, K)
        b   = head[f"convs.{idx}.bias"]      # (C,)
        cnn_outs.append(_conv1d_relu_numpy(x, W, b, kw // 2))
    cnn_cat = np.concatenate(cnn_outs, axis=1)   # (L, C*n_filters)
    # Batch-norm with running statistics (eval mode)
    bn_w    = head["cnn_bn.weight"]
    bn_b    = head["cnn_bn.bias"]
    bn_mean = head["cnn_bn.running_mean"]
    bn_var  = head["cnn_bn.running_var"]
    cnn_cat = (cnn_cat - bn_mean) / np.sqrt(bn_var + 1e-5) * bn_w + bn_b
    return np.concatenate([x, cnn_cat], axis=1)  # (L, D+cnn_out)


def _bilstm_cnn_forward(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """bilstm2_cnn forward: CNN branch → augmented input → BiLSTM → sigmoid."""
    x_aug = _apply_cnn_branch(emb, head)
    return _bilstm_forward(x_aug, head)


def _bilstm_cnn_forward_ss3(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """bilstm2_ss3_cnn forward: CNN branch → BiLSTM → (L, 3) logits."""
    import numpy as np
    x_aug  = _apply_cnn_branch(emb, head)
    n_layers = int(head.get("lstm_layers", 2))
    hidden   = int(head["lstm_hidden"].item() if hasattr(head["lstm_hidden"], "item")
                   else head["lstm_hidden"])
    x = x_aug.astype(np.float32)
    L = len(x)
    for layer in range(n_layers):
        W_ih_f = head[f"lstm.weight_ih_l{layer}"]
        W_hh_f = head[f"lstm.weight_hh_l{layer}"]
        b_ih_f = head[f"lstm.bias_ih_l{layer}"]
        b_hh_f = head[f"lstm.bias_hh_l{layer}"]
        W_ih_b = head[f"lstm.weight_ih_l{layer}_reverse"]
        W_hh_b = head[f"lstm.weight_hh_l{layer}_reverse"]
        b_ih_b = head[f"lstm.bias_ih_l{layer}_reverse"]
        b_hh_b = head[f"lstm.bias_hh_l{layer}_reverse"]
        h_f = np.zeros(hidden, np.float32); c_f = np.zeros(hidden, np.float32)
        fwd = []
        for t in range(L):
            h_f, c_f = _lstm_step(x[t], h_f, c_f, W_ih_f, W_hh_f, b_ih_f, b_hh_f)
            fwd.append(h_f)
        h_b = np.zeros(hidden, np.float32); c_b = np.zeros(hidden, np.float32)
        bwd = [None] * L
        for t in range(L - 1, -1, -1):
            h_b, c_b = _lstm_step(x[t], h_b, c_b, W_ih_b, W_hh_b, b_ih_b, b_hh_b)
            bwd[t] = h_b
        x = np.concatenate([np.stack(fwd), np.stack(bwd)], axis=1)
    W_out = head["W_out"]   # (3, 2*hidden)
    b_out = head["b_out"]   # (3,)
    return x @ W_out.T + b_out   # (L, 3)


def _bilstm_pos_forward(emb: "np.ndarray", head: dict) -> "np.ndarray":
    """bilstm2_pos forward: add positional embedding → BiLSTM → sigmoid."""
    import numpy as np
    pos_w  = head["pos_emb.weight"]   # (max_len, D)
    max_len = len(pos_w)
    L      = len(emb)
    emb_f  = emb.astype(np.float32)
    if L <= max_len:
        x = emb_f + pos_w[:L]
    else:
        # Sequence longer than training max_len: clamp table and zero-pad remainder.
        x = emb_f.copy()
        x[:max_len] += pos_w
    return _bilstm_forward(x, head)


def bilstm_predict(seq: str, embedder, head: dict | None) -> "list[float] | None":
    """Run any BiLSTM head on seq, return per-residue probabilities or None.

    Returns None when the embedder or head is unavailable, or embedding fails.

    For bilstm2_crf (TM head): returns softmax P(TM_HELIX) per residue for
    display. Viterbi hard labels are available via bilstm_crf_topology().
    """
    if embedder is None or not embedder.is_available() or head is None:
        return None
    import numpy as np
    emb = embedder.embed(seq)
    if emb is None or len(emb) != len(seq):
        return None
    arch = head.get("architecture", "bilstm2")
    if hasattr(arch, "item"):
        arch = arch.item()
    if arch == "bilstm2":
        return _bilstm_forward(emb.astype(np.float32), head).tolist()
    if arch == "bilstm2_crf":
        emits = _bilstm_crf_emit(emb.astype(np.float32), head)
        emits -= emits.max(axis=1, keepdims=True)
        e = np.exp(emits)
        probs = e / e.sum(axis=1, keepdims=True)
        return probs[:, 1].tolist()
    if arch == "bilstm2_window":
        return _bilstm_forward_window(emb.astype(np.float32), head).tolist()
    if arch == "bilstm2_cnn":
        return _bilstm_cnn_forward(emb.astype(np.float32), head).tolist()
    if arch == "bilstm2_pos":
        return _bilstm_pos_forward(emb.astype(np.float32), head).tolist()
    return None


def bilstm_predict_ss3(
    seq: str, embedder, head: "dict | None"
) -> "tuple[list[float], list[float], list[float]] | None":
    """Run the SS3 BiLSTM head.

    Returns (h_probs, e_probs, c_probs) — three parallel lists of per-residue
    softmax probabilities, each value in [0, 1].  Classes: 0=Coil, 1=Helix, 2=Strand.
    """
    if embedder is None or not embedder.is_available() or head is None:
        return None
    import numpy as np
    arch = head.get("architecture", "")
    if hasattr(arch, "item"):
        arch = arch.item()
    emb = embedder.embed(seq)
    if emb is None or len(emb) != len(seq):
        return None
    if arch == "bilstm2_ss3_cnn":
        raw = _bilstm_cnn_forward_ss3(emb.astype(np.float32), head)   # (L, 3) logits
    elif arch in ("bilstm2_ss3", "bilstm2_ss3_crf"):
        raw = _bilstm_crf_emit(emb.astype(np.float32), head)          # (L, 3) logits
    else:
        return None
    raw  -= raw.max(axis=1, keepdims=True)
    exp_l = np.exp(raw)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)   # (L, 3)
    return (
        probs[:, 1].tolist(),  # helix
        probs[:, 2].tolist(),  # strand
        probs[:, 0].tolist(),  # coil
    )


def bilstm_crf_topology(seq: str, embedder, head: dict | None) -> "list[int] | None":
    """Viterbi-decoded topology sequence for bilstm2_crf heads.

    Returns per-residue state indices (0=outside, 1=TM, 2=inside) or None.
    Use this for hard TMD prediction and topology display, not bilstm_predict().
    """
    if embedder is None or not embedder.is_available() or head is None:
        return None
    import numpy as np
    arch = head.get("architecture", "bilstm2")
    if hasattr(arch, "item"):
        arch = arch.item()
    if arch != "bilstm2_crf":
        return None
    emb = embedder.embed(seq)
    if emb is None or len(emb) != len(seq):
        return None
    emits       = _bilstm_crf_emit(emb.astype(np.float32), head)
    transitions = head["transitions"].astype(np.float32)   # (S, S)
    return _viterbi_decode(emits, transitions).tolist()


def bilstm_predict_mc(
    seq: str,
    embedder,
    head: dict | None,
    n_passes: int = 20,
    dropout_p: float | None = None,
) -> "tuple[list[float], list[float]] | None":
    """MC-Dropout inference: run *n_passes* stochastic forward passes.

    Applies dropout to inter-layer activations (matching training behaviour).
    Returns ``(mean_probs, std_probs)`` per residue, or ``None`` if unavailable.

    Parameters
    ----------
    dropout_p:
        Override the dropout probability. Defaults to the value stored in
        ``head["lstm_dropout"]`` if present, else 0.3 (training default).
    """
    if embedder is None or not embedder.is_available() or head is None:
        return None
    import numpy as np
    emb = embedder.embed(seq)
    if emb is None or len(emb) != len(seq):
        return None
    arch = head.get("architecture", "bilstm2")
    if hasattr(arch, "item"):
        arch = arch.item()
    if arch != "bilstm2":
        return None

    x = emb.astype(np.float32)
    p = dropout_p
    if p is None:
        stored = head.get("lstm_dropout", None)
        p = float(stored) if stored is not None else 0.3

    n_layers = int(head.get("lstm_layers", np.array(2)).item()
                   if hasattr(head.get("lstm_layers", 2), "item")
                   else head.get("lstm_layers", 2))
    hidden   = int(head["lstm_hidden"].item()
                   if hasattr(head["lstm_hidden"], "item")
                   else head["lstm_hidden"])
    L = len(x)

    all_probs = np.zeros((n_passes, L), dtype=np.float32)
    for i in range(n_passes):
        xi = x.copy()
        for layer in range(n_layers):
            W_ih_f = head[f"lstm.weight_ih_l{layer}"]
            W_hh_f = head[f"lstm.weight_hh_l{layer}"]
            b_ih_f = head[f"lstm.bias_ih_l{layer}"]
            b_hh_f = head[f"lstm.bias_hh_l{layer}"]
            W_ih_b = head[f"lstm.weight_ih_l{layer}_reverse"]
            W_hh_b = head[f"lstm.weight_hh_l{layer}_reverse"]
            b_ih_b = head[f"lstm.bias_ih_l{layer}_reverse"]
            b_hh_b = head[f"lstm.bias_hh_l{layer}_reverse"]

            h_f = np.zeros(hidden, dtype=np.float32)
            c_f = np.zeros(hidden, dtype=np.float32)
            fwd = []
            for t in range(L):
                h_f, c_f = _lstm_step(xi[t], h_f, c_f, W_ih_f, W_hh_f, b_ih_f, b_hh_f)
                fwd.append(h_f)

            h_b = np.zeros(hidden, dtype=np.float32)
            c_b = np.zeros(hidden, dtype=np.float32)
            bwd = [None] * L
            for t in range(L - 1, -1, -1):
                h_b, c_b = _lstm_step(xi[t], h_b, c_b, W_ih_b, W_hh_b, b_ih_b, b_hh_b)
                bwd[t] = h_b

            xi = np.concatenate([np.stack(fwd), np.stack(bwd)], axis=1)

            # Apply dropout between layers (not after the last layer)
            if layer < n_layers - 1 and 0 < p < 1:
                mask = np.random.binomial(1, 1.0 - p, xi.shape).astype(np.float32)
                xi = xi * mask / (1.0 - p)  # inverted dropout (scale-correct)

        W_out = head["W_out"]
        b_out = head["b_out"]
        logits = xi @ W_out.T + b_out
        all_probs[i] = 1.0 / (1.0 + np.exp(-logits.ravel()))

    mean_p = all_probs.mean(axis=0).tolist()
    std_p  = all_probs.std(axis=0).tolist()
    return mean_p, std_p


def _add_window_context(emb, win: int):
    """Concatenate per-residue embedding with mean of ±win neighbours."""
    import numpy as np
    L, D = emb.shape
    ctx = np.empty((L, D), dtype=emb.dtype)
    for i in range(L):
        lo = max(0, i - win)
        hi = min(L, i + win + 1)
        ctx[i] = emb[lo:hi].mean(axis=0)
    return np.concatenate([emb, ctx], axis=1)  # (L, 2D)


def _mlp2_forward(feat, W1, b1, W2, b2, W3, b3):
    """Two-hidden-layer MLP forward pass with ReLU activations."""
    import numpy as np
    h1 = np.maximum(0.0, feat @ W1.T + b1)
    h2 = np.maximum(0.0, h1 @ W2.T + b2)
    logits = h2 @ W3.T + b3
    return 1.0 / (1.0 + np.exp(-logits.ravel()))


def calc_disorder_profile(
    seq: str,
    window: int = 9,
    embedder=None,
    head: dict | None = None,
) -> list:
    """Per-residue disorder propensity, normalised to 0-1.

    When *embedder* is provided, is available, and *head* contains trained
    weights, ESMC embeddings are used to compute per-residue disorder scores.
    Supports ``architecture="mlp2"`` (2-hidden-layer MLP with window context)
    and legacy linear probe (``coef``/``intercept`` keys).  Falls back to
    metapredict or classical sliding-window otherwise.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size for the classical fallback (default 9).
    embedder:
        Optional :class:`beer.embeddings.base.SequenceEmbedder` instance.
    head:
        Optional dict loaded from ``disorder_head.npz``.
    """
    # Priority 1: ESMC head (most accurate)
    if embedder is not None and embedder.is_available() and head is not None:
        emb = embedder.embed(seq)
        if emb is not None and len(emb) == len(seq):
            import numpy as np
            arch = head.get("architecture", "linear")
            if hasattr(arch, "item"):
                arch = arch.item()

            if arch in ("bilstm2", "bilstm2_cnn"):
                return _bilstm_cnn_forward(emb.astype(np.float32), head).tolist() \
                    if arch == "bilstm2_cnn" \
                    else _bilstm_forward(emb.astype(np.float32), head).tolist()

            if arch == "bilstm2_pos":
                return _bilstm_pos_forward(emb.astype(np.float32), head).tolist()

            if arch == "mlp2":
                W1, b1 = head["W1"], head["b1"]
                W2, b2 = head["W2"], head["b2"]
                W3, b3 = head["W3"], head["b3"]
                mu, std = head["mu"], head["std"]
                win = int(head.get("window", np.array(10)).item()
                          if hasattr(head.get("window", 10), "item")
                          else head.get("window", 10))
                feat = _add_window_context(emb, win)
                feat = (feat - mu) / np.where(std < 1e-8, 1.0, std)
                return _mlp2_forward(feat, W1, b1, W2, b2, W3, b3).tolist()

            # Legacy linear probe
            coef = head.get("coef")
            intercept = head.get("intercept", 0.0)
            if coef is not None and coef.shape[-1] == emb.shape[-1]:
                logits = emb @ coef.T + intercept
                scores = 1.0 / (1.0 + np.exp(-logits.ravel()))
                return scores.tolist()
    # Priority 2: metapredict (DL-based, installed as core dependency)
    try:
        import metapredict as meta
        scores = meta.predict_disorder(seq)
        return [float(s) for s in scores]
    except Exception:
        pass
    # Priority 3: classical sliding-window fallback
    return _classical_disorder_profile(seq, window)


def detect_larks(seq: str, window: int = 7, min_arom: int = 1,
                 min_lc_frac: float = 0.50, max_entropy: float = 1.8) -> list:
    """Detect LARKS (Low-complexity Aromatic-Rich Kinked Segments).

    A LARKS is a short window (6–8 residues) that:
      • Contains >= 1 aromatic residue (F/W/Y)
      • Has >= 50 % low-complexity residues (G/A/S/T/N/Q)
      • Has Shannon entropy < 1.8 bits

    Returns list of dicts: {start, end, seq, n_arom, lc_frac, entropy}
    """
    n = len(seq)
    hits = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        n_arom = sum(1 for aa in w if aa in LARKS_AROMATIC)
        if n_arom < min_arom:
            continue
        lc_frac = sum(1 for aa in w if aa in LARKS_LC) / window
        if lc_frac < min_lc_frac:
            continue
        # Shannon entropy of window
        cnt = Counter(w)
        H = -sum((v / window) * math.log2(v / window) for v in cnt.values())
        if H >= max_entropy:
            continue
        # Merge overlapping hits
        span = (i, i + window - 1)
        overlap = False
        for prev in hits:
            if prev["start"] <= span[1] and span[0] <= prev["end"]:
                overlap = True
                break
        if not overlap:
            hits.append({"start": i, "end": i + window - 1,
                         "seq": w, "n_arom": n_arom,
                         "lc_frac": round(lc_frac, 3),
                         "entropy": round(H, 3)})
    return hits


def predict_coiled_coil(seq: str, window: int = 28) -> list:
    """COILS algorithm — heptad-register log-odds coiled-coil scorer.

    Implements the COILS method (Lupas et al. 1991 Science 252:1162;
    Lupas 1996 Methods Enzymol. 266:513) using the MTIDK 20×7 position-weight
    matrix.  For each window of length *window* (default 28 = 4 heptads) all
    seven heptad phase registers are tried and the maximum log-odds score is
    retained.  The log-odds are converted to P(CC) via a calibrated sigmoid:

        P(CC) = 1 / (1 + exp(−1.25 × (log_odds − 2.5)))

    Calibration: GCN4 leucine zipper log_odds ≈ 4.3 → P ≈ 0.90;
    (LEALELK)₄ synthetic log_odds ≈ 6.1 → P ≈ 0.99;
    random protein log_odds ≈ 0 → P ≈ 0.04.

    Returns list of per-residue P(CC) scores (float, 0–1), length == len(seq).

    References
    ----------
    Lupas, A., Van Dyke, M. & Stock, J. (1991) Science 252:1162-1164.
    Lupas, A. (1996) Methods Enzymol. 266:513-525.
    """
    n = len(seq)
    scores = [0.0] * n
    if n < window:
        return scores

    # Pre-compute background log-score per heptad position (sum over window).
    # Each window covers (window // 7) full heptads + remainder positions.
    # Background = sum of COILS_BG_LOG[pos % 7] for pos in range(window).
    bg_total = sum(COILS_BG_LOG[p % 7] for p in range(window))

    win_pcc: list[float] = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        best_log_score = -1e9
        for phase in range(7):
            s = sum(
                math.log(max(COILS_MTIDK.get(aa, (1.0,)*7)[(phase + j) % 7], 1e-9))
                for j, aa in enumerate(w)
            )
            if s > best_log_score:
                best_log_score = s
        log_odds = best_log_score - bg_total
        pcc = 1.0 / (1.0 + math.exp(-1.25 * (log_odds - 2.5)))
        win_pcc.append(pcc)

    # Distribute window P(CC) to residues (average over all covering windows)
    counts = [0] * n
    for i, pcc in enumerate(win_pcc):
        for r in range(i, i + window):
            scores[r] += pcc
            counts[r] += 1
    for i in range(n):
        if counts[i] > 0:
            scores[i] = scores[i] / counts[i]
    return scores


def scan_linear_motifs(seq: str) -> list:
    """Scan sequence against the built-in LINEAR_MOTIFS library.

    Returns list of dicts: {name, description, start, end, match}
    """
    hits = []
    for name, pattern, description in LINEAR_MOTIFS:
        for m in re.finditer(pattern, seq):
            hits.append({
                "name": name,
                "description": description,
                "start": m.start(),
                "end": m.end() - 1,
                "match": m.group(),
            })
    hits.sort(key=lambda h: h["start"])
    return hits
