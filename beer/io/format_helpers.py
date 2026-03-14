"""Sequence formatting utilities for BEER."""

def format_sequence_block(seq: str, name: str = "", width: int = 60, group: int = 10) -> str:
    """Format sequence in UniProt-style with ruler."""
    lines = []
    if name:
        lines.append(f">{name}")
    n = len(seq)
    for i in range(0, n, width):
        chunk = seq[i:i+width]
        grouped = " ".join(chunk[j:j+group] for j in range(0, len(chunk), group))
        ruler_start = i + 1
        lines.append(f"{ruler_start:>6}  {grouped}")
    return "\n".join(lines)
