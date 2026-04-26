import numpy as np
from typing import Optional, Callable, List, Tuple

# -------------------- rate vector extraction helpers --------------------

def build_chain_rate_positions(num_states: int) -> List[Tuple[int, int]]:
    """
    Adjacent chain rates: (0<->1), (1<->2), ..., (S-2<->S-1)

    Returns positions in the order:
      [(0,1),(1,0), (1,2),(2,1), ..., (S-2,S-1),(S-1,S-2)]
    """
    pos: List[Tuple[int, int]] = []
    for i in range(num_states - 1):
        pos.append((i, i + 1))   # forward
        pos.append((i + 1, i))   # backward
    return pos


def extract_chain_rates_from_Q(Q: np.ndarray) -> np.ndarray:
    """
    Extract the chain-adjacent rates from a square Q matrix.

    Supports:
      - Q shape (S, S)  -> returns (D,)
      - Q shape (N, S, S) -> returns (N, D)

    where D = 2*(S-1).
    """
    Q = np.asarray(Q)
    assert Q.ndim in (2, 3), f"Q must be (S,S) or (N,S,S), got {Q.shape}"
    S = Q.shape[-1]
    assert Q.shape[-2] == S, f"Q must be square, got {Q.shape}"

    pos = build_chain_rate_positions(S)

    if Q.ndim == 2:
        return np.asarray([Q[i, j] for (i, j) in pos], dtype=Q.dtype)
    else:
        # batched
        return np.stack([Q[:, i, j] for (i, j) in pos], axis=1).astype(Q.dtype)


# -------------------- canonicalization helpers (1D y) --------------------

def lex_less(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a is lexicographically smaller than b."""
    for i in range(len(a)):
        if a[i] < b[i]:
            return True
        if a[i] > b[i]:
            return False
    return False


def canonicalize_by_permutations(y: np.ndarray, perms: List[List[int]]) -> np.ndarray:
    """
    y is 1D (D,). Return lexicographically smallest among y[p] for p in perms.
    """
    y = np.asarray(y)
    assert y.ndim == 1, f"expected y to be 1D, got {y.shape}"
    best = y.copy()

    for p in perms:
        yp = y[np.asarray(p, dtype=np.int64)]
        if lex_less(yp, best):
            best = yp
    return best


def perms_chain_reversal(D: int) -> List[List[int]]:
    """
    For your chain ordering [(0,1),(1,0),(1,2),(2,1),...],
    the symmetry under reversal is exactly full vector reversal.

    Example D=8 -> [0..7] and [7..0]
    """
    return [list(range(D)), list(range(D - 1, -1, -1))]


# --- Known topologies ---

def cococ(y: np.ndarray) -> np.ndarray:
    """
    COCOC is palindromic => symmetric under reversal.
    Works for any chain length because it uses D=len(y).
    """
    D = int(np.asarray(y).shape[0])
    return canonicalize_by_permutations(y, perms_chain_reversal(D))


def coco(y: np.ndarray) -> np.ndarray:
    """
    COCO is NOT symmetric under reversal.
    Keep as identity (but in practice we return None canonicalizer for COCO in auto mode).
    """
    return np.asarray(y)


# Map: topology name -> (is_symmetric, canonicalizer)
TOPOLOGY_CANON: dict[str, tuple[bool, Optional[Callable[[np.ndarray], np.ndarray]]]] = {
    "COCOC": (True,  cococ),
    "COCO":  (False, None),  # <--- important: not symmetric, so canonicalizer is None
}


def choose_canonicalizer(topology: str, symmetry: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    symmetry:
      - "off": always None
      - "on":  use canonicalizer if known; if unknown, None
      - "auto": use canonicalizer only if topology is known symmetric
    """
    topology = str(topology).upper()
    is_known = topology in TOPOLOGY_CANON
    is_symm  = TOPOLOGY_CANON[topology][0] if is_known else False
    canon    = TOPOLOGY_CANON[topology][1] if is_known else None

    if symmetry == "off":
        return None
    if symmetry == "on":
        return canon
    # symmetry == "auto"
    return canon if is_symm else None
