import numpy as np
from typing import Optional, Callable, List

def lex_less(a: np.ndarray, b: np.ndarray) -> bool:
    for i in range(len(a)):
        if a[i] < b[i]: return True
        if a[i] > b[i]: return False
    return False

def canonicalize_by_permutations(y: np.ndarray, perms: List[List[int]]) -> np.ndarray:
    best = y
    for p in perms:
        yp = y[p]
        if lex_less(yp, best):
            best = yp
    return best

# --- Known symmetric topologies ---

# COCOC linear chain, 8 rates laid out as:
# [k12,k21,k23,k32,k34,k43,k45,k54]
# Symmetry: forward/backward reversal only.
PERMS_COCOC_LINEAR_8 = [
    list(range(8)),            # identity
    list(range(7, -1, -1)),    # full reversal
]

def cococ(y: np.ndarray) -> np.ndarray:
    return canonicalize_by_permutations(y, PERMS_COCOC_LINEAR_8)

# Map: topology name -> (is_symmetric, canonicalizer)
TOPOLOGY_CANON: dict[str, tuple[bool, Optional[Callable[[np.ndarray], np.ndarray]]]] = {
    "COCOC": (True,  cococ),
    # add more:  "ring_6": (True, canon_ring_6),  "asymmetric_8": (False, None), ...
}

def choose_canonicalizer(topology: str, symmetry: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    is_known = topology in TOPOLOGY_CANON
    is_symm  = TOPOLOGY_CANON[topology][0] if is_known else False
    canon    = TOPOLOGY_CANON[topology][1] if is_known else None

    if symmetry == "off":
        return None
    if symmetry == "on":
        # use known canonicalizer if we have one; otherwise leave None
        return canon
    # symmetry == "auto"
    return canon if is_symm else None
