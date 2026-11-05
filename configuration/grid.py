"""Deterministic near-square MPI process-grid selection."""
import math

def resolve_grid(ranks, rows=None, cols=None):
    if ranks <= 0: raise ValueError('ranks must be positive')
    if rows is not None and rows <= 0: raise ValueError('grid rows must be positive')
    if cols is not None and cols <= 0: raise ValueError('grid cols must be positive')
    if rows is not None and cols is not None:
        if rows * cols != ranks: raise ValueError('grid rows * grid cols must equal ranks')
        return rows, cols
    if rows is not None:
        if ranks % rows: raise ValueError('grid rows must divide ranks')
        return rows, ranks // rows
    if cols is not None:
        if ranks % cols: raise ValueError('grid cols must divide ranks')
        return ranks // cols, cols
    rows = math.isqrt(ranks)
    while ranks % rows: rows -= 1
    return rows, ranks // rows
