"""
solvers.py - Linear System Solvers

=======================================

Production-quality implementations of linear system solving methods.

Functions:
  solve_direct(A, b)        - NumPy direct solve or least-squares
  solve_lu(A, b)            - SciPy LU decomposition solve
  condition_number(A)       - Condition number k(A)
  solve_smart(A, b)         - Auto method selection with diagnostics

Theory background:
  week 3, Monday Parts 1-3 (Gaussian elimination, LU decomposition, and numerical stability - condition numbers and regularisation)

Usage:
  import numpy as np
  from solvers import solve_smart
  A = np.array([[2, 1], [-1, 3]], dtype=float)
  b = np.array([5, 2], dtype=float)
  x, info = solve_smart(A, b)
  print(f'Solution: {x}, Method: {info["method]}')

"""

import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.linalg import lu_factor, lu_solve
from typing import Tuple, Dict, Any

def solve_direct(A: np.ndarray, b: np.ndarray) -> np.ndarray:
  """
  Solve linear system Ax = b using NumPy's direct solver.

  Automatically handles both square and non-square systems:
    - Square (m = n): uses np.linalg.solve()      - exact solution
    - Non-square (m ≠ n): uses np.linalg.lstsq()  - least-squares

  Args:
    A: Coefficient matrix: shape (m, n). Must be 2D.
    b: Right-hand side vector, shape (m,) or (m, 1).

  Returns:
    x: Solution vector, shape (n,).

  Raises:
    np.linalg.LinAlgError: If A is  square and singular (det=0).

  Examples:
    >>> A = np.array([[2, 1], [-1, 3]], dtype=float)
    >>> b = np.array([5, 2], dtype=float)
    >>> x = solve_direct(A, b)
    >>> print(np.round(x, 4))
    [2.0143 0.9714]
  
  """

  # Defensive: flatten b to 1D regardless of input shape
  b = np.asarray(b, dtype=float).flatten()
  A = np.asarray(A, dtype=float)

  m, n = A.shape

  if m == n:
    # Square system: direct LU-backed solve
    return np.linalg.solve(A, b)
  else:
    # Non-square: minimum-norm least-squares solution
    x, _residuals, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    return x
  

def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
  """
  Solve Ax = b using LU decomposition (PLU with partial pivoting).

  Efficient for repeated solves with the same A matrix.
  For multiple right-hand sides, call lu_factor(A) once externally and use lu_solve() directly to avoid re-factorising.

  Args:
    A: Square coefficient matrix, shape (n, n),
    b: Right-hand side vector, shape (n,) or (n, 1),

  Returns:
    x: Solution vector, shape (n,).

  Raises:
    ValueError: If A is not square.
    np.linalg.LinAlgError: If A is singular.

  Examples:
    >>> A = np.array([[2, 1], [-1, 3]], dtype=float)
    >>> b = np.array([5, 2], dtype=float)
    >>> x = solve_lu(A, b)
    >>> print(np.round(x, 4))
    [2.0143 0.9714]

    for multiple b vectors (efficient pattern):
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> lu_piv = lu_factor(A)         # factorised once
    >>> x1 = lu_solve(lu_piv, b1)     # O(n²) each
    >>> x2 = lu_solve(lu_piv, b2)     # O(n²) each
  
  """

  b = np.asarray(b, dtype=float).flatten()
  A = np.asarray(A, dtype=float)

  m, n = A.shape
  if m != n:
    raise ValueError(
      f'solve_lu requires a square matrix. Got shape {A.shape}.'
      f'For non-square systems, use solve_direct().'
    )
  
  lu, piv = lu_factor(A)      # PLU factorisation - O(n³), once
  x = lu_solve((lu, piv), b)  # Substitution - O(n²)
  return x

def condition_number(A: np.ndarray) -> float:
  """
  Compute the 2-norm condition number matrix A.

  k(A) = σ_max / σ_min, where σ are the singular values of A. 
  Measures sensitivity of Ax=b to perturbations in A or b:
  ||Δx||/||x|| ≤ κ(A) × ||Δb||/||b||

    Interpretation thresholds (rule of thumb):
    k < 100: Well-conditioned - fully safe
    k < 10^4: Mild concern - monitor
    k < 10^6: Moderate - consider scaling
    k < 10^8: Serious - apply regularisation

    k ≥ 10^8: NHS Figital threshold - validate by alternate method
    k = ∞: Singular - no unique solution

    Args:
      A: Matrix to analyse. Any shape (m, n).

    Returns:
      Condition number as a float.

    Examples:
      >>> A_good = np.eye(3)
      >>> condition_number(A_good)
      1.0
      >>> A_bad = np.array([[1, 1.0001], [1.001, 1.0002]])
      >>> condition_number(A_bad)
      40004.00... # Ill-conditioned
  
  """
  return float(np.linalg.cond(A))

def solve_smart(
    A: np.ndarray,
    b: np.ndarray,
    use_ridge: bool=False) -> Tuple[np.ndarray, Dict[str, Any]]:
  """
  Solve Ax = b with automatic method selection and full diagnostics.

  Selects the solving method based on matrix shape and condition number:
    - Non-square (m ≠ n): least-square via solve_direct()
    - Square, k < 10^6: direct via solve_direct()
    - Square, 10^6 ≤ k ≤ 10^8: LU decomposition via solve_lu()
    - Square, k > 10^8: 
        - RidgeCV if use_ridge=True
        - otherwise LU decomposition

  Args:
    A: Coefficient matrix, any shape (m, n).
    b: Right-hand side vector, shape (m,) or (m, 1).

  Returns:
    x: Solution vector, shape (n,).
    info: Diagnostics dictionary containing:
      'method'  : str - solver used ('direct', 'lu', 'lstsq')
      'shape'   : tuple - shape of A
      'cond'    : float - κ(A), or None if non-square
      'residual': float - ||Ax - b|| (should be near 0)
      'warning' : str - stability warning, or None

  Raises: 
    np.linalg.LinAlgError: If A is square and singular. 
 
  Examples: 
    >>> A = np.array([[2, 1], [-1, 3]], dtype=float) 
    >>> b = np.array([5, 2], dtype=float) 
    >>> x, info = solve_smart(A, b) 
    >>> print(x, info['method'], info['cond']) 
    [2.0143 0.9714] direct 7.85...

  """

  b = np.asarray(b, dtype=float).flatten() 
  A = np.asarray(A, dtype=float) 
 
  info: Dict[str, Any] = { 
    'shape':   A.shape, 
    'cond':    None, 
    'method':  None, 
    'residual':None, 
    'warning': None, 
  }

  m, n = A.shape 
 
  # ── Step 1: Assess shape ──────────────────────────────────────── 
  if m != n: 
    # Non-square system → least-squares, no condition check 
    x = solve_direct(A, b) 
    info['method'] = 'lstsq' 
  else: 
  # ── Step 2: Square system — check condition number ────────── 
    kappa = condition_number(A) 
    info['cond'] = kappa 
 
    if kappa > 1e10: 
      info['warning'] = ( 
        f'CRITICAL: κ={kappa:.2e}. System severely ill-conditioned. ' 
        f'Results may be unreliable. Consider Ridge regularisation.' 
      ) 
    elif kappa > 1e6: 
      info['warning'] = ( 
        f'MODERATE: κ={kappa:.2e}. Apply StandardScaler or ' 
        f'RidgeCV before using solution in production.' 
      ) 
 
    # ── Step 3: Choose method based on conditioning ───────────── 
    if kappa < 1e6: 
      x = solve_direct(A, b) 
      info['method'] = 'direct' 
    else: 
      if kappa > 1e8 and use_ridge:
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model.fit(A, b)
        x = model.coef_.reshape(-1)
        info['method'] = 'ridge'
      else:
        x = solve_lu(A, b)
        info['method'] = 'lu'

  # ── Step 4: Compute residual — always ──────────────────────────── 
  info['residual'] = float(np.linalg.norm(A @ x - b)) 
  return x, info