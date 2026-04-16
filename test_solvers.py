"""
test_solvers.py - Test suite for linear system solvers

========================================================================

Test for: solve_direct(), solve_lu(), condition_number(), solve_smart()

Run with: pytest test_solvers.py -v

"""

import numpy as np
import pytest
from solvers import(
  solve_direct,
  solve_lu,
  condition_number,
  solve_smart,
)

class TestSolveDirect:
  """
  Tests for solve_direct() covering square, non-square, and edge cases.

  """

  def test_square_well_conditioned(self):
    """
    Happy path: well-conditioned 2x2 system with known solution.

    System: 2x + y = 5, x + 2y = 4 -> solution: x = 2, y = 1
    
    """

    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([5, 4], dtype=float)
    x = solve_direct(A, b)
    np.testing.assert_allclose(x, [2, 1], rtol=1e-10)

    # Also verify residual directly
    assert np.linalg.norm(A @ x - b) < 1e-10

  def test_monday_example(self):
    """
    Verify Monday Part 1 worked example: solution should be [2, 3, -1].
    
    """

    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    x = solve_direct(A, b)
    np.testing.assert_allclose(x, [2, 3, -1], rtol=1e-10)

  def test_identity_matrix(self):
    """
    Trivial case: identity matrix should return b unchanged.
    
    """
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0], dtype=float)
    x = solve_direct(A, b)
    np.testing.assert_allclose(x, b)

  def test_overdetermined_system(self):
    """
    Non-square (m > n): should return least-squares solution.
    
    Fitting y = mx + c to 4 data points: (1, 2), (2, 3), (3, 4), (4, 5)

    """
    A = np.array([[1, 1], [2, 1], [3, 1], [4, 1]], dtype=float)
    b = np.array([2, 3, 4, 5], dtype=float)
    x = solve_direct(A, b)
    assert x.shape == (2,)                   # Two unknowns: slope and intercept
    residual = np.linalg.norm(A @ x - b)
    assert residual < 1.0                     # Not zero (overdetermined), but small

    # The true line is y = x + 1, so slope ≈ 1, intercept  ≈ 1
    np.testing.assert_allclose(x, [1.0, 1.0], rtol=1e-8)

  def test_column_vector_b(self):
    """
    Edge case: column vector b with shape (n, 1) instead of (n,).
    
    """
    A = np.array([[2, 1], [1, 2]], dtype=float)
    b = np.array([[5], [4]], dtype=float)         # Shape (2, 1), not (2, )
    x = solve_direct(A, b)
    # flatten() in solve_direct() should handle this transparently
    np.testing.assert_allclose(x, [2, 1], rtol=1e-10)

  def test_singular_matrix_raises(self):
    """
    Edge case: singular matrix should raise LinAlgError.
    
    Row 2 = 2 * row 1 -> singular (rank 1 not rank 2)

    """
    A = np.array([[1, 2], [2, 4]], dtype=float)
    b = np.array([3, 6], dtype=float)
    with pytest.raises(np.linalg.LinAlgError):
      solve_direct(A, b)

class TestSolveLU:
  """
  Tests for solve_lu() covering correctness and error handling.
  
  """
  def test_matches_direct_solver(self):
    """
    Verify LU solver matches direct solver on a 3x3 system.
    
    """
    A = np.array([[3, 1, -1], [1, 3, 1], [-1, 1, 3]], dtype=float)
    b = np.array([1.0, 5.0, 3.0], dtype=float)
    x_direct = solve_direct(A, b)
    x_lu = solve_lu(A, b)
    np.testing.assert_allclose(x_lu, x_direct, rtol=1e-10)

  def test_monday_lu_2x2_example(self):
    """
    LU and direct methods must produce identical results.
    
    """
    A = np.array([[2, 3], [4, 7]], dtype=float) 
    b = np.array([5.0, 13.0], dtype=float) 
    x = solve_lu(A, b) 
    np.testing.assert_allclose(x, [-2, 3], rtol=1e-10) 

  def test_monday_lu_3x3_example(self):
    """
    Verify Monday Part 2 3x3 LU example:
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    Expected solution: [2, 3, -1]
    
    """
    A = np.array([[2, 1, -1],
                  [-3, -1, 2], 
                  [-2, 1, 2]], dtype=float) 
    b = np.array([8, -11, -3], dtype=float) 
    x = solve_lu(A, b) 
    np.testing.assert_allclose(x, [2, 3, -1], rtol=1e-10) 
  
  def test_non_square_raises_value_error(self):
    """
    LU requires square matrix - non-square should raise ValueError.
    
    """
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)   # 3×2, not square
    b = np.array([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ValueError, match="square matrix"):
      solve_lu(A, b)
    # match="square matrix" confirms the error message contains that phrase

class TestConditionNumber:
  """
  Tests for condition_number() covering well and ill-conditioned matrices.
  
  """
  def test_identity_has_condition_one(self):
    """
    Identity matrix: k = 1.0 (perfect conditioning).
    
    """
    for n in [2, 3, 5, 10]:   # Test multiple sizes
      kappa = condition_number(np.eye(n))
      assert abs(kappa - 1.0) < 1e-10, f"Identity {n} x {n}: expected k = 1, got {kappa}"

  def test_diagonal_matrix(self):
    """
    Diagonal matrix: k = max_diag / min_diag.
    
    """
    A = np.diag([2.0, 6.0])     # k = 6/2 = 3
    kappa = condition_number(A)
    np.testing.assert_allclose(kappa, 3.0, rtol=1e-10)

  def test_hilbert_matrix_ill_conditioned(self):
    """
    Hilbert matrix: notoriously ill-conditioned - k should be >> 1.
    
    """
    n = 5
    # H[i, j] = 1 / (i + j + 1) - entries become small and similar
    H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)]) 
    kappa = condition_number(H)
    
    # Hilbert(5) has k ≈ 4.8 × 10⁵ — well above our warning threshold 
    assert kappa > 1e4, f"Hilbert(5) should be ill-conditioned, got k = {kappa:.2e}"
    print(f"Hilbert(5) condition number: {kappa:.2e}")  # Educational output

class TestSolveSmart: 
  """
  Tests for solve_smart() covering method selection, diagnostics, and warnings.
  
  """ 
 
  def test_returns_tuple_of_solution_and_info(self): 
    """
    API contract: must return (x, info) tuple with correct types.
    
    """

    A = np.array([[2, 1], [1, 2]], dtype=float) 
    b = np.array([3.0, 3.0], dtype=float) 
    result = solve_smart(A, b) 
    assert isinstance(result, tuple), "solve_smart() must return a tuple" 
    assert len(result) == 2, "Tuple must have exactly 2 elements" 
    x, info = result 
    assert isinstance(x, np.ndarray), "First element must be np.ndarray" 
    assert isinstance(info, dict), "Second element must be dict" 
 
  def test_info_dict_has_required_keys(self): 
    """
    API contract: info dict must contain all documented keys.
  
    """ 
    A = np.array([[2, 1], [1, 2]], dtype=float) 
    b = np.array([3.0, 3.0], dtype=float) 
    _, info = solve_smart(A, b) 
    required_keys = {"method", "shape", "cond", "residual", "warning"} 
    assert required_keys.issubset(info.keys()), f"Missing keys: {required_keys - info.keys()}" 
 
  def test_selects_lstsq_for_non_square(self): 
    """
    Auto-selection: overdetermined system must use least-squares.
  
    """ 
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)   # 3×2 
    b = np.array([2.0, 3.0, 4.0], dtype=float) 
    _, info = solve_smart(A, b) 
    assert info["method"] == "lstsq", f"Expected lstsq for non-square, got {info['method']}" 
 
  def test_issues_warning_for_ill_conditioned(self): 
    """
    Stability check: ill-conditioned matrix must produce a warning.

    Nearly identical rows → very high condition number 
  
    """ 
    A = np.array([[1.0, 1.0001], [1.0001, 1.0002]], dtype=float) 
    b = np.array([2.0001, 2.0003], dtype=float) 
    _, info = solve_smart(A, b) 
    assert info["warning"] is not None, "Ill-conditioned matrix should produce a warning" 

    # Warning message should mention condition or κ 
    warning_lower = info["warning"].lower() 
    assert any(kw in warning_lower for kw in ["cond", "ill", "κ", "kappa"]), f"Warning should mention conditioning, got: {info['warning']}"

  # Ridge feature tests
  def test_uses_ridge_when_enabled_for_ill_conditioned(self):
    """
    Ridge should be selected when matrix is ill-conditioned and use_ridge=True
    
    """
    A = np.array([[1.0, 1.0000001],
                  [1.0000001, 1.0000002]
                  ], dtype=float)  # This pushes k very high
    b = np.array([2.0001, 2.0003], dtype=float)
    x, info = solve_smart(A, b, use_ridge=True)
    assert info["method"] == "ridge", f"Expected ridge, got {info['method']}" 
    assert x.shape == (2,)
    assert np.all(np.isfinite(x))

  def test_falls_back_to_lu_when_ridge_disabled(self):
    """
    Ill-conditioned system should fall back to LU when use-ridge=False.
    
    """
    A = np.array([[1.0, 1.0000001],
                  [1.0000001, 1.0000002]
                  ], dtype=float)
    b = np.array([2.0001, 2.0003], dtype=float)
    x, info = solve_smart(A, b, use_ridge=False)
    assert info["method"] == "lu", f"Expected lu, got {info['method']}"
    assert x.shape == (2,)
    assert np.all(np.isfinite(x))