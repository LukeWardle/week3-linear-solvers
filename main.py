"""
main.py - Demonstration of linear system solvers

================================================================

Four demonstrations:
  1. Well-conditioned square system
  2. Ill-conditioned matrix with stability warning
  3. Overdetermined system (least-square line fitting)
  4. LU vs direct performance benchmark

"""

import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve
from solvers import solve_direct, solve_lu, solve_smart, condition_number

def demo_well_conditioned():
  """
  Example 1: standard well-conditioned system from Monday Part 1.
  
  """
  print("=" * 62)
  print("EXAMPLE 1: Well-Conditioned Square System")
  print("=" * 62)

  # The 3x3 system from Monday Part 1
  A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
  b = np.array([8, -11, -3], dtype=float)

  print(f"System Ax = b:") 
  print(f"A =\n{A}") 
  print(f"b = {b}") 
  print(f"\nCondition number κ(A): {condition_number(A):.2f}") 
  print("  → Well-conditioned (κ < 100). Direct solver appropriate.\n")

  x, info = solve_smart(A, b)
  print(f"Solution: x = {x}")
  print(f"Method:   {info['method']}")
  print(f"Residual  ǁAx−bǁ: {info['residual']:.2e}  (should be ≈ machine zero)")
  print()

def demo_ill_conditioned():
  """
  Example 2: system with nearly identical rows - Monday Part 3 case.
  
  """
  print("=" * 62)
  print("EXAMPLE 2: Ill-Conditioned Matrix")#
  print("=" * 62)

  A = np.array([[1.0, 1.0001], [1.0001, 1.0002]], dtype=float)
  b = np.array([2.0001, 2.0003], dtype=float)

  print(f"System with nearly identical rows:")
  print(f"A =\n{A}")
  print(f"\nCondition number κ(A): {condition_number(A):.2e}")
  print("  → Ill-conditioned! Tiny perturbation in b → large change in x.\n")
  
  x, info = solve_smart(A, b)
  if info["warning"]:
    print(f"WARNING: {info['warning']}")
    print()

  print(f"Solution: x = {np.round(x, 4)}")
  print(f"Residual: {info['residual']:.2e}")
  print("See Monday Part 3 for remediation: StandardScaler + RidgeCV")
  print()

def demo_overdetermined():
  """
  Example 3: fitting a line y = mx + c to noisy data - lstsq path.
  
  """
  print("=" * 62)
  print("EXAMPLE 3: Overdetermined System (Line Fitting)")
  print("=" * 62) 

  # 5 data points on y ≈ 2x + 1 with small noise
  np.random.seed(42)
  x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(5)

  # Design matrix [x, 1] — fitting y = m*x + c
  A = np.column_stack([x_data, np.ones(5)])   # shape (5, 2)
  b = y_data

  print(f"Fitting y = mx + c to {len(x_data)} data points")
  print(f"True line: y = 2x + 1  (noise added)")
  print(f"System shape: {A.shape} (overdetermined)")
  print()

  x, info = solve_smart(A, b)
  m, c = x
  print(f"Best-fit line: y = {m:.4f}x + {c:.4f}")
  print(f"True coefficients: m = 2.0, c = 1.0")
  print(f"Method: {info['method']}  Residual: {info['residual']:.4f}")
  print()

def demo_lu_performance():
  """
  Example 4: benchmark LU vs repeated direct — confirms Monday Part 2 theory.
  
  """
  print("=" * 62)
  print("EXAMPLE 4: LU vs Direct — Performance Benchmark")
  print("=" * 62)

  n = 200       # Matrix size
  k = 50        # Number of different b vectors (same A)

  np.random.seed(0)

  # Build a well-conditioned positive-definite matrix
  X_rand = np.random.randn(n, n)
  A_big = X_rand.T @ X_rand + np.eye(n)
  b_vectors = [np.random.randn(n) for _ in range(k)]

  print(f"Matrix: {n}×{n},  right-hand sides: {k}")
  print(f"κ(A) = {condition_number(A_big):.2f}  (well-conditioned)\n")

  # Method 1: repeated solve_direct — O(n³) every time
  t0 = time.perf_counter()
  for b_vec in b_vectors:
    _ = solve_direct(A_big, b_vec)
  t_direct = time.perf_counter() - t0 

  # Method 2: lu_factor once, lu_solve k times 
  t0 = time.perf_counter()
  lu_piv = lu_factor(A_big)          # O(n³) — paid once
  for b_vec in b_vectors:
    _ = lu_solve(lu_piv, b_vec)    # O(n²) — paid k times
  t_lu = time.perf_counter() - t0

  speedup = t_direct / t_lu if t_lu > 0 else float("inf")
  print(f"  Repeated solve_direct ({k} calls):   {t_direct*1000:.1f} ms")
  print(f"  lu_factor + lu_solve  ({k} calls):   {t_lu*1000:.1f} ms")
  print(f"  Speedup:                             {speedup:.1f}×")
  print()
  print(f"  Theory: LU saves ({k}-1)×O(n³) − {k}×O(n²) operations")
  print(f"  For n={n}: direct={k}×{n}³/3 ≈ {k*(n**3)//3_000_000}M ops, " f"LU≈{(n**3)//3_000_000 + k*(n**2)//1_000_000}M ops total")

if __name__ == "__main__":
  demo_well_conditioned()
  demo_ill_conditioned()
  demo_overdetermined()
  demo_lu_performance()

  print("=" * 62)
  print("ALL DEMOS COMPLETE")
  print("=" * 62)
  print()
  print("Next steps:")
  print("  pytest test_solvers.py -v     (run all 16 tests)")
  print("  python -c \"from solvers import solve_smart; help(solve_smart)\"")
  print("  git log --oneline             (review commit history)") 