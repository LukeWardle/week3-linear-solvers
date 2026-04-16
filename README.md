# Week 3: Linear System Solvers 
Production-quality implementation of linear system solvers for AI/ML 
engineering, 
demonstrating Gaussian elimination, LU decomposition, and numerical stability 
checking. 
Built as part of the AI Engineering Foundations Programme, Week 3. 
## Features 
  - **Multiple solving strategies**: Direct (NumPy), LU decomposition, least-squares 
  - **Automatic method selection**: Smart solver chooses best approach based on matrix shape, κ(A), and optional RidgeCV fallback for extreme ill-conditioning (when κ(A) exceeds threshold and use_ridge=True)
  - **Numerical stability checking**: Condition number computed and thresholds applied (heuristic numerical stability thresholds) 
  - **19 test cases**: Happy path, edge cases, numerical correctness, API contracts, and regularisation behaviour
  - **Performance benchmarked**: LU decomposition provides speedup for repeated solves by amortising factorisation cost (see main.py benchmark) 
  - **Professional code**: Type hints, docstrings, error handling, defensive input processing 
 
  ## Installation 
 
  ```bash 
  # Clone the repository 
  git clone https://github.com/LukeWardle/week3-linear-solvers.git 
  cd week3-linear-solvers 
 
  # Create and activate virtual environment (Windows) 
  python -m venv venv 
  venv\Scripts\activate 
 
  # Install dependencies 
  pip install -r requirements.txt 
  ``` 
 
  ## Quick Start 
 
  ```python 
  import numpy as np 
  from solvers import solve_smart 
 
  # Define and solve a linear system 
  A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float) 
  b = np.array([8, -11, -3], dtype=float) 
 
  x, info = solve_smart(A, b) 
  print(f"Solution: {x}")              # [ 2.  3. -1.] 
  print(f"Method: {info['method']}")    # direct 
  print(f"κ(A): {info['cond']:.2f}")    # 17.88 — well-conditioned 
  print(f"||Ax - b||: {info['residual']:.2e}")  # ~2e-15 (machine zero) 
  ``` 
 
  ## Usage 
 
  ### Automatic (recommended) 
 
  ```python 
  from solvers import solve_smart 
  x, info = solve_smart(A, b)  # Selects direct, LU, or lstsq automatically 
  if info["warning"]: 
      print(info["warning"])   # Prints if κ(A) > 10^6 
  ``` 
 
  ### Direct (square and non-square systems) 
 
  ```python 
  from solvers import solve_direct 
  x = solve_direct(A, b)  # Handles m×n where m = n or m > n 
  ``` 
 
  ### LU (amortised factorisation for repeated right-hand sides) 
 
  ```python 
  from scipy.linalg import lu_factor, lu_solve 
  lu_piv = lu_factor(A)              # Factorise once — O(n³) 
  x1 = lu_solve(lu_piv, b1)          # Solve — O(n²) 
  x2 = lu_solve(lu_piv, b2)          # Solve again — O(n²) 
  # Repeat for many b vectors (amortises factorisation cost)
  ``` 
 
  ### Condition Number Checking 
 
  ```python 
  from solvers import condition_number 
  kappa = condition_number(A) 
  if kappa > 1e8: 
      print(f"WARNING: κ={kappa:.2e}. Apply regularisation.") 
  ``` 
 
  ## Running the Demos 
 
  ```bash 
  python main.py 
  ``` 
 
  Runs four examples: 
  1. Well-conditioned square system (Monday Part 1 example) 
  2. Ill-conditioned matrix with stability warning (Monday Part 3) 
  3. Overdetermined system — least-squares line fitting 
  4. LU vs direct performance benchmark (repeated solve scenario demonstrating amortised LU factorisation cost)
 
  ## Testing 
 
  ```bash 
  # Run all tests (19 total)
  pytest test_solvers.py -v 
 
  # Run one test class 
  pytest test_solvers.py::TestSolveLU -v 
 
  # Run with short tracebacks on failure 
  pytest test_solvers.py -v --tb=short 
  ``` 
 
  ## Project Structure 
 
  ``` 
  week3-linear-solvers/ 
  ├── solvers.py          # direct, LU, condition number analysis, smart routing, RidgeCV fallback
  ├── test_solvers.py     # 19 tests covering solver correctness, LU consistency, numerical stability, and smart routing behaviour 
  ├── main.py             # 4 demonstrations including LU performance benchmark 
  ├── requirements.txt    # Pinned: numpy, scipy, scikit-learn, pytest 
  ├── README.md           # This file 
  └── .gitignore          # Excludes venv/, __pycache__/, IDE files 
  ``` 
 
  ## Mathematical Background 
 
  | Concept | Implementation | Reference | 
  |---------|---------------|-----------| 
  | Gaussian elimination | np.linalg.solve() | Week 3, Monday Part 1 | 
  | LU decomposition | scipy.linalg.lu_factor / scipy.linalg.lu_solve | Week 3, Monday Part 2 |
  | Condition number | np.linalg.cond() | Week 3, Monday Part 3 | 
  | Least-squares | np.linalg.lstsq() | Week 3, Monday Part 1 | 
 
  ## Skills Demonstrated 
 
  **Linear Algebra**: Gaussian elimination, LU decomposition, condition numbers, nullity 
  **Numerical Methods**: Floating-point stability, partial pivoting, error analysis 
  **Python**: NumPy, SciPy, pytest, type hints, docstrings, defensive programming 
  **Engineering**: Test-driven development, Git workflow, professional documentation 
 
  ## License 
 
  MIT — Educational Project 
