[![Checks](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/gh/aladshaw3/pjfnk_matlab/branch/main/graph/badge.svg)](https://codecov.io/gh/aladshaw3/pjfnk_matlab) 

# PJFNK MATLAB
This repository is for an implementation of the Preconditioned Jacobian-Free Newton-Krylov method in Matlab,
as well as methods for standard full Newton method and formulation of full and sparse Numerical
Jacobian matrices. 

# Requirements

- MATLAB
- [Optional] Optimization Toolbox

**NOTE**: The Optimization Toolbox in MATLAB is not required to run these 
codes, but is used for validation checks in some of the unit tests. 

# Functions

- `JacobianOperator`

Performs an action of a Jacobian on any vector v without forming the Jacobian
explicitly. 

```
% Pseudo Code
Jv = (F(x+epsilon*v) - F(x))/epsilon;
```

where F(x) evaluates the non-linear system at state x.

where v is the vector being multiplied by the Jacobian.

where epsilon is a small perturbation value. 

---

- `NumericalJacobianMatrix`

Approximates the full Jacobian with finite-differences approach. Can return
either a 'dense' or 'sparse' Jacobian via user request. Default is to return 
a 'dense' Jacobian. 

```
% Pseudo Code
for i=1:N
    dx = x;
    dx(i,1) =  x(i,1) + epsilon;
    J(:, i) = (F(dx) - F(x))/epsilon;
end
```

where F(x) evaluates the non-linear system at state x.

where epsilon is a small perturbation value. 

---

- `NewtonMethod`

Performs a basic Newton's method with direct inversion of the Jacobian matrix.
The user may provide an exact Jacobian function as a function handle. If they 
do not, then the method will use the `NumericalJacobianMatrix` function to 
approximate the Jacobian. 

```
% Pseudo Code
while (norm(F(x)) > ftol)
    % Perform a Newton Step
    %   J*s = -F --> s = -J\F
    x = x0 - J(x)\F(x);

    % Update x
    x0=x;
end
```

where F(x) evaluates the non-linear system at state x.

where J(x) evaluates the Jacobian matrix at state x.

where ftol is the error or residual tolerance for convergence. 

---

- `JacobianFreeNewtonKrylov`

Performs a Jacobian-Free Newton-Krylov solve with associated user options.
The user may choose any valid Krylov method to solve the Newton step.  

```
% Pseudo Code
while (norm(F(x)) > ftol)
    % Perform a Newton Step
    %   J*s = -F --> s = -J\F
    x = x0 + KrylovSolve( @JacOp( @F(x0),x0,F), -F );

    % Update x
    x0=x;
end
```

where F(x) evaluates the non-linear system at state x.

where JacOp evaluates the Jacobian operator at state x.

where KrylovSolve evaluates the solution to the linear system.

where ftol is the error or residual tolerance for convergence. 

---

# Linesearch Methods

- `StandardNewtonStep`

Takes a standard Newton step without any linesearching.

```
% Invoked via the 'options' struct() for Newton methods
options.linesearch = 'none';
```

---

- `BacktrackingLinesearch`

Attempts to take a standard Newton step, then if the step fails to show sufficient
reduction in the norm, it reduces the step size by the 'contraction_rate' factor.
Reduction of step size stops when it reaches the 'min_step' tolerance, regardless
of whether or not sufficient norm reduction was achieved. 

Guarantees that the residual reduction is monotonic, but may result in step sizes
too small to get a solution in reasonable timeframe. 

```
% Invoked via the 'options' struct() for Newton methods
options.linesearch = 'backtracking';
options.linesearch_opts = struct('min_step',1e-3,'contraction_rate',0.5);
```

---

# Citation

Ladshaw, A.P., "PJFNK MATLAB: A MATLAB implementation for Jacobian-Free
Newton-Krlov solvers for non-linear systems," https://github.com/aladshaw3/pjfnk_matlab, 
Accessed (Month) (Day), (Year).