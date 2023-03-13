[![Checks](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/gh/aladshaw3/pjfnk_matlab/branch/main/graph/badge.svg)](https://codecov.io/gh/aladshaw3/pjfnk_matlab) 

# PJFNK Matlab
This repository is for an implementation of the Preconditioned Jacobian-Free Newton-Krylov method in Matlab

# Requirements

- MATLAB

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
