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


# Basic Usage

To use this function, you must provide (at a minimum) a:

 - Function Handle to evaluate a set of equations (F = @(x) ...)

 - An initial guess vector (x0)

The size of the vector (x0) and the size of the evaluated function handle (F(x))
must both be Nx1. 

```
% Example 1 - Basics

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x0 = [1; 1; 1];

[x,stats,opts] = JacobianFreeNewtonKrylov(F,x0);

```

The function will return the solution to the system (x) and 2 struct() objects 
containing: (i) the status information of the solve (stats) and (ii) a set 
of options used in the solve (opts). Several default options are provided.


You can also provide a struct() to the input arguments of the function to 
change the behavior of the solve as needed.

```
% Example 2 - Solver Options
options = struct();
options.maxiter = 20;  % Set the maximum number of non-linear steps
options.ftol = 1e-6;   % Set the solver tolerance
options.krylov_solver = 'gmres';        % Pick from a selection of Krylov methods
options.linesearch = 'backtracking';    % Pick from a selection of linesearch methods

% [Optional] Provide a function handle for a known Jacobian
options.jacfun = J = @(x)    [2*x(2), 2*x(1)-2*x(2), 0;
                                -1, 2, exp(-x(3));
                                 0, -1, 6*x(3)^2];
```

This is a Jacobian-Free method, so you technically never need to supply 
the Jacobian. However, you can actually choose to use the full (or sparse)
Jacobian (if you desire). 

```
options.use_matrix = true;   % Specifies to use the Jacobian for the solve
```

By default, the method will set `use_matrix` to false, and just use the 
`JacobianOperator` function in conjunction with a Krylov solver. However, 
if you tell the solver to use the matrix form, it will do that instead. 

For the Krylov solvers, you can also control how those methods are solved 
by adding to the `options` data structure. 

```
% Example 3 - Setting Krylov options for 'gmres'
options.krylov_opts = struct();
options.krylov_opts.restart = 50;  % Iterations before restart
options.krylov_opts.tol = 1e-6;    % Linear solver tolerance
```

**NOTE**: Each Krylov solver will have its own set of options. The options 
available (including preconditioners) follows exactly with the Matlab standard
set of options for each iterative solver (see [here](https://www.mathworks.com/help/matlab/math/iterative-methods-for-linear-systems.html)). 

Lastly, there is also a full `NewtonMethod` implementation. You can use this 
method if you do not want to use Krylov solvers underneath the non-linear 
problem. In this case, the search direction is resolved using the Matlab
`mldivide` (\) function. This method also has almost all the same options
available to it as the `JacobianFreeNewtonKrylov` method. 

```
% Example 4 - Using full Newton Method
F = @(x) [2*x(1) - x(2); 
            -x(1)+2*x(2)-x(3); 
            -x(2)+2*x(3) + 1];

x0 = [1; 0; 0];

% Exact Jacobian
J = @(x) [2, -1, 0;
            -1, 2, -1;
            0, -1, 2];

% Store the Jacobian function handle in the data structure 
options = struct();
options.jacfunc = J;

[x,stats,opts] = NewtonMethod(F,x0,options);
```

For more usage examples, see the `tests` folder of this repository.

---

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

# Preconditioning

Users may provide any custom preconditioning matrices or function handles 
to the `JacobianFreeNewtonKrylov` method. Those preconditioning functions 
must have the following format

```
% Custom preconditioner function used to help establish the search direction
%
%       Consider this function to be a Pseudo-Inverse for the following:
%           s = J(x)\b
%               where s = a vector to solve for, J(x) is the current 
%               Jacobian at the current non-linear state, and b is the 
%               known right-hand side vector
%
%   @param b = Nx1 vector representing the right-hand size of J*s = b
%   @param Jacfun = Function handle for the Jacobian (J = Jacfun(x))
%   @param x = Nx1 vector for the current non-linear state
%   @param options = An optional user defined struct() for any other information 
%               you may need to perform this action.
function s = custom_precon(b,Jacfun,x,options)
    ...
end
```

You provide this preconditioner to the `JacobianFreeNewtonKrylov` function
through the `options` struct().

```
options.krylov_opts = struct();
options.krylov_opts.M1 = @(b,Jacfun,x,options) custom_precon(b,Jacfun,x,options);
```

**NOTE**: We use the same preconditioning syntax from the standard Matlab 
library of Iterative Krylov Methods (see [here](https://www.mathworks.com/help/matlab/math/iterative-methods-for-linear-systems.html)). 
This allows users to provide either a single preconditioner Matrix/Function [M1], 
or a pair of preconditioners [M1 and M2]. If you provide both preconditioners, note 
that they are resolve in the following order: M2\(M1\b).

---

# Citation

Ladshaw, A.P., "PJFNK MATLAB: A MATLAB implementation for Jacobian-Free
Newton-Krlov solvers for non-linear systems," https://github.com/aladshaw3/pjfnk_matlab, 
Accessed (Month) (Day), (Year).