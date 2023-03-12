[![Checks](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/aladshaw3/pjfnk_matlab/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/gh/aladshaw3/pjfnk_matlab/branch/main/graph/badge.svg)](https://codecov.io/gh/aladshaw3/pjfnk_matlab) 

# PJFNK Matlab
This repository is for an implementation of the Preconditioned Jacobian-Free Newton-Krylov method in Matlab

# Requirements

- MATLAB

# Functions

- JacobianOperator

Performs an action of a Jacobian on any vector v without forming the Jacobian
explicitly. 

Jv= (F(x+epsilon*v) - F(x))/epsilon;

where F(x) evaluates the non-linear system at state x.
where v is the vector being multiplied by the Jacobian.
where epsilon is a small perturbation value. 
