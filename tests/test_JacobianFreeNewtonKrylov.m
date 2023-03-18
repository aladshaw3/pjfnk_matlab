% Unit tests for JacobianFreeNewtonKrylov

%% Test 1 - Basic non-linear solve (gmres - default)
F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x0 = [1; 1; 1];

% Solver options
solver_info = struct();

tic;
[x,stats,opts] = NewtonMethod(F,x0,solver_info);
toc;
tic;
[xjf,statsjf,optsjf] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;


% Assert the solutions are the same and took same number of nl steps
assert( norm(x-xjf) < 1e-6 )
assert( stats.nl_iter == statsjf.nl_iter )

% Check the opts and stats for defaults
assert( strcmpi(optsjf.krylov_solver,'gmres')  )


%% Test 2 - Picking a different Krylov method
F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x0 = [1; 1; 1];

% Solver options
solver_info = struct();
solver_info.linesearch = 'backtracking';
solver_info.krylov_solver = 'bicgstab';

tic;
[x,stats,opts] = NewtonMethod(F,x0,solver_info);
toc;
tic;
[xjf,statsjf,optsjf] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;


% Assert the solutions are the same and took same number of nl steps
assert( norm(x-xjf) < 1e-6 )
assert( stats.nl_iter == statsjf.nl_iter )

% Check the opts and stats for defaults
assert( strcmpi(optsjf.krylov_solver,'bicgstab')  )


%% Test 3 - Use gmres with preconditioning on linear system through PJFNK

% Load a sparse matrix A
load("gmres_test.mat")

% Construct BCs such that solution is vector of all 1s
b = sum(A,2);
x0 = zeros(size(b,1),1);

% Build the function to pass
F = @(x) [A*x-b];

% Setup solver into
solver_info = struct();
solver_info.krylov_solver = 'gmres';
solver_info.maxiter = 20;
solver_info.ftol = 1e-6;

% Create a constant preconditioner (only works because system is linear)
%       M = M1*M2  --> factor matrix A into L*U using ilu (approximate)
[L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));

solver_info.krylov_opts = struct();
solver_info.krylov_opts.restart = []; % No restarts
solver_info.krylov_opts.tol = 1e-6;
solver_info.krylov_opts.M1 = L;
solver_info.krylov_opts.M2 = U;

% Call the solver
tic;
[xjf,statsjf,optsjf] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;

assert( statsjf.fnorm(end) <= optsjf.ftol )

% NOTE: Specifying M as a Function Handle: 
% You can optionally specify any of M, M1, or M2 as function handles 
% instead of matrices. The function handle performs matrix-vector 
% operations instead of forming the entire preconditioner matrix, making 
% the calculation more efficient.
%
% To use a function handle, use the function signature function 
% x = mfun(b). Parameterizing Functions explains how to provide additional 
% parameters to the function mfun, if necessary. The function call mfun(b)
% must return the value of M\b or M2\(M1\b).


%% Test 4 - Use gmres on numerical jacobian with function preconditioner

% Load a sparse matrix A
load("gmres_test.mat")

% Construct BCs such that solution is vector of all 1s
b = sum(A,2);
x0 = zeros(size(b,1),1);

% Build the function to pass
F = @(x) [A*x-b];

% Setup solver into
solver_info = struct();
solver_info.krylov_solver = 'bicgstab';
solver_info.use_matrix = true;
solver_info.maxiter = 20;
solver_info.ftol = 1e-6;

% We will manually add the jacobian function here so that we can pass this
% info into our preconditioner. 
%   If we say 'use_matrix = true', but do not provide one, then it will 
%   be formulated automatically from NumericalJacobianMatrix
% solver_info.jacfun = @(x) [A];

solver_info.krylov_opts = struct();
solver_info.krylov_opts.restart = []; % No restarts
solver_info.krylov_opts.tol = 1e-6;


% NOTE: In order to generalize the preconditioning, we will need to be able
% to pass more information to the M1 and M2 functions. They must be
% templated to only accept a single vector 'b', but if our jacobian is
% numerical, then we may need to evaluate the jacobian or pass the jacobian
% to it. This may require checking to see if M1 and M2 are function
% handles, and if they are then passing additional args.

% Create a function preconditioner. The templated function must ONLY accept
% a vector x. HOWEVER, we can pass other args to it if they are defined
% before hand. NOTE: The scope of these are local, so 'A' has to be defined
% above here (and be unchanged) for this to work. Thus, more complex
% preconditioners may need some additional infrastructure to generalize. 
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) ilu_precon(b,Jacfun,x,options);

% Alternative: 1 part preconditioner
%[L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));
%solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) U\(L\b);

% Alternative: 2 part preconditioner 
%[L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));
%solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) L\b;
%solver_info.krylov_opts.M2 = @(b,Jacfun,x,options) U\b;

% Call the solver

tic;
[xjf,statsjf,optsjf] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;

assert( statsjf.fnorm(end) <= optsjf.ftol )

% Defined preconditioner helper
%       NOTE: x comes in as 'b' (or '-F') in this context
function M = ilu_precon(b,Jac,x,options)
    [L,U] = ilu(Jac(x),struct('type','ilutp','droptol',1e-6));
    M = U\(L\b);
end