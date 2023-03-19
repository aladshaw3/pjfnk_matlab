% Extended integration testing for JFNK methods

%% Test 1 - Replication of Matlab example for large non-linear systems
n = 100;
x0 = zeros(n,1);

Jac = NumericalJacobianMatrix(@nlsf1a,x0,'sparse');

% Solver options
solver_info = struct();
solver_info.krylov_solver = 'bicgstab';
solver_info.linesearch = 'backtracking';
solver_info.use_matrix = false; 
solver_info.maxiter = 50;
solver_info.ftol = 1e-6;
solver_info.jacfun = @(x) dnlsf1a(x);

solver_info.krylov_opts = struct();
solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) tridiag_precon(b,Jacfun,x,options);

tic;
[x,stats,opts] = JacobianFreeNewtonKrylov(@nlsf1a,x0,solver_info);
toc;

assert( stats.fnorm(end) <= opts.ftol )


%% Test 2 - JFNK method with equilibriate options and dissect reordering

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
solver_info.maxiter = 5;
solver_info.ftol = 1e-6;

solver_info.use_matrix = true;
solver_info.jacfun = @(x) [A];

% Krylov options
solver_info.krylov_opts = struct();
solver_info.krylov_opts.restart = 20;
solver_info.krylov_opts.maxit = 20;

solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) ilu_precon(b,Jacfun,x,options);

solver_info.use_matrix = true;
solver_info.equilibrate = true;
solver_info.reordering_method = 'dissect';

tic;
[x,stats,opts] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;

assert( stats.fnorm(end) <= opts.ftol )



%% Test 3 - JFNK method with equilibriate options and amd reordering

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
solver_info.maxiter = 5;
solver_info.ftol = 1e-6;

solver_info.use_matrix = true;
solver_info.jacfun = @(x) [A];

% Krylov options
solver_info.krylov_opts = struct();
solver_info.krylov_opts.restart = 20;
solver_info.krylov_opts.maxit = 20;

solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) ilu_precon(b,Jacfun,x,options);

solver_info.use_matrix = true;
solver_info.equilibrate = true;
solver_info.reordering_method = 'amd';

tic;
[x,stats,opts] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;

assert( stats.fnorm(end) <= opts.ftol )


%% Test 4 - JFNK method with equilibriate options and symrcm reordering

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
solver_info.maxiter = 5;
solver_info.ftol = 1e-6;

solver_info.use_matrix = true;
solver_info.jacfun = @(x) [A];

% Krylov options
solver_info.krylov_opts = struct();
solver_info.krylov_opts.restart = 20;
solver_info.krylov_opts.maxit = 20;

solver_info.krylov_opts.M1 = @(b,Jacfun,x,options) ilu_precon(b,Jacfun,x,options);

solver_info.use_matrix = true;
solver_info.equilibrate = true;
solver_info.reordering_method = 'symrcm';

tic;
[x,stats,opts] = JacobianFreeNewtonKrylov(F,x0,solver_info);
toc;

assert( stats.fnorm(end) <= opts.ftol )


% Matlab function from the example 
function F = nlsf1a(x)
    % Evaluate the vector function
    n = length(x);
    F = zeros(n,1);
    i = 2:(n-1);
    F(i) = (3-2*x(i)).*x(i)-x(i-1)-2*x(i+1) + 1;
    F(n) = (3-2*x(n)).*x(n)-x(n-1) + 1;
    F(1) = (3-2*x(1)).*x(1)-2*x(2) + 1;
end

% Jacobian function (sparse)
function dF = dnlsf1a(x)
    % dF(1)/dx(1) = 3-4*x(1);
    % dF(1)/dx(2) = -2;

    % dF(i)/dx(i-1) = -1;
    % dF(i)/dx(i) = 3-4*x(i);
    % dF(i)/dx(i+1) = -2;


    % dF(n)/dx(n-1) = -1;
    % dF(n)/dx(n) = 3-4*x(n);

    n = size(x,1);
    e = ones(n,1);
    dF = spdiags([(e*-1) (e.*(3-4*x)) (e*-2)],-1:1,n,n);
end

% Efficiency can be improved by not call the Jacobian function and 
%   just passing in vectors through 'options' for known coefficients,
%   but this is a good generalization of this method to have on record
function M = tridiag_precon(d,Jac,x,options)
    J = Jac(x);
    n = size(J,1);
    app = zeros(n,1);
    dpp = zeros(n,1);
    M = zeros(n,1);

    % Elementwise division
    v = diag(J);
    dp = d./v;
    ap = [0; diag(J,-1)./v(2:end)];
    cp = [diag(J,1)./v(1:end-1); 0];

    % Reverse sweep
    for i=n:-1:1
        if (i==1)
            dpp(i) = (d(i)-(cp(i)*dpp(i+1)) ) / (1 - (cp(i)*app(i+1)));
            app(i) = 0;
        elseif (i==n)
            dpp(i) = dp(i);
            app(i) = ap(i);
        else
            dpp(i) = (d(i)-(cp(i)*dpp(i+1)) ) / (1 - (cp(i)*app(i+1)));
            app(i) = ap(i) / (1 - (cp(i)*app(i+1)));
        end
    end

    % Forward sweep
    for i=1:n
        if (i==1)
            M(i) = dpp(i);
        else
            M(i) = dpp(i) - (app(i) * M(i-1));
        end
    end
end


%% Helper function for ilu and ichol preconditioners
% Defined preconditioner helper
%       NOTE: x comes in as 'b' (or '-F') in this context
function y = ilu_precon(b,Jac,x,options)
    [L,U] = ilu(Jac(x),struct('type','ilutp','droptol',1e-6));
    y = U\(L\b);
end