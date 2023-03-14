% Tests for the basic Newton Method

%% Test 1 - Simple linear functions (exact Jacobian)

F = @(x) [2*x(1) - x(2); 
            -x(1)+2*x(2)-x(3); 
            -x(2)+2*x(3) + 1];

x0 = [1; 0; 0];

% Exact Jacobian
J = @(x) [2, -1, 0;
            -1, 2, -1;
            0, -1, 2];

% Store the Jacobian function handle in the data structure 
solver_info = struct();
solver_info.jacfunc = J;

tic;
[x,stats] = NewtonMethod(F,x0,solver_info);
toc;

% Check for correct solution
assert( norm(x-[-0.25;-0.5;-0.75]) < 1e-6)
assert( stats.nl_iter==1 )

tic;
[x2,~,~,out] = fsolve(F,x0);
toc;

% Test against Matlab 'fsolve' solution
assert( norm(x-x2) < 1e-6)

%% Test 2 - Non-linear functions ('sparse')

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x0 = [1; 1; 1];

% Solver options
solver_info = struct();
solver_info.mtype = 'sparse';

tic;
[x,~,opts] = NewtonMethod(F,x0,solver_info);
toc;

assert(strcmp(opts.mtype,'sparse'))

tic;
[x2,~,~,out] = fsolve(F,x0);
toc;

% Test against Matlab 'fsolve' solution
%       NOTE: This problem has a non-unique solution space (i.e., 
%       more than one vector x can satisfy the solution tolerance)
%
%   Solution you land at depends on your starting guess
assert( norm(F(x)-F(x2)) < 1e-6)

% Force solution to stop immediately
x0 = [-1;0;0];
[x,stats,opts] = NewtonMethod(F,x0,solver_info);
assert(stats.nl_iter==1)

%% Test 3 - Plotting fnorm, xnorm, snorm with iterations

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3;
            x(3) - x(4)^2;
            -x(4) + x(5)*x(1)];

x0 = [1; 1; 1; 1; 1];

% Solver options
solver_info = struct();
solver_info.mtype = 'sparse';

tic;
[x,stats,opts] = NewtonMethod(F,x0,solver_info);
toc;

assert( norm(F(x)) < 1e-6 )

% Use the following to plot norm reduction with each iteration
% plot(1:stats.nl_iter, stats.fnorm)