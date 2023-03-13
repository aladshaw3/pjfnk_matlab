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
newton = toc;

% Check for correct solution
assert( norm(x-[-0.25;-0.5;-0.75]) < 1e-6)

tic;
x2 = fsolve(F,x0);
dogleg = toc;

% Test against Matlab 'fsolve' solution
assert( norm(x-x2) < 1e-6)

% Validate that this implementation is faster
assert(newton<dogleg)