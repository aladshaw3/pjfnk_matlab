% Unit tests for linesearch methods

%% Test 1 - Backtracking vs No Line Search for very non-linear problem


% This function will actually produce worse norm reduction at the first
% newton step. When you take a basic newton step, it allows the norm to get
% worse. When you use backtracking, you force the method to produce
% monotonically decreasing residuals. 
F = @(x) [2*x(1)*x(2) - 3.3*x(2)^2; 
            -x(1)+2*x(2)-10*exp(-10*x(3)); 
            -x(2)^2+2*x(3)^3];

x0 = [1; 1; 1];

% Solver options
solver_info = struct();
solver_info.mtype = 'sparse';
solver_info.linesearch = 'backtracking';
solver_info.linesearch_opts = struct();
solver_info.linesearch_opts.min_step = 1e-3;

tic;
[x,stats,opts] = NewtonMethod(F,x0,solver_info);
toc;

% Solver options (no linesearch)
solver_info2 = struct();
solver_info2.mtype = 'sparse';
solver_info2.linesearch = 'none';

tic;
[x2,stats2,opts2] = NewtonMethod(F,x0,solver_info2);
toc;

% Prove that the backtracking has better norm reduction at the initial step
assert( stats.fnorm(2,1) < stats2.fnorm(2,1) )

% Plot reduction of norms against each other
% plot([1:stats.nl_iter+1],log10(stats.fnorm),[1:stats2.nl_iter+1],log10(stats2.fnorm))
