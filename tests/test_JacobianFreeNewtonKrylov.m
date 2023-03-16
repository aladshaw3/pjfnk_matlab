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