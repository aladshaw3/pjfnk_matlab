% Tests for the Numerical Jacobian

%% Test 1 - Simple linear functions (dense)

F = @(x) [2*x(1) - x(2); 
            -x(1)+2*x(2)-x(3); 
            -x(2)+2*x(3)];

x = [1; 0; 0];

% Exact Jacobian
J = [2, -1, 0;
     -1, 2, -1;
     0, -1, 2];

% Numerical Jacobian

J_approx = NumericalJacobianMatrix(F,x);

assert( norm(J-J_approx) < 1e-6)


%% Test 2 - Simple linear functions (sparse)

F = @(x) [2*x(1) - x(2); 
            -x(1)+2*x(2)-x(3); 
            -x(2)+2*x(3)];

x = [1; 0; 0];

% Exact Jacobian
J = [2, -1, 0;
     -1, 2, -1;
     0, -1, 2];

% Numerical Jacobian

J_approx = NumericalJacobianMatrix(F,x,'sparse');

assert( norm(J-J_approx) < 1e-6)

%% Test 3 - Non-linear functions ('dense')

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x = [1; 0; 0];

% Exact Jacobian
J = @(x)    [2*x(2), 2*x(1)-2*x(2), 0;
            -1, 2, exp(-x(3));
            0, -1, 6*x(3)^2];


% Approximate Jv
J_approx = NumericalJacobianMatrix(F,x);

assert( norm(J(x)-J_approx) < 1e-6)

%% Test 4 - Non-linear functions ('sparse')

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x = [1; 0; 0];

% Exact Jacobian
J = @(x)    [2*x(2), 2*x(1)-2*x(2), 0;
            -1, 2, exp(-x(3));
            0, -1, 6*x(3)^2];


% Approximate Jv
J_approx = NumericalJacobianMatrix(F,x,'sparse');

assert( norm(J(x)-J_approx) < 1e-6)