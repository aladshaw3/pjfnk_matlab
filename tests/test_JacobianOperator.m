% Tests for the Jacobian Operator

%% Test 1 - Simple linear functions

F = @(x) [2*x(1) - x(2); 
            -x(1)+2*x(2)-x(3); 
            -x(2)+2*x(3)];

x = [1; 0; 0];

v = [0; 1; 0];

% Exact Jacobian
J = [2, -1, 0;
     -1, 2, -1;
     0, -1, 2];


% Exact Jv
Jv = J*v;

% Approximate Jv
Jv_approx = JacobianOperator(F,x,v);

assert( norm(Jv-Jv_approx) < 1e-6)

%% Test 2 - Non-linear functions

F = @(x) [2*x(1)*x(2) - x(2)^2; 
            -x(1)+2*x(2)-exp(-x(3)); 
            -x(2)+2*x(3)^3];

x = [1; 0; 0];

v = [0; 1; 0];

% Exact Jacobian
J = @(x)    [2*x(2), 2*x(1)-2*x(2), 0;
            -1, 2, exp(-x(3));
            0, -1, 6*x(3)^2];


% Exact Jv
Jv = J(x)*v;

% Approximate Jv
Jv_approx = JacobianOperator(F,x,v);

assert( norm(Jv-Jv_approx) < 1e-6)