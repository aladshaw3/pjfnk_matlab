%%  @package NewtonMethod
%
%   @brief NewtonMethod function that solves F(x) = 0 where F(x) is a
%   vector of non-linear functions and x is a vector of non-linear
%   variables.
%
%   @details NewtonMethod is a function that solves F(x) = 0 where F(x) is 
%   a vector of non-linear functions and x is a vector of non-linear
%   variables. Both F(x) and x must be Nx1 vectors of numerical values.
%   User may also supply some convergence criterion information, as well as
%   a choice of line-search method. If no Jacobian matrix is provided, then
%   one is formed numerically.
%
%   @author Austin Ladshaw
%
%   @date 03/12/2023
%
%   @copyright This software was designed and built by Austin Ladshaw,
%   2023. It is openly available through the MIT License. 

%% NewtonMethod
%
%       Calling this function will use the standard Newton's method to
%       solve a non-linear system of equations for a user provided function
%       'fun' and user provided initial guess vector x0;
%
%   @param fun User defined function handle [F = fun(x)]
%   @param x Nx1 vector of non-linear variables
%   @param options A Matlab data structure that holds all function options
%      struct:
%        maxiter: The maximum allowable number of iterations
%        ftol: The solver residual tolerance
%        xtol: The solver non-linear variable tolerance
%        gtol: The solver slope/Jacobian tolerance 
%        mtype: The type of jacobian to use ['dense' or 'sparse']
%        epsilon: The size of the step to use for the numerical Jacobian
%         jacfun: function handle for evaluation of the Jacobian matrix at 
%                 each iteration [J = jacfun(fun,x)]
%
%   Returns:
%       x: Solution vector
%       stats: Solver stats
function [x,stats] = NewtonMethod(fun,x0,options)
    % Validate the inputs
    arguments
        fun (1,1) {mustBeA(fun,'function_handle')}
        x0 (:,1) {mustBeNumeric}
        options (:,1) {mustBeA(options,'struct')} = struct()
    end

    % Check on user provided 'options'
    if (~isfield(options,'mtype'))
        options.mtype = 'dense';
    end
    if (~isfield(options,'epsilon'))
        options.epsilon = sum(sqrt(eps('double'))*(1+x0));
    end
    if (~isfield(options,'jacfun'))
        options.jacfun = @(x) NumericalJacobianMatrix(fun,x,options.mtype,options.epsilon);
    end
    if (~isfield(options,'maxiter'))
        options.maxiter = 10*size(x0,1);
    end
    if (~isfield(options,'ftol'))
        options.ftol = 1e-6;
    end
    if (~isfield(options,'xtol'))
        options.xtol = 1e-10;
    end
    if (~isfield(options,'gtol'))
        options.gtol = 1e-10;
    end

    % Initialize vectors and structures 
    F = fun(x0);
    stats = struct();
    stats.nl_iter=0;
    stats.converged_id = 0;
    stats.converged_message = 'Not Started';
    
    % Iterate up to maxiter
    for i=1:options.maxiter
        % Perform 1 Newton Step
        %   J*s = -F --> s = -J\F
        x = x0 - options.jacfun(x0)\F;

        % Update residuals and iteration count
        F = fun(x);
        stats.nl_iter=stats.nl_iter+1;

        % Check status
        if (norm(F) <= options.ftol)
            stats.converged_id = 1;
            stats.converged_message = 'Residual below function tolerance';
            break;
        end
        if (norm(x-x0) <= options.xtol)
            stats.converged_id = 2;
            stats.converged_message = 'Difference in x below tolerance';
            break;
        end
        if (norm(options.jacfun(x)) <= options.gtol)
            stats.converged_id = 3;
            stats.converged_message = 'Jacobian norm below slope tolerance';
            break;
        end

        % Update x
        x0=x;
    end

    % Report any errors
    if (stats.converged_message < 1)
        stats.converged_message = 'Newton method failed to converge after' ...
                + num2str(stats.nl_iter) + ' iterations';
        stats.converged_id = -1;
    end
end
