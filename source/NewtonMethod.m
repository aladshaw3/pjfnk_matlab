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
%   @param x0 Nx1 vector of non-linear variables
%   @param options A Matlab data structure that holds all function options
%      struct:
%        maxiter: The maximum allowable number of iterations
%        ftol: The solver residual tolerance
%        xtol: The solver non-linear variable tolerance
%        gtol: The solver slope/Jacobian tolerance 
%        mtype: The type of jacobian to use ['dense' or 'sparse']
%        epsilon: The size of the step to use for the numerical Jacobian
%        jacfun: function handle for evaluation of the Jacobian matrix at 
%                 each iteration [J = jacfun(fun,x)]
%        linesearch: ['none','backtracking']
%        linesearch_opts:  struct() whose form depends on linesearch method
%               for 'none' --> empty
%               for 'backtracking'
%                       min_step: The minimun step size
%                       contraction_rate: How fast to shrink step size
%
%   Returns:
%       x: Solution vector
%       stats: Solver stats
%       options: A copy of the options structure
function [x,stats,options] = NewtonMethod(fun,x0,options)
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
    if (~isfield(options,'linesearch'))
        options.linesearch = 'none';
    else
        if (~all(ismember(options.linesearch, {'none','backtracking'}), 'all'))
            options.linesearch = 'none';
        end
    end
    if (~isfield(options,'linesearch_opts'))
        options.linesearch_opts = struct();
    end

    % Register the linesearch method
    if (strcmpi(options.linesearch,'none'))
        options.linfun = @(fun,x,F,s,linopts) StandardNewtonStep(fun,x,F,s,linopts);
    elseif (strcmpi(options.linesearch,'backtracking'))
        options.linfun = @(fun,x,F,s,linopts) BacktrackLinesearch(fun,x,F,s,linopts);
        if (~isfield(options.linesearch_opts,'min_step'))
            options.linesearch_opts.min_step = 1e-3;
        end
        if (~isfield(options.linesearch_opts,'contraction_rate'))
            options.linesearch_opts.contraction_rate = 0.5;
        end
    end

    % Initialize vectors and structures 
    F = fun(x0);
    stats = struct();
    stats.nl_iter=0;
    stats.fnorm = zeros(options.maxiter+1,1);
    stats.xnorm = zeros(options.maxiter+1,1);
    stats.gnorm = zeros(options.maxiter+1,1);
    stats.converged_id = 0;
    stats.converged_message = 'Not Started';

    stats.fnorm(1,1) = norm(F);
    stats.xnorm(1,1) = norm(-x0);
    stats.gnorm(1,1) = norm(options.jacfun(x0),"Inf");
    
    % Iterate up to maxiter
    for i=1:options.maxiter
        % Perform a Newton Step
        %   J*s = -F --> s = -J\F
        %   step = s*a --> comes from linesearch function 
        %   s = - options.jacfun(x0)\F;
        x = x0 + options.linfun(fun,x0,F,-options.jacfun(x0)\F,options.linesearch_opts);

        % Update residuals and iteration count
        F = fun(x);
        stats.nl_iter=stats.nl_iter+1;
        stats.fnorm(i+1,1) = norm(F);
        stats.xnorm(i+1,1) = norm(x-x0);
        stats.gnorm(i+1,1) = norm(options.jacfun(x0),"Inf");

        % Check status
        if (stats.fnorm(i+1,1) <= options.ftol)
            stats.converged_id = 1;
            stats.converged_message = 'Residual below function tolerance';
            break;
        end
        if (stats.xnorm(i+1,1) <= options.xtol)
            stats.converged_id = 2;
            stats.converged_message = 'Difference in x below tolerance';
            break;
        end
        if (stats.gnorm(i+1,1) <= options.gtol)
            stats.converged_id = 3;
            stats.converged_message = 'Jacobian norm below slope tolerance';
            break;
        end

        % Update x
        x0=x;
    end

    % Slice the norm reports
    stats.fnorm = stats.fnorm(1:stats.nl_iter+1,1);
    stats.xnorm = stats.xnorm(1:stats.nl_iter+1,1);
    stats.gnorm = stats.gnorm(1:stats.nl_iter+1,1);

    % Report any errors
    if (stats.converged_id == 0)
        stats.converged_message = append('Newton method failed to converge after', ...
                ' ', num2str(stats.nl_iter),' ','iterations');
        stats.converged_id = -1;
    end
end

