%%  @package JacobianFreeNewtonKrylov
%
%   @brief JacobianFreeNewtonKrylov function that solves F(x) = 0 using a
%   Krylov subspace method with the Jacobian operator
%
%   @details JacobianFreeNewtonKrylov is a function that solves F(x) = 0
%   using a Krylov subspace method. The system does not actually need a
%   Jacobian, nor does it need to form a numerical Jacobian. Instead, it
%   uses a Jacobian Operator function that approximates the action of a
%   matrix on a vector. In this way, this methodology is entirely
%   matrix-free. 
%
%   Source: Knoll D.A., Keyes D.E. "Jacobian-Free Newton-Krylov methods: a 
%       survey of approaches and applications", Journal of Computational 
%       Physics, August 2003.
%
%   @author Austin Ladshaw
%
%   @date 03/12/2023
%
%   @copyright This software was designed and built by Austin Ladshaw,
%   2023. It is openly available through the MIT License. 

%% JacobianFreeNewtonKrylov
%
%       Calling this function will use the Jacobian-Free Newton-Krylov
%       method to solve a non-linear system of equations for a user 
%       provided function 'fun' and user provided initial guess vector x0;
%
%   @param fun User defined function handle [F = fun(x)]
%   @param x0 Nx1 vector of non-linear variables
%   @param options A Matlab data structure that holds all function options
%      struct:
%        maxiter: The maximum allowable number of iterations
%        ftol: The solver residual tolerance
%        xtol: The solver non-linear variable tolerance
%        disp_warnings: ['on','off'] (default = 'off')
%        use_matrix: [true, false] (default = false)
%        epsilon: The size of the step to use for the Jacobian operator
%        mtype: The type of jacobian to use ['dense','sparse'] (optional)
%        jacfun: function handle for evaluation of the Jacobian matrix at 
%                 each iteration [J = jacfun(fun,x)] (optional)
%        krylov_solver: ['gmres', 'bicgstab', 'pcg', 'minres', 'symmlq', ...
%                        'bicgstabl', 'cgs', 'tfqmr']
%        krylov_opts: struct() whose form depends on the krylov solver
%               for more information, see the MathWorks documentation:
%               https://www.mathworks.com/help/matlab/math/iterative-methods-for-linear-systems.html
%        linesearch: ['none','backtracking']
%        linesearch_opts:  struct() whose form depends on linesearch method
%               for 'none' --> empty
%               for 'backtracking'
%                       min_step: The minimun step size
%                       contraction_rate: How fast to shrink step size
%         equilibrate: [true, false] (default = false)
%         reordering_method: ['dissect','amd','symrcm'] (default = 'dissect')
%
%   Returns:
%       x: Solution vector
%       stats: Solver stats
%       options: A copy of the options structure
function [x,stats,options] = JacobianFreeNewtonKrylov(fun,x0, options)
    % Validate the inputs
    arguments
        fun (1,1) {mustBeA(fun,'function_handle')}
        x0 (:,1) {mustBeNumeric}
        options (:,1) {mustBeA(options,'struct')} = struct()
    end

    % Check on user provided 'options'
    if (~isfield(options,'epsilon'))
        options.epsilon = sum(sqrt(eps('double'))*(1+x0));
    end
    if (~isfield(options,'use_matrix'))
        options.use_matrix = false;
    else
        if (~islogical(options.use_matrix))
            options.use_matrix = false;
        end
    end
    if (~isfield(options,'disp_warnings'))
        options.disp_warnings = 'off';
    else
        if (~all(ismember(options.disp_warnings, {'on','off'}), 'all'))
            options.disp_warnings = 'off';
        end
    end
    if (~isfield(options,'mtype'))
        options.mtype = 'sparse';
    end
    if (~isfield(options,'jacfun'))
        options.jacfun = @(x) NumericalJacobianMatrix(fun,x,options.mtype,options.epsilon);
    end
    if (~isfield(options,'maxiter'))
        options.maxiter = min(10*size(x0,1), 30);
    end
    if (~isfield(options,'ftol'))
        options.ftol = 1e-6;
    end
    if (~isfield(options,'xtol'))
        options.xtol = 1e-10;
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
    if (~isfield(options,'krylov_solver'))
        options.krylov_solver = 'gmres';
    else
        if (~all(ismember(options.krylov_solver, {'gmres','bicgstab','pcg','minres', 'symmlq','bicgstabl', 'cgs', 'tfqmr'}), 'all'))
            options.krylov_solver = 'gmres';
        end
    end
    if (~isfield(options,'krylov_opts'))
        options.krylov_opts = struct();
    end
    if (~isfield(options.krylov_opts,'user_data'))
        options.krylov_opts.user_data = struct();
    end
    if (~isfield(options,'equilibrate'))
        options.equilibrate = false;
    end
    if (~isfield(options,'reordering_method'))
        options.reordering_method = 'dissect';
    else
        if (~all(ismember(options.reordering_method, {'dissect','amd','symrcm'}), 'all'))
            options.reordering_method = 'dissect';
        end
    end
    if (strcmpi(options.reordering_method,'dissect'))
        options.reorder_func = @(A) dissect(A);
    elseif (strcmpi(options.reordering_method,'amd'))
        options.reorder_func = @(A) amd(A);
    elseif (strcmpi(options.reordering_method,'symrcm'))
        options.reorder_func = @(A) symrcm(A);
    end

    % Apply warning choice
    warning(options.disp_warnings,'MATLAB:illConditionedMatrix')
    warning(options.disp_warnings,'MATLAB:gmres:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:bicgstab:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:symmlq:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:minres:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:pcg:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:bicgstabl:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:cgs:tooSmallTolerance')
    warning(options.disp_warnings,'MATLAB:tfqmr:tooSmallTolerance')
    

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

    % Initialize vectors
    F = fun(x0);
    s = zeros(size(x0,1), 1);
    Jop = @(F) JacobianOperator(fun,x0,F,options.epsilon);

    % Register the Krylov method
    if (strcmpi(options.krylov_solver,'gmres'))

        % Checkfor gmres options
        %       gmres(A,b,restart,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'restart'))
            options.krylov_opts.restart = min(30,size(x0,1));
        end
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) gmres(Jop, F, ...
            options.krylov_opts.restart, options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);

    elseif (strcmpi(options.krylov_solver,'bicgstab'))

        % Checkfor bicgstab options
        %       bicgstab(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) bicgstab(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);

    elseif (strcmpi(options.krylov_solver,'pcg'))

        % Checkfor pcg options
        %       pcg(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) pcg(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);


    elseif (strcmpi(options.krylov_solver,'minres'))

        % Checkfor minres options
        %       minres(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) minres(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);


    elseif (strcmpi(options.krylov_solver,'symmlq'))

        % Checkfor symmlq options
        %       symmlq(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) symmlq(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);


    elseif (strcmpi(options.krylov_solver,'bicgstabl'))

        % Checkfor symmlq options
        %       symmlq(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) bicgstabl(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);


    elseif (strcmpi(options.krylov_solver,'cgs'))

        % Checkfor symmlq options
        %       symmlq(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) cgs(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);


    elseif (strcmpi(options.krylov_solver,'tfqmr'))

        % Checkfor symmlq options
        %       symmlq(A,b,tol,maxit,M1,M2,x0)
        if (~isfield(options.krylov_opts,'tol'))
            options.krylov_opts.tol = 1e-4;
        end
        if (~isfield(options.krylov_opts,'maxit'))
            options.krylov_opts.maxit = min(2*size(x0,1),20);
        end
        if (~isfield(options.krylov_opts,'M1'))
            options.krylov_opts.M1 = [];
            M1 = [];
        else
            if (isa(options.krylov_opts.M1,'function_handle'))
                M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M1 = options.krylov_opts.M1;
            end
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
            M2 = [];
        else
            if (isa(options.krylov_opts.M2,'function_handle'))
                M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
            else
                M2 = options.krylov_opts.M2;
            end
        end
        options.krylov_fun = @(Jop, F, x0, M1, M2) tfqmr(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                M1, M2, []);

    end

    % Check on preconditioners
    if (~isa(options.krylov_opts.M1,'function_handle') && ~isempty(options.krylov_opts.M1))
        if (options.equilibrate && options.use_matrix)
            warning("Cannot use Matrix Reordering and Equilibriate methods if user DOES NOT supply M1 preconditioner as a function handle")
            options.equilibrate = false;
        end
    end
    if (~isa(options.krylov_opts.M2,'function_handle') && ~isempty(options.krylov_opts.M2))
        if (options.equilibrate && options.use_matrix)
            warning("Cannot use Matrix Reordering and Equilibriate methods if user DOES NOT supply M2 preconditioner as a function handle")
            options.equilibrate = false;
        end
    end

    % Initialize structures 
    stats = struct();
    stats.nl_iter=0;
    stats.fnorm = zeros(options.maxiter+1,1);
    stats.xnorm = zeros(options.maxiter+1,1);
    stats.converged_id = 0;
    stats.converged_message = 'Not Started';
    stats.krylov_stats = struct();
    stats.krylov_stats.exit_status_id = zeros(options.maxiter,1);
    stats.krylov_stats.linear_res = zeros(options.maxiter,1);
    stats.krylov_stats.linear_steps = zeros(options.maxiter,1);
    stats.krylov_stats.resvec = cell(options.maxiter,1);

    stats.fnorm(1,1) = norm(F);
    stats.xnorm(1,1) = norm(-x0);
    
    % Iterate up to maxiter
    for i=1:options.maxiter
        % Perform a Jacobian-Free Newton-Krylov Step
        %   J*s = -F --> s = -J\F --> solved with Krylov Subspace
        %   step = s*a --> comes from linesearch function 
        %        Jop = @(F) JacobianOperator(fun,x,F,eps);
        %        s = gmres(Jop,-F); % (or other Krylov method)

        % NOTE: Can replace Jop with options.jacfun(x0) to use the matrix 
        Jop = @(F) JacobianOperator(fun,x0,F,options.epsilon);

        % These resets of handles are required due to the need to update x0
        % values in the jacobian each time 
        if (isa(options.krylov_opts.M1,'function_handle'))
            M1 = @(b) options.krylov_opts.M1(b,options.jacfun,x0,options.krylov_opts.user_data);
        end
        if (isa(options.krylov_opts.M2,'function_handle'))
            M2 = @(b) options.krylov_opts.M2(b,options.jacfun,x0,options.krylov_opts.user_data);
        end
        

        if (options.use_matrix)
            if (options.equilibrate == false)
                [s,lin_flag,lin_relres,lin_iter,lin_resvec] = options.krylov_fun(options.jacfun(x0),-F, x0, M1, M2);
            else
                J = options.jacfun(x0);
                [P,R,C] = equilibrate(options.jacfun(x0));
                Jnew = R*P*J*C;
                Fnew = R*P*F;
                q = options.reorder_func(Jnew);
                Jnew = Jnew(q,q);
                Fnew = Fnew(q);
                JacfunNew = @(x0) [Jnew;];
                if (isa(options.krylov_opts.M1,'function_handle'))
                    M1 = @(b) options.krylov_opts.M1(b,JacfunNew,x0,options.krylov_opts.user_data);
                end
                if (isa(options.krylov_opts.M2,'function_handle'))
                    M2 = @(b) options.krylov_opts.M2(b,JacfunNew,x0,options.krylov_opts.user_data);
                end
                [snew,lin_flag,lin_relres,lin_iter,lin_resvec] = options.krylov_fun(Jnew,-Fnew, x0, M1, M2);
                s(q) = snew;
                s = C*s(:);
            end
        else
            [s,lin_flag,lin_relres,lin_iter,lin_resvec] = options.krylov_fun(Jop,-F, x0, M1, M2);
        end

        stats.krylov_stats.exit_status_id(i) = lin_flag;
        stats.krylov_stats.linear_res(i) = lin_relres;
        stats.krylov_stats.linear_steps(i) = max(lin_iter);
        stats.krylov_stats.resvec{i,1} = lin_resvec;

        x = x0 + options.linfun(fun,x0,F,s,options.linesearch_opts);

        % Update residuals and iteration count
        F = fun(x);
        stats.nl_iter=stats.nl_iter+1;
        stats.fnorm(i+1,1) = norm(F);
        stats.xnorm(i+1,1) = norm(x-x0);

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

        % Update x
        x0=x;
    end

    % Slice the norm reports
    stats.fnorm = stats.fnorm(1:stats.nl_iter+1,1);
    stats.xnorm = stats.xnorm(1:stats.nl_iter+1,1);
    stats.krylov_stats.exit_status_id = stats.krylov_stats.exit_status_id(1:stats.nl_iter,1);
    stats.krylov_stats.linear_res = stats.krylov_stats.linear_res(1:stats.nl_iter,1);
    stats.krylov_stats.linear_steps = stats.krylov_stats.linear_steps(1:stats.nl_iter,1);
    stats.krylov_stats.resvec(stats.nl_iter+1:end,:) = [];

    % Report any errors
    if (stats.converged_id == 0)
        stats.converged_message = append('Newton method failed to converge after', ...
                ' ', num2str(stats.nl_iter),' ','iterations');
        stats.converged_id = -1;
    end
end

