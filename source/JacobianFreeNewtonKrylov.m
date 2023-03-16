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
    if (~isfield(options,'maxiter'))
        options.maxiter = 10*size(x0,1);
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
        if (~all(ismember(options.krylov_solver, {'gmres','bicgstab'}), 'all'))
            options.krylov_solver = 'gmres';
        end
    end
    if (~isfield(options,'krylov_opts'))
        options.krylov_opts = struct();
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

    % Initialize vectors
    F = fun(x0);
    s = zeros(size(x0,1), 1);
    Jop = @(F) JacobianOperator(fun,x0,F,options.epsilon);

    % Register the Krylov method
    if (strcmpi(options.krylov_solver,'gmres'))
        warning('off','MATLAB:gmres:tooSmallTolerance')

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
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
        end
        options.krylov_fun = @(Jop, F, s) gmres(Jop, F, ...
            options.krylov_opts.restart, options.krylov_opts.tol, options.krylov_opts.maxit, ...
                options.krylov_opts.M1, options.krylov_opts.M2, s);

    elseif (strcmpi(options.krylov_solver,'bicgstab'))
        warning('off','MATLAB:bicgstab:tooSmallTolerance')

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
        end
        if (~isfield(options.krylov_opts,'M2'))
            options.krylov_opts.M2 = [];
        end
        options.krylov_fun = @(Jop, F, s) bicgstab(Jop, F, ...
            options.krylov_opts.tol, options.krylov_opts.maxit, ...
                options.krylov_opts.M1, options.krylov_opts.M2, s);
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

    % NOTE: Likely need to make this a cell array since it is 'ragged'
    %stats.krylov_stats.resvec = zeros(options.maxiter,options.krylov_opts.maxit+1);

    stats.fnorm(1,1) = norm(F);
    stats.xnorm(1,1) = norm(-x0);
    
    % Iterate up to maxiter
    for i=1:options.maxiter
        % Perform a Jacobian-Free Newton-Krylov Step
        %   J*s = -F --> s = -J\F --> solved with Krylov Subspace
        %   step = s*a --> comes from linesearch function 
        %        Jop = @(F) JacobianOperator(fun,x,F,eps);
        %        s = gmres(Jop,-F); % (or other Krylov method)

        Jop = @(F) JacobianOperator(fun,x0,F,options.epsilon);
        [s,lin_flag,lin_relres,lin_iter,lin_resvec] = options.krylov_fun(Jop,-F, s);
        stats.krylov_stats.exit_status_id(i) = lin_flag;
        stats.krylov_stats.linear_res(i) = lin_relres;
        stats.krylov_stats.linear_steps(i) = max(lin_iter);

        %stats.krylov_stats.resvec(i,:) = lin_resvec(1:options.krylov_opts.maxit+1)';

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

    %stats.krylov_stats.resvec = stats.krylov_stats.resvec(1:stats.nl_iter,:);

    % Report any errors
    if (stats.converged_id == 0)
        stats.converged_message = append('Newton method failed to converge after', ...
                ' ', num2str(stats.nl_iter),' ','iterations');
        stats.converged_id = -1;
    end
end

