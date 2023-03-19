%%  @package BacktrackLinesearch
%
%   @brief BacktrackLinesearch function that returns a Newton type step
%
%   @details BacktrackLinesearch is a function that returns a Newton type
%   step that is contracted to ensure smooth convergence towards a local
%   solution/minima.
%
%   Source: Armijo, Larry (1966). "Minimization of functions having 
%   Lipschitz continuous first partial derivatives". Pacific J. Math. 
%   16 (1): 1–3. doi:10.2140/pjm.1966.16.1.
%
%   @author Austin Ladshaw
%
%   @date 03/14/2023
%
%   @copyright This software was designed and built by Austin Ladshaw,
%   2023. It is openly available through the MIT License. 

%% BacktrackLinesearch
%
%       Calling this function will attempt to take a Newton step, but will
%       also shrink the size of that step as long as the step results in
%       worse residual norms.
%
%   @param fun User defined function handle [F = fun(x)]
%   @param x Nx1 vector of non-linear variables
%   @param F Nx1 vector of current residuals
%   @param s Nx1 vector of search directions (s = -J\F)
%   @param options A Matlab data structure that holds all function options
function s = BacktrackLinesearch(fun,x,F,s,options)
    % Initial step size
    a = 1;
    amin = options.min_step;
    tau = options.contraction_rate;

    while (norm(fun(x+a*s)) > norm(F) && a > amin)
        a = a*tau;
    end

    % Finalized search direction (scaled)
    s=s*a;
end

