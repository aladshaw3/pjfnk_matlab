%%  @package StandardNewtonStep
%
%   @brief StandardNewtonStep function that returns a standard Newton step
%   without any linesearching applied.
%
%   @details StandardNewtonStep is a function that returns a standard 
%   Newton step without any linesearching applied.
%
%           s = -J\F
%
%   @author Austin Ladshaw
%
%   @date 03/14/2023
%
%   @copyright This software was designed and built by Austin Ladshaw,
%   2023. It is openly available through the MIT License. 

%% StandardNewtonStep
%
%       Calling this function will just return s because the standard
%       Newton step doesn't scale the step size at all. However, this
%       function does template other linesearch functions with specific
%       info needed to evaluate step sizes (i.e., a function handle 'fun'
%       for the system, current non-linear state 'x', current step 's', and
%       a struct for data options 'options')
function s = StandardNewtonStep(fun,x,F,s,options)
    % NOTE: Since this is a standard step, we do nothing but return s
end

