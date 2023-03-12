%%  @package JacobianOperator
%
%   @brief JacobianOperator function that approximates the action of the
%   Jacobian matrix on any vector v.
%
%   @details JacobianOperator function that approximates the action of the
%   Jacobian matrix on any Nx1 vector v. The user provides a function 'fun'
%   that evaluates an Nx1 column vector of non-linear equations with an Nx1
%   column vector 'x' of non-linear variables. 
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

%% JacobianOperator
%
%       Calling this function will approximate the action of a Jacobian
%       matrix on vector v and return that Nx1 column vector result.
%
%   @param fun User defined function handle [F = fun(x)]
%   @param x Nx1 vector of non-linear variables
%   @param v Nx1 vector to multiply by the Jacobian of fun
%   @param epsilon A perturbation value to formulate the approximate
%   Jacobian-vector product with [default = sum(sqrt(eps('double'))*(1+x))]
function Jv = JacobianOperator(fun, x, v, epsilon)
    % Validate the inputs
    arguments
        fun (1,1) {mustBeA(fun,'function_handle')}
        x (:,1) {mustBeNumeric}
        v (:,1) {mustBeNumeric}
        epsilon (1,1) {mustBeNumeric,mustBeReal} = sum(sqrt(eps('double'))*(1+x))
    end

    Jv= (fun(x+epsilon*v) - fun(x))/epsilon;
end

