%%  @package NumericalJacobianMatrix
%
%   @brief NumericalJacobianMatrix function that approximates the Jacobian
%   of a non-linear system numerically.
%
%   @details NumericalJacobianMatrix is a function to fully evaluate a 
%   Jacobian of non-linear system numerically. This action is performed
%   column-by-column using a small perturbation. This is useful for when
%   the JacobianOperator is not enough to solve the system and you want to
%   grab a full Jacobian matrix. 
%
%   @author Austin Ladshaw
%
%   @date 03/12/2023
%
%   @copyright This software was designed and built by Austin Ladshaw,
%   2023. It is openly available through the MIT License. 

%% NumericalJacobianMatrix
%
%       Calling this function will approximate the full Jacobian matrix
%       numerically with a small perturbation 'epsilon'.
%
%   @param fun User defined function handle [F = fun(x)]
%   @param x Nx1 vector of non-linear variables
%   @param mtype A string that is either 'sparse' or 'dense', depending on
%   the matrix type to return.
%   @param epsilon A perturbation value to formulate the approximate
%   Jacobian matrix [default = sum(sqrt(eps('double'))*(1+x))]
function J = NumericalJacobianMatrix(fun,x, mtype, epsilon)
    % Validate the inputs
    arguments
        fun (1,1) {mustBeA(fun,'function_handle')}
        x (:,1) {mustBeNumeric}
        mtype (1,:) {mustBeMember(mtype,{'dense','sparse'})} = 'dense'
        epsilon (1,1) {mustBeNumeric,mustBeReal} = sum(sqrt(eps('double'))*(1+x))
    end

    F = fun(x);
    N = size(x,1);
    M = size(F,1);

    if (strcmpi(mtype,'dense'))
        % Preallocate dense matrix
        J = zeros(M,N);
        for i=1:N
            dx = x;
            dx(i,1) =  x(i,1) + epsilon;
            J(:, i) = (fun(dx) - F)/epsilon;
        end
    else
        % Preallocate sparse matrix with M*N/4 non-zeros
        J = spalloc(M,N,ceil(max([M*N/100,10*M,10*N])));
        for i=1:N
            dx = x;
            dx(i,1) =  x(i,1) + epsilon;
            J = J+sparse(1:M,i*ones(1,M),((fun(dx) - F)/epsilon),M,N);
        end
    end
end

