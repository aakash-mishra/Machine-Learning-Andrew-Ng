function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




J = 0;
z = X * theta;
prediction = sigmoid(z);
J = sum( ( -y .* log(prediction) ) -  (1 - y) .* log( 1 - prediction) );
J = J/m;
reg_item = lambda / 2;
reg_item = reg_item / m;
reg = 0;
for j = 2:rows(theta)
  reg+=theta(j,1)^2 ;
endfor  
  J = J + (reg_item * reg) ;
  
% size of theta - 3 x 1

ans = sigmoid(z) - y;
for j = 1:rows(theta)
  if j == 1
    grad(j,1) = sum( ans .* X(:,j) );
    grad(j,1) = grad(j,1) / m;
  else 
    grad(j,1) = sum( ans .* X(:,j) );
    grad(j,1) = grad(j,1) / m;
    term = lambda / m;
    grad(j,1) = grad(j,1) + ( ( lambda * theta(j,1) ) / m );
  endif   
endfor



% =============================================================

end
