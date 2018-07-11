function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); % 1682 x 100
Theta_grad = zeros(size(Theta)); % 943 x 100

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features - 1682 x 100
%        Theta - num_users  x num_features matrix of user features - 943 x 100
%        Y - num_movies x num_users matrix of user ratings of movies - 1682 x 943
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user -  1682 x 943
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


predictions = zeros(num_movies, num_users);
predictions = X * Theta'; %1682 x 943
error =  (predictions - Y);
error_factor = (error .* R).^2;
J = sum(error_factor(:)); 
Theta_sqr = Theta.^2;
X_sqr = X.^2;
J = J/2.0 + (sum(Theta_sqr(:)) * 0.5 * lambda) + (sum(X_sqr(:)) * 0.5 * lambda);

for i = 1:num_movies
  
  idx = find(R(i,:) == 1); 
  Theta_temp = Theta(idx,:);
  Y_temp = Y(i,idx);
  X_grad(i,:) = (X(i,:)*Theta_temp' - Y_temp)*Theta_temp + ( lambda * X(i,:) ); %  (1 x 100 * 100 x idx) = 1 x idx - 1 x idx = 1 x idx * idx * 100 = 1 x 100
  
endfor

for i = 1:num_users
  
  idx = find(R(:,i) == 1); % for a particular user, how many movie has he rated - [idx]
  X_temp = X(idx, :); % idx x 100 features of the movies that particular user has rated
  Y_temp = Y(idx,i); % idx x 1  
  Theta_grad(i,:) = (X_temp * Theta(i,:)' - Y_temp)'*X_temp + ( lambda * Theta(i,:) ) ; % idx x 100 * 100x 1 = idx x 1 - idx x 1 = (idx x 1)' = 1 x idx *  idx x 100
                    
endfor


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
