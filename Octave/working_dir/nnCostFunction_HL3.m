 function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size_1, ...
                                   hidden_layer_size_2, ...
                                   hidden_layer_size_3, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size_1 * (input_layer_size + 1)), hidden_layer_size_1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + prod(size(Theta1))):((prod(size(Theta1)))+(hidden_layer_size_2*(hidden_layer_size_1+1)))), hidden_layer_size_2, (hidden_layer_size_1 + 1));
                 
Theta3 = reshape(nn_params((1 + prod(size([Theta2(:); Theta1(:)]))):((prod(size([Theta2(:); Theta1(:)])))+(hidden_layer_size_3*(hidden_layer_size_2+1)))), hidden_layer_size_3, (hidden_layer_size_2 + 1));
                   
Theta4 = reshape(nn_params((1 + prod(size([Theta3(:); Theta2(:); Theta1(:)]))):((prod(size([Theta3(:); Theta2(:); Theta1(:)])))+(num_labels*(hidden_layer_size_3+1)))), num_labels, (hidden_layer_size_3 + 1));          

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X=[ones(m,1) X];

%Parameter Initialisation
a_1=zeros(1, size(X)(2));
a_2=zeros(1, size(Theta2)(2));
a_3=zeros(1, size(Theta3)(2));
a_4=zeros(1, size(Theta4)(2));
a_5=zeros(1, num_labels);
dk_5=zeros(m,num_labels);

%Cost Function
for k=1:num_labels
 
    a_5(k)= [-(y==k)'*log(sigmoid([ones(m,1) sigmoid([ones(m,1) sigmoid([ones(m,1) sigmoid(X*Theta1')]*Theta2')]*Theta3')]*Theta4'))-(1-(y==k))'*log(1-sigmoid([ones(m,1) sigmoid([ones(m,1) sigmoid([ones(m,1) sigmoid(X*Theta1')]*Theta2')]*Theta3')]*Theta4'))](k);
    
end
J=sum(a_5) * 1/m + lambda/(2*m)*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2))+sum(sum(Theta3(:, 2:end).^2))+sum(sum(Theta4(:, 2:end).^2)));




a_1=X;
a_2=[ones(m,1) sigmoid(a_1*Theta1')];
a_3=[ones(m,1) sigmoid(a_2*Theta2')];
a_4=[ones(m,1) sigmoid(a_3*Theta3')];
a_5=sigmoid(a_3*Theta3');

for k=1:num_labels
  dk_5(:,k)=a_5(:,k)-(y==k);
end

dk_4=(dk_5*Theta4)(:,2:end).*sigmoidGradient(a_3*Theta3');
dk_3=(dk_4*Theta3)(:,2:end).*sigmoidGradient(a_2*Theta2');
dk_2=(dk_3*Theta2)(:,2:end).*sigmoidGradient(a_1*Theta1');

Theta1_grad=1/m*(Theta1_grad+ dk_2'*a_1);
Theta2_grad=1/m*(Theta2_grad+ dk_3'*a_2);
Theta3_grad=1/m*(Theta3_grad+ dk_4'*a_3);
Theta4_grad=1/m*(Theta4_grad+ dk_5'*a_4);


Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+ lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+ lambda/m*Theta2(:,2:end);
Theta3_grad(:,2:end)=Theta3_grad(:,2:end)+ lambda/m*Theta3(:,2:end);
Theta4_grad(:,2:end)=Theta4_grad(:,2:end)+ lambda/m*Theta4(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];


end
