function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);     
num_labels = size(Theta2, 1);  

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  %m x 1 



%DIMENSIONS: 
% theta1 = 25 x 401 
% theta2 = 10 x 26 

% layer1 (input) = 400 nodes + 1 bias 
% layer2 (hidden) = 25 nodes + 1 bias 
% layer 3 (output) = 10 nodes 


a1 = [ones(m,1) X]; %Adding 1 in X 
%No. of rows = no. of input images 
%No. of Column  = No. of features in each image 

z2 = a1 * Theta1' %5000 x 25 
a2 = sigmoid(z2); % 5000 x 25 


a2 = [ones(size(a2,1),1) a2] % 5000 x 26 

z3 = a2 * Theta2' % 5000 x 10 
a3 = sigmoid(z3) % 5000 x 10 

[prob, p] = max(a3, [], 2);

%returns maximum element in each row == max. probability and its index for
%each input image 

%p = predicted output (index) 
%prob = probability of predicted outputs 
















% =========================================================================


end
