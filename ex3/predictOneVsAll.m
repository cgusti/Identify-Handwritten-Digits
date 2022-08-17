function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);      %No. of Input Examples to predict (Each row = 1 Example)
num_labels = size(all_theta, 1);   %No. of Output Classifier 

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);           %No_of_Input_Examples x 1 == m x 1 

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
%following code to make predictions using
%your learned logistic regression parameters (one-vs-all).
%You should set p to a vector of predictions (from 1 to
%num_labels).
%       

%Dimensions : all_theta = 10 x 401 

prob_mat = X * all_theta';   %5000 x 10 
[prob, p] = max(prob_mat, [], 2); %m x 1 


%returns maximum element in each row == max. probability and its index for
%each input image 

%p: predicted output (index) 
%prob: probability of predicted ouput 





% =========================================================================


end
