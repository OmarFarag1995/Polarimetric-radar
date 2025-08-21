clc;
clear;
close all;

% Load the extracted features
load('Data\Feature_Vector_Data\ego_doppler_spectral_features.mat');

% Convert labels to categorical
Y = combinedLabels;

% Normalize features
mu = mean(combinedFeatures, 1);
sigma = std(combinedFeatures, [], 1);
X = (combinedFeatures - mu) ./ sigma;

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(X, 1));
X = X(shuffledIndices, :);
Y = Y(shuffledIndices, :);

% Initialize cross-validation
k = 5;
cv = cvpartition(size(X, 1), 'KFold', k);

% Initialize array to hold accuracies
accuracies = zeros(k, 1);

% Store trained models
models = cell(k, 1);

for i = 1:k
    % Get the training and validation indices for this fold
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx, :);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx, :);
    
    % Train logistic regression model with regularization using fitcecoc
    model = fitcecoc(X_train, Y_train, ...
        'Learners', templateLinear('Learner', 'logistic', 'Regularization', 'lasso'), ...
        'Coding', 'onevsall', ...
        'Verbose', 0);

    % Store the trained model
    models{i} = model;

    % Make predictions
    Y_pred = predict(model, X_test);
    
    % Calculate accuracy
    accuracy = sum(Y_pred == Y_test) / numel(Y_test);
    accuracies(i) = accuracy;
    fprintf('Fold %d - Classification Accuracy: %.2f%%\n', i, accuracy * 100);
end

% Average accuracy over all folds
meanAccuracy = mean(accuracies);
fprintf('Average Classification Accuracy: %.2f%%\n', meanAccuracy * 100);

% Determine the best model based on accuracy
[~, bestFold] = max(accuracies);
bestModel = models{bestFold};

% Save the best model
save('EgoBinSepectrum_LogisticRegressionModel.mat', 'bestModel', 'mu', 'sigma');

% Evaluate the final model on the entire dataset (optional)
Y_pred_best = predict(bestModel, X);

% Calculate accuracy on the full dataset
accuracy_best = sum(Y_pred_best == Y) / numel(Y);
disp(['Final Model Classification Accuracy on Full Data: ', num2str(accuracy_best * 100), '%']);

% Plot confusion matrix for the final model
figure;
confusionchart(Y, Y_pred_best);
title('Confusion Matrix - Final Model');

% Feature importance analysis (using a simple method)
% Note: fitcecoc with lasso does not directly provide feature importance.
% We'll use the mean of absolute values of coefficients across learners.

% Initialize a matrix to collect coefficients
numFeatures = size(X, 2);
numLearners = length(bestModel.BinaryLearners);
coefficients = zeros(numFeatures, numLearners);

% Extract coefficients from each binary learner
for j = 1:numLearners
    learner = bestModel.BinaryLearners{j};
    % Check if the binary learner has the Beta field
    if isprop(learner, 'Beta')
        coefficients(:, j) = abs(learner.Beta);
    end
end

% Compute mean absolute coefficient values for feature importance
feature_importance = mean(coefficients, 2);

% Plot feature importance
figure;
bar(feature_importance);
xlabel('Feature Index');
ylabel('Importance Score');
title('Feature Importance');
xticks(1:numFeatures);
xticklabels(cellstr(num2str((1:numFeatures)')));
xtickangle(90);


fprintf('Classification is complete. best model saved to EgoBinSepectrum_LogisticRegressionModel.mat\n');

