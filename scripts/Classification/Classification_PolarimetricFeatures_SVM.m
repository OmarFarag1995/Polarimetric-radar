
clear;
close all;

% Load the extracted features
load('..\Data\Feature_Vector_Data\polarimteric_features.mat');

% Convert labels to categorical
Y = combinedLabels;
nClasses = unique(Y);

% Note: the intial SVM model only had first two features
combinedFeatures = combinedFeatures;

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
accuracies_train = zeros(k, 1);
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
    
    % Train SVM classifier using fitcecoc
    t = templateSVM('Standardize',true,'KernelFunction','gaussian');
    model = fitcecoc(X_train,Y_train,'Learners',t,...
    'ClassNames',unique(Y_train)');

    % if needed with hyperparameter tuning
    % model = fitcecoc(X_train,Y_train,'Learners',t,...
    % 'ClassNames',unique(Y_train)','OptimizeHyperparameters','auto',...
    % 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    % 'expected-improvement-plus'));


    % Store the trained model
    models{i} = model;

    % Train Acccuracy 
    Y_train_pred = predict(model, X_train);
    
    % Calculate accuracy
    accuracy_train = sum(Y_train_pred == Y_train) / numel(Y_train);
    accuracies_train(i) = accuracy_train;

    % Test Data Accuracy 
    % Make predictions
    Y_test_pred = predict(model, X_test);
    
    % Calculate accuracy
    accuracy = sum(Y_test_pred == Y_test) / numel(Y_test);
    accuracies(i) = accuracy;
    fprintf('Fold %d - Classification Train Accuracy: %.2f%%  Test Accuracy: %.2f%%\n', i,accuracy_train*100,accuracy*100);
end

% Average accuracy over all folds
meanAccuracy = mean(accuracies);
fprintf('Average Test Classification Accuracy: %.2f%%\n', meanAccuracy * 100);

% Determine the best model based on accuracy
[~, bestFold] = max(accuracies);
bestModel = models{bestFold};

% Save the best model
save('..\Saved_Models\PolarimetricFeatures_SVMModel.mat', 'bestModel', 'mu', 'sigma');

% Evaluate the final model on the entire dataset (optional)
Y_pred_best = predict(bestModel, X);

% Calculate accuracy on the full dataset
accuracy_best = sum(Y_pred_best == Y) / numel(Y);
disp(['Final Model Classification Accuracy on Full Data: ', num2str(accuracy_best * 100), '%']);

% Plot confusion matrix for the final model
figure;
confusionchart(Y, Y_pred_best);
title('Confusion Matrix - Final Model');

fprintf('Classification is complete. best model saved to PolarimetricFeature_SVMModel.mat\n');

