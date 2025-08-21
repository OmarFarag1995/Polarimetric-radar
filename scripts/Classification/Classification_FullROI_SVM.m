clc;
clear;
close all;

%% Load the extracted ROIs and labels
load('Data\Feature_Vector_Data\full_roi_doppler.mat', 'allROIs', 'allLabels');

% Convert labels to categorical
Y = categorical(allLabels);

% Determine the size of the ROIs
numSamples = length(allROIs);
[rangeSize, dopplerSize] = size(allROIs{1});

% Convert the cell array to a 2D array for SVM input [samples, features]
combinedROIsArray = zeros(numSamples, rangeSize * dopplerSize);
for i = 1:numSamples
    combinedROIsArray(i, :) = reshape(allROIs{i}, [1, rangeSize * dopplerSize]);
end

% 5-fold cross-validation
k = 5;
cv = cvpartition(numSamples, 'KFold', k);

% Initialize results
bestAccuracy = 0;
bestModel = [];
bestHyperparams = struct();

for fold = 1:k
    % Training and test indices for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    % Create training and test sets
    X_train = combinedROIsArray(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test = combinedROIsArray(testIdx, :);
    Y_test = Y(testIdx);

    % Train SVM with hyperparameter optimization
    t = templateSVM('Standardize', true);
    svmModel = fitcecoc(X_train, Y_train, 'Learners', t, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'ShowPlots', false));

    % Predict on the test data
    YPred = predict(svmModel, X_test);
    accuracy = sum(YPred == Y_test) / numel(Y_test);
    disp(['Fold ', num2str(fold), ' - Classification Accuracy: ', num2str(accuracy)]);

    % Store the best model based on validation accuracy
    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestModel = svmModel;
        bestHyperparams = svmModel.HyperparameterOptimizationResults.XAtMinObjective;
    end
end

% Display best accuracy and hyperparameters
disp(['Best Mean Classification Accuracy: ', num2str(bestAccuracy)]);
disp('Best Hyperparameters:');
disp(bestHyperparams);

% Save the best model based on accuracy
save('Data\Models\best_full_roi_svm_model.mat', 'bestModel', 'bestHyperparams');

% Evaluate the best model on the full dataset for a final confusion matrix
Y_pred_full = predict(bestModel, combinedROIsArray);
accuracy_full = sum(Y_pred_full == Y) / numel(Y);
disp(['Final Classification Accuracy with Best Model: ', num2str(accuracy_full)]);

% Plot confusion matrix for the best model
figure;
confusionchart(Y, Y_pred_full);
title('Confusion Matrix');

