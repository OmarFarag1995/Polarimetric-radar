clc;
clear;
close all;

% Load the extracted features
load('enhanced_road_types_features.mat');

% Reshape features for classification
[num_samples, num_channels, num_features] = size(combinedFeatures);
X = reshape(combinedFeatures, num_samples, num_channels * num_features);

% Convert labels to categorical
Y = categorical(combinedLabels);

% Number of folds for cross-validation
k = 5;

% Initialize arrays to store accuracies and models
accuracies = zeros(k, 1);
models = cell(k, 1);

% Create cross-validation partition
cv = cvpartition(size(X,1), 'KFold', k);

for fold = 1:k
    % Training and test indices for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Create training and test sets
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx);

    % Train regularized logistic regression model with lasso regularization using fitcecoc
    t = templateLinear('Learner', 'logistic', 'Regularization', 'lasso', 'Solver', 'sparsa');
    model = fitcecoc(X_train, Y_train, 'Learners', t, 'Coding', 'onevsall');

    % Evaluate the model
    Y_pred = predict(model, X_test);
    accuracy = sum(Y_pred == Y_test) / numel(Y_test);
    disp(['Fold ', num2str(fold), ' - Classification Accuracy: ', num2str(accuracy)]);
    
    % Store accuracy and model
    accuracies(fold) = accuracy;
    models{fold} = model;
end

% Display mean accuracy across all folds
meanAccuracy = mean(accuracies);
disp(['Mean Classification Accuracy: ', num2str(meanAccuracy)]);

% Save the best model based on accuracy
[~, bestFoldIdx] = max(accuracies);
bestModel = models{bestFoldIdx};
save('best_cv_logistic_model.mat', 'bestModel');

% Evaluate the best model on the full dataset for a final confusion matrix
Y_pred_full = predict(bestModel, X);
accuracy_full = sum(Y_pred_full == Y) / numel(Y);
disp(['Final Classification Accuracy with Best Model: ', num2str(accuracy_full)]);

% Plot confusion matrix for the best model
figure;
confusionchart(Y, Y_pred_full);
title('Confusion Matrix - Best CV Logistic Regression Classification');

% Feature importance analysis (using a simple method)
feature_importance = abs(bestModel.BinaryLearners{1}.Beta);

% Plot feature importance
figure;
bar(feature_importance);
xlabel('Feature Index');
ylabel('Importance Score');
title('Feature Importance');
xticks(1:size(feature_importance, 1));
xticklabels(cellstr(num2str((1:size(feature_importance, 1))')));
xtickangle(90);
