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

% Convert the cell array to a 2D array for NN input [samples, features]
combinedROIsArray = zeros(numSamples, rangeSize * dopplerSize);
for i = 1:numSamples
    combinedROIsArray(i, :) = reshape(allROIs{i}, [1, rangeSize * dopplerSize]);
end

% Define the parameter grid for NN
paramGrid = [
    struct('layerSize1', 64, 'layerSize2', 32, 'dropoutRate', 0.3);
    struct('layerSize1', 128, 'layerSize2', 64, 'dropoutRate', 0.4);
    struct('layerSize1', 256, 'layerSize2', 128, 'dropoutRate', 0.5);
    % Add more parameter combinations if needed
];

% Initialize results
bestAccuracy = 0;
bestParams = struct();
bestNet = [];

% 5-fold cross-validation
k = 5;
cv = cvpartition(numSamples, 'KFold', k);

for paramIdx = 1:length(paramGrid)
    params = paramGrid(paramIdx);
    accuracies = zeros(k, 1);

    for fold = 1:k
        % Training and test indices for this fold
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        % Create training and test sets
        X_train = combinedROIsArray(trainIdx, :);
        Y_train = Y(trainIdx);
        X_test = combinedROIsArray(testIdx, :);
        Y_test = Y(testIdx);

        % Define the NN architecture
        layers = [
            featureInputLayer(rangeSize * dopplerSize)
            fullyConnectedLayer(params.layerSize1)
            reluLayer
            dropoutLayer(params.dropoutRate)
            fullyConnectedLayer(params.layerSize2)
            reluLayer
            dropoutLayer(params.dropoutRate)
            fullyConnectedLayer(numel(categories(Y)))
            softmaxLayer
            classificationLayer];

        % Set training options
        options = trainingOptions('adam', ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', 32, ...
            'ValidationData', {X_test, Y_test}, ...
            'ValidationFrequency', 30, ...
            'Verbose', false, ...
            'Plots', 'none');

        % Train the NN
        net = trainNetwork(X_train, Y_train, layers, options);

        % Evaluate the NN
        YPred = classify(net, X_test);
        accuracy = sum(YPred == Y_test) / numel(Y_test);
        disp(['Fold ', num2str(fold), ' - Classification Accuracy: ', num2str(accuracy)]);

        % Store accuracy
        accuracies(fold) = accuracy;
    end

    % Calculate mean accuracy for the current parameter set
    meanAccuracy = mean(accuracies);
    disp(['Parameter Set ', num2str(paramIdx), ' - Mean Classification Accuracy: ', num2str(meanAccuracy)]);

    % Update best model if the current one is better
    if meanAccuracy > bestAccuracy
        bestAccuracy = meanAccuracy;
        bestParams = params;
        bestNet = net;
    end
end

% Display best accuracy and parameters
disp(['Best Mean Classification Accuracy: ', num2str(bestAccuracy)]);
disp('Best Parameters:');
disp(bestParams);

% Save the best network based on accuracy
save('Data\Models\best_full_roi_nn_model.mat', 'bestNet', 'bestParams');

% Evaluate the best model on the full dataset for a final confusion matrix
Y_pred_full = classify(bestNet, combinedROIsArray);
accuracy_full = sum(Y_pred_full == Y) / numel(Y);
disp(['Final Classification Accuracy with Best Model: ', num2str(accuracy_full)]);

% Plot confusion matrix for the best model
figure;
confusionchart(Y, Y_pred_full);
title('Confusion Matrix');




