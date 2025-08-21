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
numChannels = 1; % Since each ROI is a 2D matrix, treated as a single-channel input

% Convert the cell array to a 4D array for CNN input [range, doppler, channels, samples]
combinedROIsArray = zeros(rangeSize, dopplerSize, numChannels, numSamples);
for i = 1:numSamples
    combinedROIsArray(:, :, 1, i) = allROIs{i};
end

% Define the parameter grid
paramGrid = [
    struct('filterSize1', 3, 'numFilters1', 8, 'filterSize2', 3, 'numFilters2', 16, 'filterSize3', 3, 'numFilters3', 32, 'dropoutRate', 0.3);
    struct('filterSize1', 3, 'numFilters1', 12, 'filterSize2', 3, 'numFilters2', 24, 'filterSize3', 3, 'numFilters3', 48, 'dropoutRate', 0.4);
    struct('filterSize1', 5, 'numFilters1', 8, 'filterSize2', 5, 'numFilters2', 16, 'filterSize3', 5, 'numFilters3', 32, 'dropoutRate', 0.3);
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
        X_train = combinedROIsArray(:, :, :, trainIdx);
        Y_train = Y(trainIdx);
        X_test = combinedROIsArray(:, :, :, testIdx);
        Y_test = Y(testIdx);

        % Define the CNN architecture
        layers = [
            imageInputLayer([rangeSize, dopplerSize, numChannels])
            convolution2dLayer(params.filterSize1, params.numFilters1, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2, 'Stride', 2)
            convolution2dLayer(params.filterSize2, params.numFilters2, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2, 'Stride', 2)
            convolution2dLayer(params.filterSize3, params.numFilters3, 'Padding', 'same')
            batchNormalizationLayer
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
        
        % Train the CNN
        net = trainNetwork(X_train, Y_train, layers, options);
        
        % Evaluate the CNN
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
save('Data\Models\best_full_roi_cnn_model.mat', 'bestNet', 'bestParams');

% Evaluate the best model on the full dataset for a final confusion matrix
Y_pred_full = classify(bestNet, combinedROIsArray);
accuracy_full = sum(Y_pred_full == Y) / numel(Y);
disp(['Final Classification Accuracy with Best Model: ', num2str(accuracy_full)]);

% Plot confusion matrix for the best model
figure;
confusionchart(Y, Y_pred_full);
title('Confusion Matrix');