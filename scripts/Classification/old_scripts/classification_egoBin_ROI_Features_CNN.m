clc;
clear;
close all;

% Load the extracted features
load('enhanced_road_types_features_2.mat');

% Define feature names (make sure this matches the actual feature layout in combinedFeatures)
featuresToExtract = {'BackscatterIntensity', 'DopplerSpread', 'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity', 'SpectralEntropy', 'Skewness', 'Kurtosis', 'LBP', 'Gabor', 'FractalDimension', 'Autocorrelation'};

% User Interface to select features
[featureIdx, tf] = listdlg('PromptString', 'Select features for training:', ...
                           'ListString', featuresToExtract, ...
                           'SelectionMode', 'multiple');
if ~tf
    error('No features selected.');
end

selectedFeatures = featureIdx;

% User Interface to select channels
channels = 1:4;
[channelIdx, tf] = listdlg('PromptString', 'Select channels for training:', ...
                           'ListString', arrayfun(@num2str, channels, 'UniformOutput', false), ...
                           'SelectionMode', 'multiple');
if ~tf
    error('No channels selected.');
end

selectedChannels = channelIdx;

% Assuming combinedFeatures is organized as (samples, features * channels)
num_samples = size(combinedFeatures, 1);
num_channels = 4; % Assuming 4 channels
num_features = length(featuresToExtract);

% Verify the combinedFeatures size matches expected dimensions
if size(combinedFeatures, 2) ~= num_channels * num_features
    error('The size of combinedFeatures does not match the expected dimensions.');
end

% Extract selected features and channels
X = [];
for i = selectedFeatures
    for j = selectedChannels
        feature_start_idx = (i - 1) * num_channels + j;
        X = [X, combinedFeatures(:, feature_start_idx)];
    end
end

% Convert labels to categorical
Y = categorical(combinedLabels);

% Number of folds for cross-validation
k = 5;

% Initialize arrays to store accuracies and networks
accuracies = zeros(k, 1);
networks = cell(k, 1);

% Create cross-validation partition
cv = cvpartition(size(X, 1), 'KFold', k);

for fold = 1:k
    % Training and test indices for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Create training and test sets
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx);

    % Define the network architecture
    layers = [
        featureInputLayer(size(X_train, 2))
        fullyConnectedLayer(128) % Increased size for more complex features
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.4) % Increased dropout to prevent overfitting
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.4)
        fullyConnectedLayer(numel(categories(Y)))
        softmaxLayer
        classificationLayer];

    % Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {X_test, Y_test}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'none'); % Disable training plots

    % Train the network
    net = trainNetwork(X_train, Y_train, layers, options);
    
    % Evaluate the network
    Y_pred = classify(net, X_test);
    accuracy = sum(Y_pred == Y_test) / numel(Y_test);
    disp(['Fold ', num2str(fold), ' - Classification Accuracy: ', num2str(accuracy)]);
    
    % Store accuracy and network
    accuracies(fold) = accuracy;
    networks{fold} = net;
end

% Display max accuracy across all folds
maxAccuracy = max(accuracies);
disp(['Max validation Accuracy: ', num2str(maxAccuracy)]);

% Display mean accuracy across all folds
meanAccuracy = mean(accuracies);
disp(['Mean validation Accuracy: ', num2str(meanAccuracy)]);

% Save the best network based on accuracy
[~, bestFoldIdx] = max(accuracies);
bestNet = networks{bestFoldIdx};
save('best_cv_cnn_model_2.mat', 'bestNet');

% Evaluate the best model on the full dataset for a final confusion matrix
Y_pred_full = classify(bestNet, X);
accuracy_full = sum(Y_pred_full == Y) / numel(Y);
disp(['Final Classification Accuracy with Best Model: ', num2str(accuracy_full)]);

% Plot confusion matrix for the best model
figure;
confusionchart(Y, Y_pred_full);
title('Confusion Matrix - Best CV-CNN Classification');

% Feature importance analysis (using a simple method)
feature_importance = zeros(1, size(X, 2));
for i = 1:size(X, 2)
    X_temp = X;
    X_temp(:, i) = mean(X(:, i)); % Replace feature with its mean
    Y_pred_temp = classify(bestNet, X_temp);
    accuracy_temp = sum(Y_pred_temp == Y) / numel(Y);
    feature_importance(i) = accuracy_full - accuracy_temp;
end 

% Plot feature importance
figure;
bar(feature_importance);
xlabel('Feature Index');
ylabel('Importance Score');
title('Feature Importance');
xticks(1:numel(feature_importance));
xtickangle(90);
