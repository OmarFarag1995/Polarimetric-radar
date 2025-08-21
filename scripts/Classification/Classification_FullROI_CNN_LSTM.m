clc;
clear;
close all;

%% Load the extracted ROIs and labels
load('Data\Feature_Vector_Data\full_roi_doppler_CNN_LSTM_sequential.mat', 'allROIs', 'allLabels');

% Combine all categories (dry, wet, snow)
categories = fieldnames(allROIs);
combinedROIs = [];
combinedLabels = [];

for i = 1:length(categories)
    category = categories{i};
    combinedROIs = [combinedROIs; allROIs.(category)'];
    combinedLabels = [combinedLabels; allLabels.(category)'];
end

% Convert labels to categorical
Y = categorical(combinedLabels);

% Determine the size of the ROIs
numSequences = length(combinedROIs);
sequenceLength = length(combinedROIs{1});
[rangeSize, dopplerSize] = size(combinedROIs{1}{1});
numChannels = 1; % Since each ROI is a 2D matrix, treated as a single-channel input

% Convert the cell array to a 4D array for CNN-LSTM input [range, doppler, channels*sequenceLength, samples]
combinedROIsArray = zeros(rangeSize, dopplerSize, numChannels * sequenceLength, numSequences);
for i = 1:numSequences
    combinedROIsArray(:, :, :, i) = reshape(cat(3, combinedROIs{i}{:}), [rangeSize, dopplerSize, numChannels * sequenceLength]);
end

%% Define the number of folds for cross-validation
kFolds = 5;
cv = cvpartition(numSequences, 'KFold', kFolds);
foldAccuracies = zeros(kFolds, 1);

%% Cross-validation loop
for fold = 1:kFolds
    % Get the training and test indices for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    % Create training and testing sets for this fold
    X_train = combinedROIsArray(:, :, :, trainIdx);
    Y_train = Y(trainIdx);
    X_test = combinedROIsArray(:, :, :, testIdx);
    Y_test = Y(testIdx);
    
    %% Determine the number of classes
    numClasses = 3;  % This gives you the correct number of classes
    
    %% Define the CNN-LSTM architecture
    layers = [
        imageInputLayer([rangeSize, dopplerSize, numChannels * sequenceLength], 'Name', 'input')
        
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'batchnorm1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
        
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'batchnorm2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'batchnorm3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
        
        flattenLayer('Name', 'flatten') % Flatten the CNN output
        
        fullyConnectedLayer(50, 'Name', 'fc1') % Reduce dimensionality to match LSTM input size
        lstmLayer(50, 'OutputMode', 'last', 'Name', 'lstm') % LSTM to process temporal information
        
        fullyConnectedLayer(numClasses, 'Name', 'fc2') % Number of units = number of classes
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    %% Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {X_test, Y_test}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'none');  % Turn off visualization

    %% Train the CNN-LSTM
    net = trainNetwork(X_train, Y_train, layers, options);

    %% Evaluate the CNN-LSTM on the test data
    YPred = classify(net, X_test);
    accuracy = sum(YPred == Y_test) / numel(Y_test);
    foldAccuracies(fold) = accuracy;
    disp(['Fold ', num2str(fold), ' Classification Accuracy: ', num2str(accuracy * 100), '%']);
end

%% Calculate and display the average accuracy across all folds
averageAccuracy = mean(foldAccuracies);
disp(['Average Classification Accuracy across all folds: ', num2str(averageAccuracy * 100), '%']);

%% Save the last trained model (or you can save the model from the best fold)
save('Data\Models\cnn_lstm_model_kfold.mat', 'net');
fprintf('CNN-LSTM classification complete. Model saved to cnn_lstm_model_kfold.mat\n');
