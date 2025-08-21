%% Road Type Classification Using LSTM (Sequence-to-Sequence)
clear; close all; clc;

%% === 1. Configuration ===
% Define which channels to use (1:HV, 2:VV, 3:HH, 4:VH)
% Example for one channel: channelsToUse = 2;
% Example for two channels: channelsToUse = [2, 3];
% Example for all channels: channelsToUse = 1:4;
channelsToUse = 1:4; % <-- EDIT THIS LINE TO SELECT YOUR CHANNELS

%% === 2. Load and Prepare Data ===
load('Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Road_Types_LOOK_A.mat',...
     'processedDataStruct');

% Extract sequences and labels
data = {};
labels = {};
fields = fieldnames(processedDataStruct);
for i = 1:length(fields)
    datasetName = fields{i};
    signal = processedDataStruct.(datasetName).meanRangeBinsSubset;
    
    % --- MODIFIED LINE ---
    % Extract the selected channel(s). This now works for one or multiple channels.
    data{end+1} = signal(channelsToUse, :); 
    
    % Infer label from dataset name (e.g., "Asphalt_1")
    parts = split(datasetName, '_');
    label = parts{1};
    labels{end+1} = label;
end

% Convert labels to categorical
labels = categorical(labels);

% Group indices by class to create a balanced test set
classNames = categories(labels);
classIndices = cellfun(@(c) find(labels == c), classNames, 'UniformOutput', false);

% Determine balanced test set size
testFraction = 0.3;
numTestPerClass = min(cellfun(@(idx) max(1, round(testFraction * numel(idx))), classIndices));
testIndices = cell2mat(cellfun(@(idx) idx(1:numTestPerClass), classIndices, 'UniformOutput', false));
trainIndices = setdiff(1:numel(labels), testIndices);

% Prepare train/test sets
XTrain = data(trainIndices);
YTrain = labels(trainIndices);
XTest  = data(testIndices);
YTest  = labels(testIndices);

% Reshape for compatibility
XTrain = reshape(XTrain, [], 1);
YTrain = reshape(YTrain, [], 1);
XTest  = reshape(XTest, [], 1);
YTest  = reshape(YTest, [], 1);

%% === 3. Expand Labels for Each Time Step (Sequence-to-Sequence logic) ===
% This part remains the same as you requested.
YTrainExpanded = {};
for i = 1:length(XTrain)
    numTimeSteps = size(XTrain{i}, 2); % Number of time steps
    label = YTrain(i); % Sequence-level label
    YTrainExpanded{i} = repmat(label, 1, numTimeSteps); % Repeat per time step
end
YTrain = YTrainExpanded;
YTrain = reshape(YTrain, length(YTrain), 1); % Ensure column cell array

YTestExpanded = {};
for i = 1:length(XTest)
    numTimeSteps = size(XTest{i}, 2);
    label = YTest(i);
    YTestExpanded{i} = repmat(label, 1, numTimeSteps);
end
YTest = YTestExpanded;
YTest = reshape(YTest, length(YTest), 1);

%% === 4. Define LSTM Network ===
% --- MODIFIED LINE ---
% Number of features is now the number of channels you selected
numFeatures = numel(channelsToUse); 
numHiddenUnits = 50;
numClasses = numel(classNames);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'GradientThreshold', 16, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% === 5. Train LSTM Network ===
net = trainNetwork(XTrain, YTrain, layers, options);

%% === 6. Evaluate on Test Set ===
sequenceAccuracies = zeros(numel(XTest), 1);
for i = 1:numel(XTest)
    YPred = classify(net, XTest{i});
    sequenceAccuracies(i) = mean(YPred == YTest{i});
end
overallAccuracy = mean(sequenceAccuracies);
fprintf('Overall Test Accuracy (average per-sequence): %.2f%%\n', overallAccuracy * 100);