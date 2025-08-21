%% Road Type Classification Using LSTM (Sequence-to-Sequence)
clear; close all; clc;

%% === Load Data ===
load('Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Road_Types_LOOK_A.mat',...
     'processedDataStruct');

% Define channel to use (1: HV, 2: VV, 3: HH, 4: VH)
selectedChannel = 2;

% Extract sequences and labels
data = {};
labels = {};
fields = fieldnames(processedDataStruct);
for i = 1:length(fields)
    datasetName = fields{i};
    signal = processedDataStruct.(datasetName).meanRangeBinsSubset;

    % Extract selected channel (1D sequence)
    data{end+1} = signal(selectedChannel, :);

    % Infer label from dataset name (e.g., "Asphalt_1")
    parts = split(datasetName, '_');
    label = parts{1};
    labels{end+1} = label;
end

% Convert labels to categorical
labels = categorical(labels);

% Group indices by class
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

%% === Expand Labels for Each Time Step ===

% Expand training labels
YTrainExpanded = {};
for i = 1:length(XTrain)
    numTimeSteps = size(XTrain{i}, 2); % Number of time steps
    label = YTrain(i); % Sequence-level label
    YTrainExpanded{i} = repmat(label, 1, numTimeSteps); % Repeat per time step
end
YTrain = YTrainExpanded;
YTrain = reshape(YTrain, length(YTrain), 1); % Ensure column cell array

% Expand test labels
YTestExpanded = {};
for i = 1:length(XTest)
    numTimeSteps = size(XTest{i}, 2);
    label = YTest(i);
    YTestExpanded{i} = repmat(label, 1, numTimeSteps);
end
YTest = YTestExpanded;
YTest = reshape(YTest, length(YTest), 1);


%% === Define LSTM Network ===
numFeatures = 1;
numHiddenUnits = 50;
numClasses = numel(classNames);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'GradientThreshold', 2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% === Train LSTM Network ===
net = trainNetwork(XTrain, YTrain, layers, options);

%% === Evaluate on Test Set ===
sequenceAccuracies = zeros(numel(XTest), 1);
for i = 1:numel(XTest)
    YPred = classify(net, XTest{i});
    sequenceAccuracies(i) = mean(YPred == YTest{i});
end
overallAccuracy = mean(sequenceAccuracies);
fprintf('Overall Test Accuracy (average per-sequence): %.2f%%\n', overallAccuracy * 100);
