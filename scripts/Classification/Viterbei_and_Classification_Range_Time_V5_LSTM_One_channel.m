clear;
close all;
clc;

%% Load Sequence Data
load('Data\Processed_Data\processedDataStruct_RangeTime_.mat', 'processedDataStruct');

% Select the radar channel (1: HV, 2: VV, 3: HH, 4: VH)
selectedChannel = 2;

% Initialize data and labels
data = {};
labels = {};

% Extract datasets
fields = fieldnames(processedDataStruct);
for i = 1:length(fields)
    datasetName = fields{i};
    meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset;
    
    % Extract selected channel data
    channelData = meanRangeBinsSubset(selectedChannel, :);
    data{end+1} = channelData;
    
    % Assign labels based on dataset name
    if startsWith(datasetName, 'dry')
        labelType = 'dry';
    elseif startsWith(datasetName, 'sno')
        labelType = 'snow';
    elseif startsWith(datasetName, 'wet')
        labelType = 'wet';
    else
        warning('Unknown class label for dataset %s. Skipping...', datasetName);
        continue;
    end
    labels{end+1} = labelType;
end

% Convert labels to categorical
labels = categorical(labels);

% Split into train & test sets
testFraction = 0.3;
dryIndices = find(labels == 'dry');
snowIndices = find(labels == 'snow');
wetIndices = find(labels == 'wet');

% Ensure at least 2 samples per class
minClassSize = min([numel(dryIndices), numel(snowIndices), numel(wetIndices)]);
numTestDry = min(3, minClassSize);
numTestSnow = min(3, minClassSize);
numTestWet = min(3, minClassSize);

% Random selection of test indices
testDryIndices = randsample(dryIndices, numTestDry);
testSnowIndices = randsample(snowIndices, numTestSnow);
testWetIndices = randsample(wetIndices, numTestWet);

% Concatenate all test indices
testIndices = [testDryIndices; testSnowIndices; testWetIndices];

% Ensure test indices are unique
testIndices = unique(testIndices);

% Remaining indices for training
trainIndices = setdiff(1:numel(labels), testIndices);

XTrain = data(trainIndices);
YTrain = labels(trainIndices);
XTest = data(testIndices);
YTest = labels(testIndices);

% Expand labels to match sequence length
YTrainExpanded = expandLabels(XTrain, YTrain);
YTestExpanded = expandLabels(XTest, YTest);

%% LSTM Model Configuration
numFeatures = 1;
numHiddenUnits = 50;
numClasses = 3;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'GradientThreshold', 2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train the LSTM network
net = trainNetwork(XTrain, YTrainExpanded, layers, options);

%% Extract Softmax Probabilities from LSTM
allSoftmaxScores = {};
allPredictions = {};
allLabels = {};

for i = 1:numel(XTest)
    [YPred, scores] = classify(net, XTest{i});  % Extract softmax scores
    allPredictions{i} = YPred;
    allLabels{i} = YTestExpanded{i};
    
    % Ensure scores have correct shape [numTimeSteps x numClasses]
    if size(scores, 2) == 1
        scores = reshape(scores, [], numel(categories(YTrain)));
    end
    allSoftmaxScores{i} = scores;
end

%% Apply Viterbi Algorithm for Post-Processing
% Define State Names & Transition Probabilities
stateNames = {'dry', 'wet', 'snow'};
A = [0.7, 0.2, 0.1;  % Dry → (Dry, Wet, Snow)
     0.3, 0.5, 0.2;  % Wet → (Dry, Wet, Snow)
     0.2, 0.4, 0.4]; % Snow → (Dry, Wet, Snow)

smoothedPredictions = {};
for i = 1:numel(XTest)
    emissionProbs = allSoftmaxScores{i};
    smoothedPredictions{i} = viterbiDecode(emissionProbs, A, stateNames);
end

% Display Example Results
disp('Original LSTM Prediction:');
disp(allPredictions{1});
disp('Viterbi Smoothed Prediction:');
disp(smoothedPredictions{1});

%% Helper Functions

% Expand categorical labels for sequence training
function expandedLabels = expandLabels(XData, YData)
    expandedLabels = {};
    for i = 1:length(XData)
        numTimeSteps = size(XData{i}, 2);
        label = YData(i);
        expandedLabels{i} = repmat(label, 1, numTimeSteps);
    end
end

% Viterbi Algorithm Implementation
function predictedPath = viterbiDecode(emissionProbs, transitionMatrix, states)
    numStates = size(transitionMatrix, 1);
    numTimeSteps = size(emissionProbs, 1);

    % Ensure emissionProbs has correct shape
    if size(emissionProbs, 2) ~= numStates
        error('Mismatch: emissionProbs should be [numTimeSteps x numStates].');
    end

    % Initialize Viterbi table
    V = zeros(numStates, numTimeSteps);
    path = zeros(numStates, numTimeSteps);

    % Initialize first step
    V(:, 1) = emissionProbs(1, 1:numStates)';

    % Forward pass
    for t = 2:numTimeSteps
        for s = 1:numStates
            [maxVal, maxState] = max(V(:, t-1) .* transitionMatrix(:, s));
            V(s, t) = maxVal * emissionProbs(t, s);
            path(s, t) = maxState;
        end
    end

    % Backtracking
    [~, finalState] = max(V(:, end));
    predictedPath = zeros(1, numTimeSteps);
    predictedPath(end) = finalState;

    for t = numTimeSteps-1:-1:1
        predictedPath(t) = path(predictedPath(t+1), t+1);
    end

    predictedPath = states(predictedPath);
end
