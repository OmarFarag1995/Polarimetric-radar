clear;
close all;
clc;

%% Load Sequence Data
% Load the processed data containing meanRangeBinsSubset
load('Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Clean_LOOK_A.mat', 'processedDataStruct');

% Specify the channel to use for classification (1: HV, 2: VV, 3: HH, 4: VH)
selectedChannel = 2; % Change this to select a different channel

% Initialize cell arrays to store data and labels
data = {};
labels = {};

% Get the dataset names from processedDataStruct
fields = fieldnames(processedDataStruct);

% Extract data and labels from processedDataStruct
for i = 1:length(fields)
    datasetName = fields{i};
    meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset; % Extract meanRangeBinsSubset
    
    % Extract the data for the selected channel
    channelData = meanRangeBinsSubset(selectedChannel, :);
    
    % Add the channel data to the data cell array
    data{end+1} = channelData; % Each dataset stored as a row in the cell array
    
    % Determine the label based on dataset name
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
    
    % Add the label to the labels array
    labels{end+1} = labelType;
end

% Convert labels to categorical
labels = categorical(labels);

% Group indices by class
dryIndices = find(labels == 'dry');
snowIndices = find(labels == 'snow');
wetIndices = find(labels == 'wet');

% Ensure at least one sequence from each class is in the test set
testFraction = 0.3; % Fraction of data to use for testing
numTestDry = max(1, round(testFraction * numel(dryIndices)));
numTestSnow = max(1, round(testFraction * numel(snowIndices)));
numTestWet = max(1, round(testFraction * numel(wetIndices)));

% Combine test indices from each class
testDryIndices = dryIndices(1:round(0.3 * numel(dryIndices)));
testSnowIndices = snowIndices(1:round(0.3 * numel(snowIndices)));
testWetIndices = wetIndices(1:round(0.3 * numel(wetIndices)));

% Ensure column vectors
testDryIndices = testDryIndices(:);
testSnowIndices = testSnowIndices(:);
testWetIndices = testWetIndices(:);

% Find the minimum length
minLength = min([numel(testDryIndices), numel(testSnowIndices), numel(testWetIndices)]);

% Limit each class to `minLength`
testDryIndices = testDryIndices(1:minLength);
testSnowIndices = testSnowIndices(1:minLength);
testWetIndices = testWetIndices(1:minLength);

% Concatenate indices
testIndices = [testDryIndices; testSnowIndices; testWetIndices];

% Get remaining indices for training
trainIndices = setdiff(1:numel(labels), testIndices);

% Create training and testing sets
XTrain = data(trainIndices);
YTrain = labels(trainIndices);
XTest = data(testIndices);
YTest = labels(testIndices);

% Reshape for compatibility
XTrain = reshape(XTrain, length(XTrain), 1);
YTrain = reshape(YTrain, length(YTrain), 1);
XTest = reshape(XTest, length(XTest), 1);
YTest = reshape(YTest, length(YTest), 1);

% Label each time sample in XTrain
TrainExpanded = {};
for i = 1:length(XTrain)
    numTimeSteps = size(XTrain{i}, 2); % Get the number of time steps for this sequence
    label = YTrain(i); % Extract the label
    YTrainExpanded{i} = repmat(label, 1, numTimeSteps); % Replicate the label for each time step
end
YTrain = YTrainExpanded; % Replace YTrain with the expanded version

YTrain = reshape(YTrain,17, 1);


% Label each time sample in XTest
YTestExpanded = {};
for i = 1:length(XTest)
    numTimeSteps = size(XTest{i}, 2); % Get the number of time steps for this sequence
    label = YTest(i); % Extract the label
    YTestExpanded{i} = repmat(label, 1, numTimeSteps); % Replicate the label for each time step
end
YTest = YTestExpanded; % Replace YTest with the expanded version

YTest = reshape(YTest,6, 1);


% Get input size (number of features is 1 because we are using one channel)
numFeatures = 1;
numHiddenUnits = 50;
numClasses = 3;

% Define the LSTM architecture
layers = [ ...
    sequenceInputLayer(numFeatures) % Single feature (channel)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence') % Output at each time step
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Define training options
% options = trainingOptions('adam', ...
%     'MaxEpochs', 100, ...
%     'MiniBatchSize', 4, ...
%     'Shuffle', 'never', ...
%     'Verbose', false, ...
%     'InitialLearnRate', 0.01);


options = trainingOptions('adam', ...
    'MaxEpochs',60, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train the LSTM network
net = trainNetwork(XTrain, YTrain, layers, options);

% % Test the network on the test set
% YPred = classify(net, XTest{2});
% 
% % Calculate accuracy for the first sequence
% acc = sum(YPred == YTest{2}) ./ numel(YTest{2});
% fprintf('Test Accuracy for First Sequence: %.2f%%\n', acc * 100);

%% Test the LSTM Network on all sequences
% Calculate accuracy for each sequence individually
sequenceAccuracies = zeros(numel(XTest), 1);
allPredictions = {};
allLabels = {};

for i = 1:numel(XTest)
    YPred = classify(net, XTest{i});
    allPredictions{i} = YPred;
    allLabels{i} = YTest{i};

    % Calculate accuracy for this sequence
    sequenceAccuracies(i) = mean(YPred == YTest{i});
end

% Calculate overall average accuracy
overallAccuracy = mean(sequenceAccuracies);
fprintf('Overall Test Accuracy (average per-sequence): %.2f%%\n', overallAccuracy * 100);






