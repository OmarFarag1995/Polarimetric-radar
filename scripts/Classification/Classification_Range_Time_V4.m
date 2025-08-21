clear;
close all;
clc;

%% Load Sequence Data
% Load the processed data containing meanRangeBinsSubset
load('Data\Processed_Data\processedDataStruct_RangeTime_.mat', 'processedDataStruct');

% Initialize cell arrays to store data and labels
data = {};
labels = {};

% Get the dataset names from processedDataStruct
fields = fieldnames(processedDataStruct);

% Extract data and labels from processedDataStruct
for i = 1:length(fields)
    datasetName = fields{i};
    meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset; % Extract meanRangeBinsSubset
    
    % Add the meanRangeBinsSubset to the data cell array
    data{end+1} = meanRangeBinsSubset; % Each dataset stored as a row in the cell array
    
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
    
    % Repeat the label for each time step in the sequence
    labels{end+1} = categorical(repmat({labelType}, 1, size(meanRangeBinsSubset, 2)));
end

% Partition data into training and testing sets
cv = cvpartition(length(labels), 'HoldOut', 0.2);
XTrain = data(training(cv));
YTrain = labels(training(cv));
XTest = data(test(cv));
YTest = labels(test(cv));

%% Define LSTM Network Architecture
% Get number of features and classes
numFeatures = size(XTrain{1}, 1); % Assumes each sample has 4 channels/features
numHiddenUnits = 200;
numClasses = numel(categories(YTrain{1}));

% Define layers
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'GradientThreshold', 2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');
XTrain = reshape(XTrain,8,1);
YTrain = reshape(YTrain,8,1);
XTest = reshape(XTest,3,1);
YTest = reshape(YTest,3,1);
%% Train the LSTM Network
net = trainNetwork(XTrain, YTrain, layers, options);

%% Test the LSTM Network
% Classify each time step in the test set
YPred = classify(net, XTest);

% Calculate accuracy for each time step
accuracy = mean(cellfun(@(pred, label) mean(pred == label), YPred, YTest));
fprintf('Test Accuracy per time step: %.2f%%\n', accuracy * 100);

% Plot example sequence and its classification
seqIdx = 1; % Choose a sequence index to visualize
figure;
plot(XTest{seqIdx}')
xlabel("Time Step");
legend("Channel " + (1:numFeatures))
title("Test Data - Example Sequence");

% Plot predictions vs. actual labels for this example sequence
figure;
plot(YPred{seqIdx}, '.-');
hold on;
plot(YTest{seqIdx});
hold off;
xlabel("Time Step");
ylabel("Road Type");
title("Predicted vs Actual Road Type - Example Sequence");
legend(["Predicted" "Actual"]);

%% Confusion Matrix for Test Set
% Combine predictions and labels for all time steps
allPredictions = vertcat(YPred{:});
allLabels = vertcat(YTest{:});

% Display confusion matrix
figure;
confMat = confusionmat(allLabels, allPredictions);
confusionchart(confMat, categories(YTest{1}));
title('Confusion Matrix for Road Surface Classification');
