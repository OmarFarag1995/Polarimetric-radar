% Sequence Classification Using Deep Learning
% This example shows how to classify sequence data using a long short-term memory (LSTM) network.

% Load Sequence Data
% Load the example data from WaveformData.
load WaveformData

% Visualize some of the sequences in a plot.
numChannels = size(data{1},2);
idx = [3 4 5 12];
figure
tiledlayout(2,2)
for i = 1:4
    nexttile
    stackedplot(data{idx(i)}, 'DisplayLabels', "Channel " + string(1:numChannels))
    xlabel("Time Step")
    title("Class: " + string(labels(idx(i))))
end

% View the class names.
classNames = categories(labels);

% Set aside data for testing.
% Partition the data into a training set containing 90% of the data and a test set containing the remaining 10% of the data.
numObservations = numel(data);
[idxTrain, idxTest] = trainingPartitions(numObservations, [0.9 0.1]);
XTrain = data(idxTrain);
TTrain = labels(idxTrain);
XTest = data(idxTest);
TTest = labels(idxTest);

% Prepare Data for Padding
% Get the sequence lengths for each observation.
sequenceLengths = cellfun(@(x) size(x,1), XTrain);

% Sort the data by sequence length.
[sequenceLengths, idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
TTrain = TTrain(idx);

% View the sorted sequence lengths in a bar chart.
figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

% Define LSTM Neural Network Architecture
numHiddenUnits = 120;
numClasses = numel(classNames);

layers = [
    sequenceInputLayer(numChannels)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Specify Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 0.002, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train LSTM Neural Network
net = trainNetwork(XTrain, TTrain, layers, options);

% Test LSTM Neural Network
% Prepare the test data.
sequenceLengthsTest = cellfun(@(x) size(x,1), XTest);
[sequenceLengthsTest, idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
TTest = TTest(idx);

% Classify the test data.
YPred = classify(net, XTest, 'MiniBatchSize', 1);

% Calculate the classification accuracy.
acc = sum(YPred == TTest) / numel(TTest);
fprintf('Test Accuracy: %.2f%%\n', acc * 100);

% Display the classification results in a confusion chart.
figure
confusionchart(TTest, YPred)
