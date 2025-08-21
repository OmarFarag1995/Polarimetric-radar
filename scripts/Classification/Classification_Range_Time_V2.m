clear;
close all;

% Load the processed data containing meanRangeBinsSubset
load('Data/Processed_Data/processedDataStruct_RangeTime.mat', 'processedDataStruct');

% Mapping of datasets to classes
classMap = struct();
classMap.dry = {'dry_AsphaltRoad', 'dry_Asphalt_RampUp', 'dry_Surface', 'dry_aptivroad', 'dry_road4'};
classMap.snow = {'sno_wAptivparking', 'sno_wy_univparking', 'sno_wyuniversityroad'};
classMap.wet = {'wet_Asphaltroad2', 'wet_MultipleSurfaces', 'wet_parking_3'};

% Frame parameters
frameSize = 100; % Number of samples per frame
frameStep = 1;  % Step size to move the window

% Initialize cell arrays to store frames and labels
data = {};
labels = {};

% Loop through each class and dataset, organizing the data
classNames = fieldnames(classMap);
for classIdx = 1:numel(classNames)
    classLabel = classNames{classIdx};
    datasets = classMap.(classLabel);
    
    % Loop through each dataset in the class
    for i = 1:length(datasets)
        datasetName = datasets{i};
        
        % Extract the meanRangeBinsSubset data
        meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset;
        
        % Break the meanRangeBinsSubset into frames
        numSamples = length(meanRangeBinsSubset);
        for startIdx = 1:frameStep:(numSamples - frameSize + 1)
            endIdx = startIdx + frameSize - 1;
            frame = meanRangeBinsSubset(startIdx:endIdx);
            data{end+1} = frame;  % Store each frame
            labels{end+1} = classLabel; % Store the label for each frame
        end
    end
end

% Convert labels to categorical array
labels = categorical(labels);

% Split data into training (90%) and testing (10%) sets
numObservations = numel(data);
[idxTrain, idxTest] = dividerand(numObservations, 0.9, 0.1);
XTrain = data(idxTrain);
YTrain = labels(idxTrain);
XTest = data(idxTest);
YTest = labels(idxTest);

% Prepare Data for Padding: Sort training data by sequence length
sequenceLengths = cellfun(@(x) size(x,1), XTrain);
[sequenceLengths, idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);

% Define LSTM Neural Network Architecture
numHiddenUnits = 120;
numClasses = numel(categories(labels));

layers = [
    sequenceInputLayer(1) % One feature per time step
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last') % Use BiLSTM layer to learn from full sequence
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Specify Training Options
options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    InitialLearnRate=0.002, ...
    GradientThreshold=1, ...
    MiniBatchSize=27, ...
    SequenceLength="longest", ...
    Shuffle="never", ...
    Plots="training-progress", ...
    Verbose=false);

% Train LSTM Neural Network
net = trainNetwork(XTrain, YTrain, layers, options);

% Prepare and Sort Test Data for Padding
sequenceLengthsTest = cellfun(@(x) size(x,1), XTest);
[sequenceLengthsTest, idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);

% Test LSTM Neural Network
YPred = classify(net, XTest, ...
    MiniBatchSize=27, ...
    SequenceLength="longest");

% Calculate the classification accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Road Surface Classification');
