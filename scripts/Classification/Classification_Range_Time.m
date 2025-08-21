clear;
clc
close all;

% Load the processed data containing meanRangeBinsSubset
load('Data/Processed_Data/processedDataStruct_RangeTime.mat', 'processedDataStruct');

% Mapping of datasets to classes
classMap = struct();
classMap.dry = {'dry_AsphaltRoad', 'dry_Asphalt_RampUp', 'dry_Surface', 'dry_aptivroad', 'dry_road4'};
classMap.snow = {'sno_wAptivparking', 'sno_wy_univparking', 'sno_wyuniversityroad'};
classMap.wet = {'wet_Asphaltroad2', 'wet_MultipleSurfaces', 'wet_parking_3'};

% Frame parameters
frameSize = 500; % Number of samples per frame
frameStep = 250;  % Step size to move the window (controls overlap)

% Initialize cell arrays to store features and labels for training and testing sets
featuresTrain = {};
labelsTrain = {};
featuresTest = {};
labelsTest = {};

% Loop through each class and dataset, organizing the data
classNames = fieldnames(classMap);
for classIdx = 1:numel(classNames)
    classLabel = classNames{classIdx};
    datasets = classMap.(classLabel);
    
    % Shuffle the datasets for each class to randomize the split
    datasets = datasets(randperm(length(datasets)));
    
    % Define split index (80% training, 20% testing)
    splitIdx = round(0.8 * length(datasets));
    
    % Loop through each dataset in the class
    for i = 1:length(datasets)
        datasetName = datasets{i};
        
        % Extract the meanRangeBinsSubset data
        meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset;
        
        % Break the meanRangeBinsSubset into frames
        numSamples = length(meanRangeBinsSubset);
        frames = {};
        for startIdx = 1:frameStep:(numSamples - frameSize + 1)
            endIdx = startIdx + frameSize - 1;
            frames{end+1} = meanRangeBinsSubset(startIdx:endIdx);
        end
        
        % Assign frames to training or testing set based on dataset index
        if i <= splitIdx
            featuresTrain = [featuresTrain, frames]; % Store each frame in the training set
            labelsTrain = [labelsTrain, repmat({classLabel}, 1, numel(frames))]; % Repeat the class label for each frame
        else
            featuresTest = [featuresTest, frames]; % Store each frame in the testing set
            labelsTest = [labelsTest, repmat({classLabel}, 1, numel(frames))]; % Repeat the class label for each frame
        end
    end
end

% Convert labels to categorical arrays
labelsTrain = categorical(labelsTrain);
labelsTest = categorical(labelsTest);

% Define the GRU model architecture
numHiddenUnits = 150;
numClasses = numel(categories(labelsTrain));

layers = [
    sequenceInputLayer(1) % One feature per time step
    gruLayer(numHiddenUnits, 'OutputMode', 'last') % Output the last time step
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 50, ...
    'Shuffle', 'never', ...
    'Verbose', 0, ...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 0.01); % Adjusted learning rate


% Train the GRU model
net = trainNetwork(featuresTrain, labelsTrain, layers, options);

% Predict labels for test set
YPred = classify(net, featuresTest);


labelsTest = categorical(labelsTest, categories(YPred));

% Calculate accuracy
accuracy = sum(YPred == labelsTest) / numel(labelsTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Display a confusion matrix
figure;
confMat = confusionmat(labelsTest, YPred);
confusionchart(confMat, categories(labelsTest));
title('Confusion Matrix for Road Surface Classification using GRU');
