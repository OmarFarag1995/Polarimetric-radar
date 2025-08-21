%% Road Type Classification Using LSTM (Sequence-to-Label)
clear; close all; clc;

%% === 1. Setup and Configuration ===
% Define constants
selectedChannel = 2;       % Channel to use (1:HV, 2:VV, 3:HH, 4:VH)
segmentLength   = 250;     % Length of each small recording chunk
testFraction    = 0.2;     % 30% of data for testing

%% === 2. Load and Chunk Data into Segments ===
load('Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Road_Types_LOOK_A.mat',...
     'processedDataStruct');

allSegments = {};
allLabels   = {};
fields = fieldnames(processedDataStruct);

fprintf('Chunking data into %d-step segments...\n', segmentLength);

for i = 1:length(fields)
    datasetName = fields{i};
    
    % Get the full signal for the selected channel
    fullSignal = processedDataStruct.(datasetName).meanRangeBinsSubset(selectedChannel, :);
    
    % Get the label from the filename (e.g., "Asphalt")
    parts = split(datasetName, '_');
    label = parts{1};
    
    % Calculate how many full segments we can create
    numSegments = floor(length(fullSignal) / segmentLength);
    
    % Loop and create the small segments
    for j = 1:numSegments
        startIndex = (j-1) * segmentLength + 1;
        endIndex   = j * segmentLength;
        
        segment = fullSignal(startIndex:endIndex);
        
        % Add the segment and its label to our lists
        allSegments{end+1, 1} = segment;
        allLabels{end+1, 1}   = label;
    end
end

fprintf('Created %d total segments.\n', numel(allSegments));

% Convert labels to a categorical type for the network
Y = categorical(allLabels);
X = allSegments;

%% === 3. Create Training and Test Sets ===
% Use cvpartition for a stratified split. This ensures both train and
% test sets have a fair representation of each road type.
cv = cvpartition(Y, 'HoldOut', testFraction, 'Stratify', true);

% Assign data to training and testing sets
XTrain = X(cv.training);
YTrain = Y(cv.training);
XTest  = X(cv.test);
YTest  = Y(cv.test);

fprintf('Training samples: %d\n', numel(YTrain));
fprintf('Test samples: %d\n\n', numel(YTest));

%% === 4. Define LSTM Network Architecture ===
% This is a Sequence-to-Label architecture
numFeatures = 1; % We selected one channel
numHiddenUnits = 100;
numClasses = numel(categories(Y));

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last') % 'last' is key for Seq-to-Label
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ... % Shuffle data each epoch for better training
    'Verbose', false, ...
    'Plots', 'training-progress');

%% === 5. Train the LSTM Network ===
fprintf('Training the network...\n');
net = trainNetwork(XTrain, YTrain, layers, options);
fprintf('Training complete.\n');

%% === 6. Evaluate the Network ===
fprintf('Evaluating the network on the test set...\n');

% Classify the entire test set at once
YPred = classify(net, XTest);

% Calculate overall accuracy
accuracy = mean(YPred == YTest);
fprintf('Overall Test Accuracy: %.2f%%\n', accuracy * 100);

% Display a confusion matrix to see performance per class
figure;
confusionchart(YTest, YPred);
title('Road Type Classification - Confusion Chart');