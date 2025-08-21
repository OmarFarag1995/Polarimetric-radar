clear;
close all;
clc;

% Paths to all processed LOOK files
lookPaths = {
    'Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Clean_LOOK_A.mat'
    'Data\Processed_Data\Range_Time_Clean_LOOK_B\processedDataStruct_RangeTime_Clean_LOOK_B.mat'
    'Data\Processed_Data\Range_Time_Clean_LOOK_C\processedDataStruct_RangeTime_Clean_LOOK_C.mat'
    'Data\Processed_Data\Range_Time_Clean_LOOK_D\processedDataStruct_RangeTime_Clean_LOOK_D.mat'
};

% Specify the channel to use for classification (1: HV, 2: VV, 3: HH, 4: VH)
selectedChannel = 2; % Change this as needed

% Create containers for combined data and labels
allData = {};
allLabels = {};

% We'll use the fields of the first look as reference
temp = load(lookPaths{1}, 'processedDataStruct');
refFields = fieldnames(temp.processedDataStruct);

% Loop over all looks and concatenate their data by dataset name
for iLook = 1:length(lookPaths)
    % Load this look
    S = load(lookPaths{iLook}, 'processedDataStruct');
    processedDataStruct = S.processedDataStruct;
    
    % Loop through each dataset/class (same across all LOOKs)
    for iField = 1:length(refFields)
        datasetName = refFields{iField};
        meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset;
        
        % Use the selected channel
        channelData = meanRangeBinsSubset(selectedChannel, :);
        allData{end+1} = channelData;
        
        % Class label by dataset name
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
        allLabels{end+1} = labelType;
    end
end

% Convert labels to categorical
allLabels = categorical(allLabels);

% --- Split train/test sets: Ensure each class is represented ---
rng(42); % For reproducibility

dryIndices = find(allLabels == 'dry');
snowIndices = find(allLabels == 'snow');
wetIndices = find(allLabels == 'wet');

% Shuffle within each class
dryIndices = dryIndices(randperm(numel(dryIndices)));
snowIndices = snowIndices(randperm(numel(snowIndices)));
wetIndices = wetIndices(randperm(numel(wetIndices)));

testFraction = 0.3;
numTestDry = max(1, round(testFraction * numel(dryIndices)));
numTestSnow = max(1, round(testFraction * numel(snowIndices)));
numTestWet = max(1, round(testFraction * numel(wetIndices)));

% Ensure column vectors before concatenation
testIndices = [
    dryIndices(1:numTestDry)';
    snowIndices(1:numTestSnow)';
    wetIndices(1:numTestWet)';
];

trainIndices = setdiff(1:numel(allLabels), testIndices);

% Prepare train/test data
XTrain = allData(trainIndices);
YTrain = allLabels(trainIndices);
XTest  = allData(testIndices);
YTest  = allLabels(testIndices);

% --- For LSTM: Each sequence can have a label vector matching its length ---
YTrainExpanded = cellfun(@(x, y) repmat(y, 1, length(x)), XTrain, num2cell(YTrain), 'UniformOutput', false);
YTestExpanded  = cellfun(@(x, y) repmat(y, 1, length(x)), XTest, num2cell(YTest), 'UniformOutput', false);

% --- Define LSTM Model ---
numFeatures = 1; % Single channel
numHiddenUnits = 50;
numClasses = 3;

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 60, ...
    'GradientThreshold', 2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% --- Train ---
net = trainNetwork(XTrain, YTrainExpanded, layers, options);

% --- Evaluate ---
sequenceAccuracies = zeros(numel(XTest), 1);

for i = 1:numel(XTest)
    YPred = classify(net, XTest{i});
    sequenceAccuracies(i) = mean(YPred == YTestExpanded{i});
end

overallAccuracy = mean(sequenceAccuracies);
fprintf('Overall Test Accuracy (average per-sequence): %.2f%%\n', overallAccuracy * 100);



% --- Save the trained model and splits for future use ---
modelFilename = 'Data/Models/LSTM_RoadSurface_AllLOOKs.mat';

% Create the folder if it doesn't exist
[modelFolder,~,~] = fileparts(modelFilename);
if ~exist(modelFolder, 'dir')
    mkdir(modelFolder);
end

save(modelFilename, 'net', 'XTrain', 'YTrainExpanded', 'XTest', 'YTestExpanded', 'selectedChannel', '-v7.3');
fprintf('Trained model saved to %s\n', modelFilename);