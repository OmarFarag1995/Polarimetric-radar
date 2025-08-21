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
cv = cvpartition(length(labels), 'HoldOut', 0.3);
XTrain = data(training(cv));
YTrain = labels(training(cv));
XTest = data(test(cv));
YTest = labels(test(cv));



XTrain = reshape(XTrain,8,1);
YTrain = reshape(YTrain,8,1);
XTest = reshape(XTest,3,1);
YTest = reshape(YTest,3,1);


numFeatures = 4;
numHiddenUnits = 25;
numClasses = 3;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];




options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'never', ...
    'Verbose', false, ...
    'InitialLearnRate', 0.01); 


% , ...
%     'Plots', 'training-progress'



net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest{3});

acc = sum(YPred == YTest{3})./numel(YTest{3});


fprintf('Test Accuracy (from Confusion Matrix): %.2f%%\n', acc * 100);

%% Test the LSTM Network
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
% 
% %% Confusion Matrix for Test Set
% % Collect all predictions and labels to the same length for confusion matrix calculation
% maxLength = max(cellfun(@length, allPredictions));
% paddedPredictions = categorical(nan(numel(XTest), maxLength));
% paddedLabels = categorical(nan(numel(XTest), maxLength));
% 
% for i = 1:numel(XTest)
%     len = length(allPredictions{i});
%     paddedPredictions(i, 1:len) = allPredictions{i};
%     paddedLabels(i, 1:len) = allLabels{i};
% end
% 
% % Remove NaN entries and compute confusion matrix
% validIdx = paddedPredictions;
% allPredictionsFlat = paddedPredictions(validIdx);
% allLabelsFlat = paddedLabels(validIdx);
% 
% % Display confusion matrix
% figure;
% confMat = confusionmat(allLabelsFlat, allPredictionsFlat);
% confusionchart(confMat, categories(YTest{1}));
% title('Confusion Matrix for Road Surface Classification using LSTM');