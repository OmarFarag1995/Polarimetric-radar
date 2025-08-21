clear; close all; clc;

% 1. Paths to all processed validation LOOK files
valLookPaths = {
    'Data/Processed_Data/VAL_Range_Time_Look_A/processedDataStruct_Validation_RangeTime_Look_A.mat'
    'Data/Processed_Data/VAL_Range_Time_Look_B/processedDataStruct_Validation_RangeTime_Look_B.mat'
    'Data/Processed_Data/VAL_Range_Time_Look_C/processedDataStruct_Validation_RangeTime_Look_C.mat'
    'Data/Processed_Data/VAL_Range_Time_Look_D/processedDataStruct_Validation_RangeTime_Look_D.mat'
};

selectedChannel = 2; % Channel you used for training

% 2. Combine all validation looks
valData = {};
valLabels = {};
temp = load(valLookPaths{1}, 'processedDataStruct');
refFields = fieldnames(temp.processedDataStruct);

for iLook = 1:length(valLookPaths)
    S = load(valLookPaths{iLook}, 'processedDataStruct');
    processedDataStruct = S.processedDataStruct;
    for iField = 1:length(refFields)
        datasetName = refFields{iField};
        meanRangeBinsSubset = processedDataStruct.(datasetName).meanRangeBinsSubset;
        channelData = meanRangeBinsSubset(selectedChannel, :);
        valData{end+1} = channelData;
        if startsWith(datasetName, 'dry')
            labelType = 'dry';
        elseif startsWith(datasetName, 'sno')
            labelType = 'snow';
        elseif startsWith(datasetName, 'wet')
            labelType = 'wet';
        else
            continue;
        end
        valLabels{end+1} = labelType;
    end
end
valLabels = categorical(valLabels);

% 3. Expand labels for LSTM (per time step)
valLabelsExpanded = cellfun(@(x, y) repmat(y, 1, length(x)), valData, num2cell(valLabels), 'UniformOutput', false);

% 4. Load your trained model
load('Data/Models/LSTM_RoadSurface_AllLOOKs.mat', 'net');

% 5. Validate on new validation set
sequenceAccuracies = zeros(numel(valData), 1);
for i = 1:numel(valData)
    YPred = classify(net, valData{i});
    sequenceAccuracies(i) = mean(YPred == valLabelsExpanded{i});
end
overallValAccuracy = mean(sequenceAccuracies);
fprintf('Validation Set Accuracy (average per-sequence): %.2f%%\n', overallValAccuracy * 100);
