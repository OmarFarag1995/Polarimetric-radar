% clear;
close all;

%% Load Processed Data
load('Data\Processed_Data\processedDataStruct.mat', 'processedDataStruct');

% Prompt the user to input which channels to process
channelsToExtract = input('Enter the channels to extract the ROI from (e.g., [1, 2, 3]): ');

% Ensure the input is a valid array of numbers
if isempty(channelsToExtract) || ~isnumeric(channelsToExtract)
    error('Invalid input. Please enter an array of numbers.');
end

% Define the range indices to be analyzed (10:435)
rangeIndicesForAnalysis = 1:150;  % MATLAB indexing starts from 1

% Define the number of Doppler bins around the ego Doppler bin to extract
dopplerROIWidth = 10;  % Number of bins to include on each side of the ego Doppler bin

% Initialize cell arrays to store ROIs and labels for all road types
allROIs = {};
allLabels = {};

% Note
% ch1  = HV, TX1RX1 H1V1 ## Antenna 1 
% ch2  = VV, TX2RX1 V2V1 ## Antenna 1 
% ch3  = HV, TX3RX1 H3V1 ## Antenna 1&2 
% ch4  = VV, TX4RX1 V4V1 ## Antenna 1&2 
% ch5  = HH, TX1RX2 H1H2 ## Antenna 1 
% ch6  = VH, TX2RX2 V2H2 ## Antenna 1 
% ch7  = HH, TX3RX2 H3H2 ## Antenna 1&2 
% ch8  = VH, TX4RX2 V4H2 ## Antenna 1&2

% ch9  = HV, TX1RX3 H1V3 ## Antenna 1&2 
% ch10 = VV, TX2RX3 V2V3 ## Antenna 1&2 
% ch11 = HV, TX3RX3 H3V3 ## Antenna 2 
% ch12 = VV, TX4RX3 V4V3 ## Antenna 2 
% ch13 = HH, TX1RX4 H1H4 ## Antenna 1&2 
% ch14 = VH, TX2RX4 V2H4 ## Antenna 1&2 
% ch15 = HH, TX3RX4 H3H4 ## Antenna 2 
% ch16 = VH, TX4RX4 V4H4 ## Antenna 2

% Iterate over each road type in the processed data
fields = fieldnames(processedDataStruct);
for iField = 1:length(fields)
    classLabel = fields{iField};
    processedData = processedDataStruct.(classLabel);

    % Initialize storage for ROIs for the current road type
    numFiles = length(processedData);
    roisForRoadType = {};

    for iFile = 1:numFiles
        adcDataIn = processedData(iFile).ProcessedData;
        rdmOutput = adcDataIn.Data;  % Load the range-Doppler map data
        [nRangeBins, nDopplerBins, ~] = size(rdmOutput);
        vehSpeed_mps = adcDataIn.VehSpeed_mps;  % Extract vehicle speed
        
        if ~isempty(rdmOutput)
            % Calculate ego Doppler bin based on vehicle speed
            egoDopplerBin = 64 + floor(vehSpeed_mps / 0.12);
            egoDopplerBin = mod(egoDopplerBin - 1, nDopplerBins) + 1;  % Ensure it stays within bounds

            % Determine the Doppler bins to extract (ROI)
            dopplerStart = egoDopplerBin - dopplerROIWidth;
            dopplerEnd = egoDopplerBin + dopplerROIWidth;
            
            % Handle wrapping of Doppler bins around the edges
            if dopplerStart < 1
                dopplerIndicesForAnalysis = [dopplerStart + nDopplerBins : nDopplerBins, 1 : dopplerEnd];
            elseif dopplerEnd > nDopplerBins
                dopplerIndicesForAnalysis = [dopplerStart : nDopplerBins, 1 : dopplerEnd - nDopplerBins];
            else
                dopplerIndicesForAnalysis = dopplerStart:dopplerEnd;
            end

            % Extract ROI for the specified channels
            for ch = channelsToExtract
                % Extract the ROI as 2D data
                roi = abs(rdmOutput(rangeIndicesForAnalysis, dopplerIndicesForAnalysis, ch));

                % Store the ROI data
                roisForRoadType{end+1} = roi;  
            end
        end
    end
    
    % Store ROIs and labels for the current road type
    allROIs = [allROIs; roisForRoadType'];
    allLabels = [allLabels; repmat({classLabel}, length(roisForRoadType), 1)];
    
    fprintf('Processed road type: %s\n', classLabel);
end

% Save combined ROIs and labels
save('Data\Feature_Vector_Data\full_roi_doppler.mat', 'allROIs', 'allLabels');

fprintf('ROI extraction complete. Data saved to roi_doppler_features.mat\n');
