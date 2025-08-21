clear;
close all;

%% Prompt the user to input range bin selection
startBin = input('Enter the starting range bin (e.g., 0): ');
endBin = input('Enter the ending range bin (e.g., 50): ');
selectedBin = input('Enter the specific range bin you want to extract (e.g., 20): ');

% Validate the input
if startBin < 0 || endBin <= startBin || selectedBin <= 0 || selectedBin > endBin
    error('Invalid range bin selection. Please enter valid start, end, and specific bin values.');
end

% Display chosen range for confirmation
fprintf('Using range bins from %d to %d, and extracting specific bin %d.\n', startBin, endBin, selectedBin);

%% Load ADC Data
load('Data\ADC_Data\LOOK_A\adcDataStruct_Seq_Model_RoadTypes.mat', 'adcDataStruct');

%% Parameters for Signal Processing
nTXChannelsForAnalysis = 4;  % Number of TX channels for analysis
nRXChannelsForAnalysis = 4;  % Number of RX channels for analysis


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



% Channels to process (User selectable)
channelsToProcess = [1, 2, 5, 6]; % Channels for Antenna 1 (e.g., TX1RX1, TX2RX1, etc.)
processedDataStruct = struct();

%% Process Data to Generate Range-Time Maps
fields = fieldnames(adcDataStruct);
for iField = 1:length(fields)
    datasets = fields{iField};
    adcData = adcDataStruct.(datasets);

    % Initialize storage for the processed data
    processedDataStruct.(datasets) = struct();
    accumulatedRangeTimeMap = [];

    for iFile = 1:length(adcData)
        % Load the ADC data
        ADCDataIn = adcData(iFile).ADCData.Data;  % Assuming the raw data is stored in the 'Data' field
        
        if isempty(ADCDataIn)
            warning('ADC data is empty for %s. Skipping...', adcData(iFile).fileName);
            continue;
        end
        
        [nRangeBins, nDopplerBins, nChannels] = size(ADCDataIn);
        channelData = double(ADCDataIn);  % Extract the channel data
        
        % Range FFT
        hannWindowRange = repmat(hanning(nRangeBins), 1, nDopplerBins, nChannels);
        rangeFFT = fft(channelData .* hannWindowRange, [], 1) / nRangeBins;

        % Separate TX channels
        rangeFffTxSeperated = zeros(nRangeBins, nDopplerBins / 4, nChannels, nTXChannelsForAnalysis);
        rangeFffTxSeperated(:,:,:,1) = rangeFFT(:,1:4:end,:); % TX1 separation
        rangeFffTxSeperated(:,:,:,2) = rangeFFT(:,2:4:end,:); % TX2 separation
        rangeFffTxSeperated(:,:,:,3) = rangeFFT(:,3:4:end,:); % TX3 separation
        rangeFffTxSeperated(:,:,:,4) = rangeFFT(:,4:4:end,:); % TX4 separation

        % Accumulate selected channels as separate 1xM rows
        accumulatedChannels = [];
        for chIdx = channelsToProcess
            % Extract data for each selected channel separately
            rangeTimeData = abs(rangeFffTxSeperated(:,:,chIdx));
            accumulatedChannels = cat(3, accumulatedChannels, rangeTimeData);
        end
        
        % Reshape to 4xM format where each row corresponds to a channel
        concatenatedRangeTimeMap = permute(accumulatedChannels, [3, 2, 1]); % 4xM format
        accumulatedRangeTimeMap = cat(2, accumulatedRangeTimeMap, concatenatedRangeTimeMap);

    end

    % Save accumulated range-time map as 4xM
    processedDataStruct.(datasets).AccumulatedRangeTimeMap = accumulatedRangeTimeMap;
    
    % Save user-selected range bins subset for sequence model as 4xM
    processedDataStruct.(datasets).RangeBinsSubset = accumulatedRangeTimeMap(:,:, startBin+1:endBin);

    % Calculate mean across the subset of selected range bins for each channel
    meanRangeBinsSubset = mean(processedDataStruct.(datasets).RangeBinsSubset, 3);
    processedDataStruct.(datasets).meanRangeBinsSubset = meanRangeBinsSubset;

    % Extract the user-specified range bin over time for each channel
    if endBin >= selectedBin
        rangeBinSpecific = accumulatedRangeTimeMap(:,:, selectedBin); % 4x1 vector
        processedDataStruct.(datasets).RangeBinSpecific = rangeBinSpecific;
    else
        warning('End range bin is less than the selected bin; unable to extract the specified bin.');
        processedDataStruct.(datasets).RangeBinSpecific = NaN(4, 1);
    end



end
close all

%% Save the processed range-time map data
save('Data\Processed_Data\Range_Time_Clean_LOOK_A\processedDataStruct_RangeTime_Road_Types_LOOK_A.mat', 'processedDataStruct', '-v7.3');

fprintf('Processing complete. Range-time maps saved to processedDataStruct_RangeTime.mat and images saved to Range_Time_Maps folder.\n');




