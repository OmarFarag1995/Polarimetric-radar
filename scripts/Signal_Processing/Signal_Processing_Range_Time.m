clear;
close all;

%% Prompt the user to input range bin selection
startBin = input('Enter the starting range bin (e.g., 0): ');
endBin = input('Enter the ending range bin (e.g., 50): ');

% Validate the input
if startBin < 0 || endBin <= startBin
    error('Invalid range bin selection. Please enter a valid start and end bin.');
end

% Display chosen range for confirmation
fprintf('Using range bins from %d to %d.\n', startBin, endBin);

%% Load ADC Data
load('Data\ADC_Data\adcDataStruct_Seq_Model.mat', 'adcDataStruct');

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
channelsToProcess = [6]; % Channels for Antenna 1 (e.g., TX2RX1, VV)

% Initialize a structure to store the processed data (Range-Time Maps)
processedDataStruct = struct();

%% Process Data to Generate Range-Time Maps
fields = fieldnames(adcDataStruct);
for iField = 1:length(fields)
    classLabel = fields{iField};
    adcData = adcDataStruct.(classLabel);

    % Initialize storage for the processed data
    processedDataStruct.(classLabel) = [];

    % Initialize an empty matrix to accumulate range-time data
    accumulatedRangeTimeMap = [];

    for iFile = 1:length(adcData)
        % Load the ADC data
        ADCDataIn = adcData(iFile).ADCData.Data;  % Assuming the raw data is stored in the 'Data' field
        
        if isempty(ADCDataIn)
            warning('ADC data is empty for %s. Skipping...', adcData(iFile).fileName);
            continue;
        end
        
        [nRangeBins, nDopplerBins, nChannels] = size(ADCDataIn);
        
        % Extract the channel data
        channelData = double(ADCDataIn);
        
        % Range FFT
        hannWindowRange = repmat(hanning(nRangeBins), 1, nDopplerBins, nChannels);
        rangeFFT = fft(channelData .* hannWindowRange, [], 1) / nRangeBins;
        
        % Separate TX channels
        rangeFffTxSeperated = zeros(nRangeBins, nDopplerBins / 4, nChannels, nTXChannelsForAnalysis);
        rangeFffTxSeperated(:,:,:,1) = rangeFFT(:,1:4:end,:); % TX1 separation
        rangeFffTxSeperated(:,:,:,2) = rangeFFT(:,2:4:end,:); % TX2 separation
        rangeFffTxSeperated(:,:,:,3) = rangeFFT(:,3:4:end,:); % TX3 separation
        rangeFffTxSeperated(:,:,:,4) = rangeFFT(:,4:4:end,:); % TX4 separation

        % Accumulate selected channels for Range-Time Map visualization
        accumulatedChannels = [];
        for chIdx = channelsToProcess
            % Extract data for selected channel
            rangeTimeData = abs(rangeFffTxSeperated(:,:,chIdx));
            accumulatedChannels = cat(3, accumulatedChannels, rangeTimeData);
        end
        
        % Concatenate data from selected channels along the time axis
        concatenatedRangeTimeMap = reshape(permute(accumulatedChannels, [1, 3, 2]), nRangeBins, []);
        % Accumulate Range-Time data over all frames for visualization
        accumulatedRangeTimeMap = cat(2, accumulatedRangeTimeMap, concatenatedRangeTimeMap);

    end

    % Save accumulated range-time map
    processedDataStruct.(classLabel).AccumulatedRangeTimeMap = accumulatedRangeTimeMap;
    
    % Save user-selected range bins subset for sequence model
    processedDataStruct.(classLabel).RangeBinsSubset = accumulatedRangeTimeMap(startBin+1:endBin, :); % Using user-defined range

    % Calculate mean across the subset of selected range bins
    meanRangeBinsSubset = mean(processedDataStruct.(classLabel).RangeBinsSubset, 1);



    processedDataStruct.(classLabel).meanRangeBinsSubset = meanRangeBinsSubset;



    % Calculate the overall average of the meanRangeBinsSubset
    overallMean = mean(meanRangeBinsSubset);

    % % Visualization of the mean of the selected Range Bins over time, with overall average
    % figure;
    % plot(meanRangeBinsSubset, 'DisplayName', sprintf('Mean of Range Bins %d-%d', startBin, endBin));
    % hold on;
    % yline(overallMean, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Overall Mean = %.2f', overallMean));
    % xlabel('Time (Accumulated Slow Time Indices)');
    % ylabel('Mean Amplitude (dB)');
    % ylim([0 15]); % Scale y-axis from 0 to 15 dB
    % title(sprintf('Mean of Range Bins %d-%d Over Time for %s (All)', startBin, endBin, classLabel));
    % legend show;
    % grid on;
    % drawnow;

    % % Save the mean plot in the specified folder
    % outputFolder = 'Data\Processed_Data\Range_Time_Maps';
    % if ~exist(outputFolder, 'dir')
    %     mkdir(outputFolder);
    % end
    % saveas(gcf, fullfile(outputFolder, sprintf('MeanRangeBinsSubset_%s_%d-%d.png', classLabel, startBin, endBin)));

    % Visualization of the accumulated Range-Time Map (Range vs. Time)
    if ~isempty(accumulatedRangeTimeMap)
        figure;
        imagesc(1:size(accumulatedRangeTimeMap, 2), 1:nRangeBins, 20*log10(accumulatedRangeTimeMap));
        colorbar;
        xlabel('Time (Accumulated Slow Time Indices)');
        ylabel('Range Bins');
        title(sprintf('Accumulated Range-Time Map for %s (VH)', classLabel));
        set(gca, 'YDir', 'normal'); % Set y-axis to start from bottom
        drawnow;

        % % Save the figure in the specified folder
        % saveas(gcf, fullfile(outputFolder, sprintf('RangeTimeMap_%s.png', classLabel)));
    end
end
% close all
% 
% %% Save the processed range-time map data
% save('Data\Processed_Data\processedDataStruct_RangeTime.mat', 'processedDataStruct', '-v7.3');
% 
% fprintf('Processing complete. Range-time maps saved to processedDataStruct_RangeTime.mat and images saved to Range_Time_Maps folder.\n');
