clear
close all

%% Load ADC Data
load('Data\ADC_Data\adcDataStruct.mat', 'adcDataStruct');

% Note
% ch1  = HV, TX1RX1 H1V1 ## Antenna 1 
% ch2  = VV, TX2RX1 V2V1 ## Antenna 1 
% ch3  = HV, TX3RX1 H3V1 ## Antenna 1&2 
% ch4  = VV, TX4RX1 V4V1 ## Antenna 1&2 
% ch5  = HH, TX1RX2 H1H2 ## Antenna 1 
% ch6  = VH, TX2RX2 V2H2 ## Antenna 1 
% ch7  = HH, TX3RX2 H3H2 ## Antenna 1&2 
% ch8  = VH, TX4RX2 V4H2 ## Antenna 1&2



% ch9  = HV, TX1RX3 H1V3 ## Antenna 1&2 
% ch10 = VV, TX2RX3 V2V3 ## Antenna 1&2 
% ch11 = HV, TX3RX3 H3V3 ## Antenna 2 
% ch12 = VV, TX4RX3 V4V3 ## Antenna 2 
% ch13 = HH, TX1RX4 H1H4 ## Antenna 1&2 
% ch14 = VH, TX2RX4 V2H4 ## Antenna 1&2 
% ch15 = HH, TX3RX4 H3H4 ## Antenna 2 
% ch16 = VH, TX4RX4 V4H4 ## Antenna 2



%% Parameters
nTXChannelsForAnalysis = 4;  % Number of TX channels for analysis
nRXChannelsForAnalysis = 4;  % Number of RX channels for analysis

% Initialize a structure to store the processed data
processedDataStruct = struct();

%% Process Data
fields = fieldnames(adcDataStruct);
for iField = 1:length(fields)
    classLabel = fields{iField};
    adcData = adcDataStruct.(classLabel);

    % Initialize storage for the processed data
    processedDataStruct.(classLabel) = [];

    for iFile = 1:length(adcData)
        ADCDataIn = adcData(iFile).ADCData.Data;  % Load the ADC data
        [nRangeBins, nDopplerBins, nChannels] = size(ADCDataIn);
        
        nRealDopplerBins = nDopplerBins/4;
        
        % Initialize storage for the processed data for the current file
        rangeDopplerOutput = zeros(nRangeBins, nRealDopplerBins, nRXChannelsForAnalysis, nTXChannelsForAnalysis);


        if ~isempty(ADCDataIn)
            % Extract the channel data
            channelData = double(ADCDataIn);
            
            % Range FFT
            hannWindowRange = (repmat(hanning(nRangeBins),1,nDopplerBins,nChannels));
            rangeFFT = fft(channelData .* hannWindowRange, [], 1) / nRangeBins;
            
            % rx,tx % rx combination are flipped
            rangeFffTxSeperated(:,:,:,1) = rangeFFT(:,1:4:512,:); % tx separation
            rangeFffTxSeperated(:,:,:,2) = rangeFFT(:,2:4:512,:); % tx separation
            rangeFffTxSeperated(:,:,:,3) = rangeFFT(:,3:4:512,:); % tx separation
            rangeFffTxSeperated(:,:,:,4) = rangeFFT(:,4:4:512,:); % tx separation
            
            
            % Doppler FFT
            hannWindowDoppler_ = repmat(hanning(128),1,nRangeBins,4,4);
            hannWindowDoppler = permute(hannWindowDoppler_ ,[2,1,3,4]);
            rangeDopplerFFT = fftshift(fft(rangeFffTxSeperated.* hannWindowDoppler, [], 2), 2) / nRealDopplerBins ;
            
       
            % Store the processed data for the channel
            rangeDopplerOutput= rangeDopplerFFT;
            rangeOutput= rangeFffTxSeperated;

             % Store the processed data for the current file
            processedDataStruct.(classLabel)(iFile).ProcessedData = adcData(iFile).ADCData;
            processedDataStruct.(classLabel)(iFile).ProcessedData.Data = rangeDopplerOutput;
            processedDataStruct.(classLabel)(iFile).ProcessedData.RangeFFT = rangeOutput;
            processedDataStruct.(classLabel)(iFile).fileName = adcData(iFile).fileName;
        end      
    end
end

%% Save the processed data
save('..\Data\Processed_Data\processedDataStruct.mat', 'processedDataStruct', '-v7.3');

fprintf('Processing the data complete. Data saved to processedDataStruct.mat\n');