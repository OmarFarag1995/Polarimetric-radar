clear;
close all;

%% Load Processed Data
load('..\Data\Processed_Data\processedDataStruct.mat', 'processedDataStruct');

% Define the range indices to be analyzed (use the entire range)
rangeIndicesForAnalysis = 12:30;  % MATLAB indexing starts from 1

% Initialize cell arrays to store features and labels for all road types
allFeatures = {};
allLabels = {};

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


% Channels to extract features from
% Channnes name    [HV,VV,HH,VH]                        
antennaChannels = [1, 2, 5, 6];

% Feature extraction flags
extractVVHV = true;
extractHHVV = true;
extractMeanDopplerSpread_VV = true;
extractMeanPower_VV = true;

% Iterate over each road type in the processed data
fields = fieldnames(processedDataStruct);
for iField = 1:length(fields)
    classLabel = fields{iField};
    processedData = processedDataStruct.(classLabel);

    % Initialize feature array for current road type
    numFiles = length(processedData);
    PolarimetricFeatures = [];

    for iFile = 1:numFiles
        adcDataIn = processedData(iFile).ProcessedData;
        rdmOutput = adcDataIn.Data;  % Load the range-Doppler map data
        [nRangeBins, nDopplerBins, nChannels] = size(rdmOutput);
        vehSpeed_mps = adcDataIn.VehSpeed_mps;  % Extract vehicle speed
        nAntenna = nChannels/8; % 2 antennas are used for recordings 
        nChannelsPerAntenna = nChannels/4; % 2TX and 2RX for each antenna
        
        if ~isempty(rdmOutput)
            % Calculate ego Doppler bin based on vehicle speed
            egoDopplerBin = 64 + floor(vehSpeed_mps / 0.12);
            egoDopplerBin = mod(egoDopplerBin - 1, nDopplerBins) + 1;  % Ensure it stays within bounds

            % Extract features for channels corresponding to Antenna 1
            for ch = antennaChannels

                % indexing manually, this had to be done manually
                iTX = floor(ch/nChannelsPerAntenna)+1;
                iRX = mod(ch,nChannelsPerAntenna);

                % Extract the Doppler spectrum at the ego Doppler bin
                % RD peak could be anywhere in between egoDopplerBin +-2 bins
                % select +-10 doppler bins around ego bin
                egoDopplerBinShifted = egoDopplerBin;
                dopplerSelect = egoDopplerBinShifted-10:1:egoDopplerBinShifted+10;
                
                if ~(egoDopplerBin+2>128 || egoDopplerBin-2<1)
                    maxDopplerSpectrum(iTX,iRX,:) = max(abs(rdmOutput(rangeIndicesForAnalysis,egoDopplerBin-2:egoDopplerBin+2,iTX,iRX)),[],2); % take the peak value of the rd matrix
                else
                    maxDopplerSpectrum(iTX,iRX,:)= abs(rdmOutput(rangeIndicesForAnalysis,egoDopplerBin,iTX,iRX)); % take the peak value of the rd matrix 
                end

                % store the doppler fft output 
                dopplerOut= squeeze(abs(rdmOutput(rangeIndicesForAnalysis,:,iTX,iRX)));

                 % if no indices are abbove threshold then set NaN
                perChannelDopplerParam(rangeIndicesForAnalysis,iTX,iRX,1) = NaN;
                perChannelDopplerParam(rangeIndicesForAnalysis,iTX,iRX,2) = NaN;

                % check if dopplerSelect are at edge of doppler
                % spectrum
                if ~(any(dopplerSelect<1) || any(dopplerSelect>128))
                
                    % reselect the doppler bins
                    dopplerSelect = egoDopplerBinShifted-10:1:egoDopplerBinShifted+10;
    
                    % apply a mask on the non-select doppler bins
                    dopplerOut_mask = ones(size(dopplerOut))*1e-5;
                    dopplerOut_mask(:,dopplerSelect) = 1;
                    dopplerOut = dopplerOut.*dopplerOut_mask;
             
                    % doppler Out in db 
                    dopplerOut_dB = mag2db(dopplerOut);
                    
                    % simple CFAR with 8dB threshold
                    for iRangeidx = 1:length(rangeIndicesForAnalysis)
                        thresholdPerRangeBin((iRangeidx)) = squeeze(max(mean(dopplerOut_dB((iRangeidx),dopplerSelect),2)+8,-100));
                        [val,ind]=find(dopplerOut_dB((iRangeidx),:)>thresholdPerRangeBin((iRangeidx))*ones(1,size(dopplerOut_dB,2)));
                         if ~isempty(ind)
                            perChannelDopplerParam(rangeIndicesForAnalysis(iRangeidx),iTX,iRX,1) = [max(ind)-min(ind)];
                            perChannelDopplerParam(rangeIndicesForAnalysis(iRangeidx),iTX,iRX,2) = squeeze(max(dopplerOut_dB((iRangeidx),ind)));
                           
                        end
                    end
                end

            end
            % Spectral Features Extraction
            freqs = (1:size(maxDopplerSpectrum))'; % Frequency vector

            features = []; % Initialize the features vector

            if extractVVHV
                % 1. VV-HV in dB 
                responseVVHV_dB = mean(mag2db(maxDopplerSpectrum(1,2,:))-mag2db(maxDopplerSpectrum(1,1,:)),3);
                features = [features, responseVVHV_dB];
            end

            if extractHHVV
                % 2. HH/VV in linear 
                responseHHVV = mean((maxDopplerSpectrum(2,1,:))./(maxDopplerSpectrum(1,2,:)),3);
                features = [features, responseHHVV];
            end

            if extractMeanDopplerSpread_VV
                % 3. Mean Doppler spread across range for VV
                MeanDopplerWidth_VV = squeeze(mean(perChannelDopplerParam(:,1,2,1),1,'omitnan'));
                features = [features, MeanDopplerWidth_VV];
            end

            if extractMeanPower_VV
                % 3. Mean Power across range for VV
                MeanDopplerPower_VV = squeeze(mean(perChannelDopplerParam(:,1,2,2),1,'omitnan'));
                features = [features, MeanDopplerPower_VV];
            end

            % Append the features for this channel
            PolarimetricFeatures = [PolarimetricFeatures; features];
            
        end
    end
    
    % Store features and labels for the current road type
    allFeatures = [allFeatures; PolarimetricFeatures];
    allLabels = [allLabels; repmat({classLabel}, size(PolarimetricFeatures, 1), 1)];
    
    fprintf('Processed road type: %s\n', classLabel);
end

% Combine all features and labels into matrices
combinedFeatures = cell2mat(allFeatures);

% Convert labels to categorical array
combinedLabels = categorical(allLabels);

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(combinedFeatures, 1));
combinedFeatures = combinedFeatures(shuffledIndices, :);
combinedLabels = combinedLabels(shuffledIndices, :);

% plot the data 
figure
subplot(1,2,1)
gscatter(combinedFeatures(:,1),combinedFeatures(:,2),combinedLabels)
xlabel('Feature 1 (VV-HV dB)')
ylabel('Feature 2 (HH/VV)')
grid on
title('Feature plot')
subplot(1,2,2)
gscatter(combinedFeatures(:,3),combinedFeatures(:,4),combinedLabels)
xlabel('Feature 3 (Mean doppler Spread - VV)')
ylabel('Feature 4 (Mean Power - VV)')
grid on
title('Feature plot')

% Save combined features and labels
save('..\Data\Feature_Vector_Data\polarimteric_features.mat', 'combinedFeatures', 'combinedLabels');

fprintf('Feature extraction complete. Data saved to polarimteric_features.mat\n');
