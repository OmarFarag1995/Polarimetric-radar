clc;
clear;
close all;

%% Load Processed Data
load('..\Data\Processed_Data\processedDataStruct.mat', 'processedDataStruct');



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


% Prompt the user to input which channels to process
channelsInput = input('Enter the channels to process (e.g., [1, 2, 3]): ');

% Ensure the input is a valid array of numbers
if isempty(channelsInput) || ~isnumeric(channelsInput)
    error('Invalid input. Please enter an array of numbers.');
end

% Define the features to extract
featuresToExtract = {'BackscatterIntensity', 'DopplerSpread', 'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity', 'SpectralEntropy', 'Skewness', 'Kurtosis'};

% Initialize the channelsToKeep structure with user-defined channels
channelsToKeep = repmat({channelsInput}, 1, length(featuresToExtract));

% Define the region of interest around the ego Doppler bin
dopplerWindowSize = 10;  % Number of Doppler bins on each side of the ego Doppler bin
rangeIndicesForAnalysis = 1:150;  % Adjust this range as needed

% Initialize cell arrays to store features and labels for all road types
allFeatures = {};
allLabels = {};

% Iterate over each road type in the processed data
fields = fieldnames(processedDataStruct);
for iField = 1:length(fields)
    classLabel = fields{iField};
    processedData = processedDataStruct.(classLabel);

    % Initialize feature arrays for the current road type
    numFiles = length(processedData);
    spectralFeatures = [];

    for iFile = 1:numFiles
        adcDataIn = processedData(iFile).ProcessedData;
        rdmOutput = adcDataIn.Data;  % Load the range-Doppler map data
        vehSpeed_mps = adcDataIn.VehSpeed_mps;  % Extract vehicle speed
        
        if ~isempty(rdmOutput)
            % Calculate ego Doppler bin based on vehicle speed
            egoDopplerBin = 64 + floor(vehSpeed_mps / 0.12);
            egoDopplerBin = mod(egoDopplerBin - 1, size(rdmOutput, 2)) + 1;  % Ensure it stays within bounds

            % Define the Doppler indices to analyze
            dopplerIndices = max(1, egoDopplerBin - dopplerWindowSize):min(size(rdmOutput, 2), egoDopplerBin + dopplerWindowSize);

            % Iterate over the selected channels to extract features
            for chIdx = 1:size(rdmOutput, 3)
                for txIdx = 1:size(rdmOutput, 4)
                    channelNumber = (txIdx - 1) * 4 + chIdx;

                    % Extract the region of interest (ROI)
                    rdMap = abs(rdmOutput(rangeIndicesForAnalysis, dopplerIndices, chIdx, txIdx));

                    % Initialize feature vector for this channel
                    features = [];

                    % Extract each feature based on the selected channels
                    for featureIdx = 1:length(featuresToExtract)
                        featureName = featuresToExtract{featureIdx};
                        if any(channelNumber == channelsToKeep{featureIdx})
                            switch featureName
                                case 'BackscatterIntensity'
                                    features = [features, mean(mean(rdMap))];
                                case 'DopplerSpread'
                                    dopplerProfile = mean(rdMap, 2);
                                    features = [features, std(dopplerProfile)];
                                case {'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity'}
                                    rdMapNormalized = mat2gray(rdMap); % Normalize to [0, 1]
                                    glcm = graycomatrix(rdMapNormalized, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                                    stats = graycoprops(glcm, featureName(6:end));
                                    features = [features, mean(stats.(featureName(6:end)))];
                                case 'SpectralEntropy'
                                    dopplerProfileFFT = abs(fft(mean(rdMap, 1)));
                                    probDoppler = dopplerProfileFFT / sum(dopplerProfileFFT); % Probability distribution
                                    features = [features, -sum(probDoppler .* log2(probDoppler + eps))];
                                case 'Skewness'
                                    dopplerProfile = mean(rdMap, 1);
                                    features = [features, skewness(dopplerProfile)];
                                case 'Kurtosis'
                                    dopplerProfile = mean(rdMap, 1);
                                    features = [features, kurtosis(dopplerProfile)];
                                % Add other cases as necessary...
                            end
                        end
                    end

                    % Append the extracted features to the spectral features array
                    spectralFeatures = [spectralFeatures; features];
                end
            end
        end
    end
    
    % Store features and labels for the current road type
    allFeatures = [allFeatures; spectralFeatures];
    allLabels = [allLabels; repmat({classLabel}, size(spectralFeatures, 1), 1)];
    
    fprintf('Processed road type: %s\n', classLabel);
end

% Combine all features and labels into matrices
combinedFeatures = cell2mat(allFeatures);
combinedLabels = categorical(allLabels);

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(combinedFeatures, 1));
combinedFeatures = combinedFeatures(shuffledIndices, :);
combinedLabels = combinedLabels(shuffledIndices, :);

% Save combined features and labels
save('..\Data\Feature_Vector_Data\Features_From_Full_ROI.mat', 'combinedFeatures', 'combinedLabels');

fprintf('Feature extraction complete. Data saved to Features_From_Full_ROI.mat\n');
