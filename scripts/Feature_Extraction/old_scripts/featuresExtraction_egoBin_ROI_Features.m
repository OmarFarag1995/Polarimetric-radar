clc;
clear;
close all;

% Define the features to extract
featuresToExtract = {'BackscatterIntensity', 'DopplerSpread', 'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity', 'SpectralEntropy', 'Skewness', 'Kurtosis'};

% Define the channels to keep for each feature
channelsToKeep = {
    1:4,        % BackscatterIntensity
    1:4,        % DopplerSpread
    1:4,        % GLCM Contrast
    1:4,        % GLCM Correlation
    1:4,        % GLCM Energy 
    1:4,        % GLCM Homogeneity
    1:4,        % SpectralEntropy 
    1:4,        % Skewness 
    1:4         % Kurtosis 
};

% Check if the length of channelsToKeep matches the length of featuresToExtract
if length(channelsToKeep) ~= length(featuresToExtract)
    error('Length of channelsToKeep must match length of featuresToExtract.');
end

% Parent directory containing all road type folders
parentDir = 'C:\Users\xjz6mk\Documents\work data\road_surface_monitoring\data\Road_types';

% Get all subdirectories (road types)
roadTypes = dir(parentDir);
roadTypes = roadTypes([roadTypes.isdir]);  % Keep only directories
roadTypes = roadTypes(~ismember({roadTypes.name}, {'.', '..'}));  % Remove . and ..

% Initialize cell arrays to store features and labels for all road types
allFeatures = cell(length(roadTypes), 1);
allLabels = cell(length(roadTypes), 1);

% Define the region of interest around the ego Doppler bin
dopplerWindowSize = 3;  % Number of Doppler bins on each side of the ego Doppler bin
rangeIndicesForAnalysis = 1:300;  % Adjust this range as needed

for roadTypeIdx = 1:length(roadTypes)
    roadType = roadTypes(roadTypeIdx).name;
    dataLocation = fullfile(parentDir, roadType);
    
    % Read the radar ADC data
    radarDataLocation = fullfile(dataLocation, 'Radar_Fwd');
    radarOutputFrames = dir(fullfile(radarDataLocation, '*.mat')); % Assuming .mat files

    % Initialize feature arrays
    numFrames = length(radarOutputFrames);
    numFeatures = length(featuresToExtract);
    extractedFeatures = zeros(numFrames, 4, numFeatures);
    labels = cell(numFrames, 1);

    for iRadarFrame = 1:numFrames
        adcDataInput = load(fullfile(radarDataLocation, radarOutputFrames(iRadarFrame).name));
        adcDataPerFrame = adcDataInput.Look_A.Data; % Only checking LOOK A
        lookParam = adcDataInput.Look_A;
        
        if ~isempty(adcDataPerFrame)
            rdmOutput = generateRangeDoppler(adcDataPerFrame); % Get the RDM with windowing
            
            % Calculate ego Doppler bin
            if lookParam.VehSpeed_mps < 64 * 0.12
                ego_doppler_bin = 64 + floor(lookParam.VehSpeed_mps / 0.12);
            else
                ego_doppler_bin = rem(64 + floor(lookParam.VehSpeed_mps / 0.12), 128);
            end
            
            % Define region of interest
            dopplerIndices = max(1, ego_doppler_bin - dopplerWindowSize):min(size(rdmOutput, 2), ego_doppler_bin + dopplerWindowSize);
            
            for iRX = 1:2
                for iTX = 1:2
                    channel_index = 2 * (iRX - 1) + iTX;
                    
                    % Extract focused range-Doppler map for this channel
                    rdMap = squeeze(rdmOutput(rangeIndicesForAnalysis, dopplerIndices, iRX, iTX));
                    
                    if ~isempty(rdMap)
                        % Extract and store selected features
                        for featureIdx = 1:numFeatures
                            featureName = featuresToExtract{featureIdx};
                            switch featureName
                                case 'BackscatterIntensity'
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = mean(mean(abs(rdMap)));
                                case 'DopplerSpread'
                                    dopplerProfile = mean(abs(rdMap), 1);
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = std(dopplerProfile);
                                case 'GLCM_Contrast'
                                    rdMapNormalized = mat2gray(abs(rdMap)); % Normalize to [0, 1]
                                    glcm = graycomatrix(rdMapNormalized, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                                    stats = graycoprops(glcm, {'Contrast'});
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = mean(stats.Contrast); 
                                case 'GLCM_Correlation'
                                    rdMapNormalized = mat2gray(abs(rdMap)); % Normalize to [0, 1]
                                    glcm = graycomatrix(rdMapNormalized, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                                    stats = graycoprops(glcm, {'Correlation'});
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = mean(stats.Correlation); 
                                case 'GLCM_Energy'
                                    rdMapNormalized = mat2gray(abs(rdMap)); % Normalize to [0, 1]
                                    glcm = graycomatrix(rdMapNormalized, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                                    stats = graycoprops(glcm, {'Energy'});
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = mean(stats.Energy); 
                                case 'GLCM_Homogeneity'
                                    rdMapNormalized = mat2gray(abs(rdMap)); % Normalize to [0, 1]
                                    glcm = graycomatrix(rdMapNormalized, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                                    stats = graycoprops(glcm, {'Homogeneity'});
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = mean(stats.Homogeneity); 
                                case 'SpectralEntropy'
                                    dopplerProfileFFT = abs(fft(mean(abs(rdMap), 1)));
                                    probDoppler = dopplerProfileFFT / sum(dopplerProfileFFT); % Probability distribution
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = -sum(probDoppler .* log2(probDoppler + eps));
                                case 'Skewness'
                                    dopplerProfile = mean(abs(rdMap), 1);
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = skewness(dopplerProfile);
                                case 'Kurtosis'
                                    dopplerProfile = mean(abs(rdMap), 1);
                                    extractedFeatures(iRadarFrame, channel_index, featureIdx) = kurtosis(dopplerProfile);
                            end
                        end
                    end
                end
            end
            
            labels{iRadarFrame} = roadType; % Store the label for the frame
        end
    end
    
    % Remove empty cells
    nonEmptyIdx = cellfun(@(x) ~isempty(x), labels);
    extractedFeatures = extractedFeatures(nonEmptyIdx, :, :);
    labels = labels(nonEmptyIdx);
    
    % Select the features to keep for each channel
    finalFeatures = [];
    for featureIdx = 1:numFeatures
        channels = channelsToKeep{featureIdx};
        if any(channels > 4)
            error('Channel index exceeds the number of available channels.');
        end
        finalFeatures = [finalFeatures, extractedFeatures(:, channels, featureIdx)];
    end
    
    % Store selected features and labels
    allFeatures{roadTypeIdx} = finalFeatures;
    allLabels{roadTypeIdx} = labels;
    
    fprintf('Processed road type: %s\n', roadType);
end
% Combine all features and labels
combinedFeatures = vertcat(allFeatures{:});
combinedLabels = vertcat(allLabels{:});

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(combinedFeatures, 1));
combinedFeatures = combinedFeatures(shuffledIndices, :);
combinedLabels = combinedLabels(shuffledIndices, :);

% Save combined features and labels
save('enhanced_road_types_features.mat', 'combinedFeatures', 'combinedLabels');

fprintf('Feature extraction complete. Data saved to enhanced_road_types_features.mat\n');
