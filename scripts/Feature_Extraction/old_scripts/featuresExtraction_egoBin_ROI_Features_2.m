clc;
clear;
close all;

% Define the features to extract
featuresToExtract = {'BackscatterIntensity', 'DopplerSpread', 'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity', 'SpectralEntropy', 'Skewness', 'Kurtosis', 'LBP', 'Gabor', 'FractalDimension', 'Autocorrelation'};


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



% Define the channels to keep for each feature
channelsToKeep = {
    [1, 2, 5, 6],        % BackscatterIntensity
    [1, 2, 5, 6],        % DopplerSpread
    [1, 2, 5, 6],        % GLCM Contrast
    [1, 2, 5, 6],        % GLCM Correlation
    [1, 2, 5, 6],        % GLCM Energy 
    [1, 2, 5, 6],        % GLCM Homogeneity
    [1, 2, 5, 6],        % SpectralEntropy 
    [1, 2, 5, 6],        % Skewness
    [1, 2, 5, 6],        % Kurtosis 
    [1, 2, 5, 6],        % LBP
    [1, 2, 5, 6],        % Gabor
    [1, 2, 5, 6],        % Fractal Dimension
    [1, 2, 5, 6]         % Autocorrelation
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
dopplerWindowSize = 5;  % Number of Doppler bins on each side of the ego Doppler bin
rangeIndicesForAnalysis = 1:150;  % Adjust this range as needed

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
save('enhanced_road_types_features_2.mat', 'combinedFeatures', 'combinedLabels');

fprintf('Feature extraction complete. Data saved to enhanced_road_types_features.mat\n');

function D = estimateFractalDimension(I)
    % Estimate the fractal dimension of an image I using the box-counting method
    % Convert the image to binary
    I = imbinarize(mat2gray(I));
    % Image size
    [rows, cols] = size(I);
    % Box sizes
    boxSizes = power(2, 0:floor(log2(min(rows, cols)))-1);
    boxCounts = zeros(size(boxSizes));
    % Count the number of boxes needed to cover the object for each box size
    for i = 1:length(boxSizes)
        boxSize = boxSizes(i);
        count = 0;
        for r = 1:boxSize:rows
            for c = 1:boxSize:cols
                if any(any(I(r:min(r+boxSize-1, rows), c:min(c+boxSize-1, cols))))
                    count = count + 1;
                end
            end
        end
        boxCounts(i) = count;
    end
    % Linear regression to find the slope of log(boxCounts) vs log(1/boxSizes)
    logBoxSizes = log(1 ./ boxSizes);
    logBoxCounts = log(boxCounts);
    p = polyfit(logBoxSizes, logBoxCounts, 1);
    D = -p(1);
end


function acf = autocorr2D(I)
    % Compute 2D autocorrelation of an image I
    I = mat2gray(I); % Normalize image to range [0, 1]
    I_mean = mean(I(:)); % Mean of the image
    I = I - I_mean; % Subtract the mean from the image
    % 2D autocorrelation
    acf = xcorr2(I);
    % Normalize by the number of elements and variance
    acf = acf / (numel(I) * var(I(:)));
end



