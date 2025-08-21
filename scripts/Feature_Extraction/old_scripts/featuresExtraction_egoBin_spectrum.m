clc;
clear;
close all;

% Parent directory containing all road type folders
parentDir = 'C:\Users\xjz6mk\Documents\work data\road_surface_monitoring\data\Road_types';

% Get all subdirectories (road types)
roadTypes = dir(parentDir);
roadTypes = roadTypes([roadTypes.isdir]);  % Keep only directories
roadTypes = roadTypes(~ismember({roadTypes.name}, {'.', '..'}));  % Remove . and ..

% Initialize cell arrays to store features and labels for all road types
allFeatures = cell(length(roadTypes), 1);
allLabels = cell(length(roadTypes), 1);

% Define the range indices to be analyzed
rangeIndicesForAnalysis = 10:150;  % Adjust this range as needed

for roadTypeIdx = 1:length(roadTypes)
    roadType = roadTypes(roadTypeIdx).name;
    dataLocation = fullfile(parentDir, roadType);
    
    % Read the radar ADC data
    radarDataLocation = fullfile(dataLocation, 'Radar_Fwd');
    radarOutputFrames = dir(fullfile(radarDataLocation, '*.mat')); % Assuming .mat files

    % Initialize feature arrays
    numFrames = length(radarOutputFrames);
    spectralFeatures = zeros(numFrames, 4 * 16); % 4 channels, 16 features per channel

    for iRadarFrame = 1:numFrames
        adcDataInput = load(fullfile(radarDataLocation, radarOutputFrames(iRadarFrame).name));
        adcDataPerFrame = adcDataInput.Look_A.Data; % Only checking LOOK A
        lookParam = adcDataInput.Look_A;
        
        if ~isempty(adcDataPerFrame)
            rdmOutput = generateRangeDoppler(adcDataPerFrame); % Get the RDM with windowing
            
            % Calculate ego Doppler bin
            if lookParam.VehSpeed_mps < 64*0.12
                ego_doppler_bin = 64 + floor(lookParam.VehSpeed_mps / 0.12);
            else
                ego_doppler_bin = rem(64 + floor(lookParam.VehSpeed_mps / 0.12), 128);
            end
            
            % Extract 1D Doppler spectrum at the ego Doppler bin for each channel
            for iRX = 1:2
                for iTX = 1:2
                    channel_index = 2*(iRX-1)+iTX;
                    
                    % Extract 1D Doppler spectrum
                    dopplerSpectrum = abs(squeeze(rdmOutput(rangeIndicesForAnalysis, ego_doppler_bin, iRX, iTX)));

                    % Spectral Features
                    % 1. Spectral Centroid
                    freqs = (1:length(dopplerSpectrum))'; % Create frequency vector
                    spectralCentroid = sum(freqs .* dopplerSpectrum) / sum(dopplerSpectrum);

                    % 2. Spectral Spread
                    spectralSpread = sqrt(sum(((freqs - spectralCentroid).^2) .* dopplerSpectrum) / sum(dopplerSpectrum));

                    % 3. Spectral Flatness
                    geometricMean = exp(mean(log(dopplerSpectrum + eps)));
                    arithmeticMean = mean(dopplerSpectrum);
                    spectralFlatness = geometricMean / arithmeticMean;

                    % 4. Spectral Skewness
                    spectralSkewness = sum(((freqs - spectralCentroid).^3) .* dopplerSpectrum) / ((sum(((freqs - spectralCentroid).^2) .* dopplerSpectrum))^(3/2));

                    % 5. Spectral Kurtosis
                    spectralKurtosis = sum(((freqs - spectralCentroid).^4) .* dopplerSpectrum) / ((sum(((freqs - spectralCentroid).^2) .* dopplerSpectrum))^2);

                    % Additional Features
                    % 6. Maximum Value of the Spectrum
                    maxSpectrum = max(dopplerSpectrum);

                    % 7. Beginning Value of the Spectrum
                    beginSpectrum = dopplerSpectrum(1);

                    % 8. End Value of the Spectrum
                    endSpectrum = dopplerSpectrum(end);

                    % 9. Spectral Roll-off
                    rolloffThreshold = 0.85 * sum(dopplerSpectrum);
                    cumulativeSum = cumsum(dopplerSpectrum);
                    spectralRolloff = find(cumulativeSum >= rolloffThreshold, 1);

                    % 10. Spectral Entropy
                    probDoppler = dopplerSpectrum / sum(dopplerSpectrum);
                    spectralEntropy = -sum(probDoppler .* log2(probDoppler + eps));

                    % 11. Spectral Contrast
                    spectralContrast = max(dopplerSpectrum) - min(dopplerSpectrum);

                    % 12. Band Energy Ratio (Low/High)
                    bandSplit = floor(length(dopplerSpectrum) / 2);
                    lowBandEnergy = sum(dopplerSpectrum(1:bandSplit));
                    highBandEnergy = sum(dopplerSpectrum(bandSplit+1:end));
                    bandEnergyRatio = lowBandEnergy / highBandEnergy;

                    % 13. RMS Amplitude
                    rmsAmplitude = sqrt(mean(dopplerSpectrum.^2));

                    % 14. Zero Crossing Rate
                    zeroCrossings = sum(diff(dopplerSpectrum > mean(dopplerSpectrum)) ~= 0);

                    % 15. Peak Frequency
                    [~, peakIndex] = max(dopplerSpectrum);
                    peakFrequency = freqs(peakIndex);

                    % 16. Total Energy
                    totalEnergy = sum(dopplerSpectrum.^2);

                    % Store features in the vector
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 1) = spectralCentroid;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 2) = spectralSpread;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 3) = spectralFlatness;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 4) = spectralSkewness;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 5) = spectralKurtosis;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 6) = maxSpectrum;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 7) = beginSpectrum;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 8) = endSpectrum;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 9) = spectralRolloff;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 10) = spectralEntropy;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 11) = spectralContrast;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 12) = bandEnergyRatio;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 13) = rmsAmplitude;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 14) = zeroCrossings;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 15) = peakFrequency;
                    spectralFeatures(iRadarFrame, (channel_index-1)*16 + 16) = totalEnergy;
                end
            end
        end
    end

    % Store features and labels
    allFeatures{roadTypeIdx} = spectralFeatures;
    allLabels{roadTypeIdx} = repmat({roadType}, numFrames, 1);
    
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
save('ego_doppler_spectral_features.mat', 'combinedFeatures', 'combinedLabels');

fprintf('Feature extraction complete. Data saved to ego_doppler_spectral_features.mat\n');