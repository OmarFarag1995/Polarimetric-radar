clc;
clear;
close all;

% Define the region of interest parameters
dopplerWindowSize = 5;  % Number of Doppler bins on each side of the ego Doppler bin
rangeIndicesForAnalysis = 1:300;  % Adjust this range as needed

% Parent directory containing all road type folders
parentDir = 'C:\Users\xjz6mk\Documents\work data\road_surface_monitoring\data\Road_types';

% Get all subdirectories (road types)
roadTypes = dir(parentDir);
roadTypes = roadTypes([roadTypes.isdir]);  % Keep only directories
roadTypes = roadTypes(~ismember({roadTypes.name}, {'.', '..'}));  % Remove . and ..

% Initialize cell arrays to store regions and labels for all road types
allRegions = cell(length(roadTypes), 1);
allLabels = cell(length(roadTypes), 1);

for roadTypeIdx = 1:length(roadTypes)
    roadType = roadTypes(roadTypeIdx).name;
    dataLocation = fullfile(parentDir, roadType);
    
    % Read the radar ADC data
    radarDataLocation = fullfile(dataLocation, 'Radar_Fwd');
    radarOutputFrames = dir(fullfile(radarDataLocation, '*.mat')); % Assuming .mat files

    % Initialize region arrays
    numFrames = length(radarOutputFrames);
    regions = cell(numFrames, 4);  % Store regions for 4 channels
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
                        regions{iRadarFrame, channel_index} = rdMap;
                    end
                end
            end
            
            labels{iRadarFrame} = roadType; % Store the label for the frame
        end
    end
    
    % Remove empty cells
    nonEmptyIdx = cellfun(@(x) ~isempty(x), regions(:, 1));  % Check first channel for non-empty cells
    regions = regions(nonEmptyIdx, :);
    labels = labels(nonEmptyIdx);
    
    % Store regions and labels
    allRegions{roadTypeIdx} = regions;
    allLabels{roadTypeIdx} = labels;
    
    fprintf('Processed road type: %s\n', roadType);
end

% Combine all regions and labels
combinedRegions = vertcat(allRegions{:});
combinedLabels = vertcat(allLabels{:});

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(combinedRegions, 1));
combinedRegions = combinedRegions(shuffledIndices, :);
combinedLabels = combinedLabels(shuffledIndices, :);

% Save combined regions and labels
save('full_ROI_road_types.mat', 'combinedRegions', 'combinedLabels');

fprintf('Region extraction complete. Data saved to regions_road_types.mat\n');
