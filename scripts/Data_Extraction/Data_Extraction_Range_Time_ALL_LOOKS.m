clear; close all; clc;

% Folder containing validation data
dataLocation = 'C:\Users\xjz6mk\OneDrive - Aptiv\Documents\work data\road_surface_monitoring\data\Datasets\validation_datasets';

% List of LOOKs you want to extract (change if more/fewer looks!)
lookNames = {'Look_A', 'Look_B', 'Look_C', 'Look_D'};

% Loop over all LOOKs
for iLook = 1:length(lookNames)
    look = lookNames{iLook};
    StreetTypes = dir(dataLocation);
    StreetTypes = StreetTypes(~ismember({StreetTypes.name}, {'.', '..'}));

    adcDataStruct = struct();
    for iStreet = 1:length(StreetTypes)
        classLabel = StreetTypes(iStreet).name;
        adcDataStruct.(classLabel) = [];

        dataLoc = fullfile(StreetTypes(iStreet).folder, StreetTypes(iStreet).name, 'Radar_Fwd');
        radarFiles = dir(fullfile(dataLoc, '*.mat'));

        for iFile = 1:length(radarFiles)
            adcDataFile = load(fullfile(dataLoc, radarFiles(iFile).name));
            if isfield(adcDataFile, look)
                ADCDataIn = adcDataFile.(look);
            else
                warning('No %s in %s. Skipping.', look, radarFiles(iFile).name);
                continue;
            end
            adcDataStruct.(classLabel)(end+1).ADCData = ADCDataIn;
            adcDataStruct.(classLabel)(end).fileName = radarFiles(iFile).name;
        end
    end

    saveFolder = fullfile('Data', 'ADC_Data', ['VAL_' look]);
    if ~exist(saveFolder, 'dir')
        mkdir(saveFolder);
    end
    saveFile = fullfile(saveFolder, ['adcDataStruct_Seq_Model_Validation_' look '.mat']);
    save(saveFile, 'adcDataStruct', '-v7.3');
    fprintf('Saved extracted ADC data for %s\n', look);
end
