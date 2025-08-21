clc;
clear
close all

%% Give the data Location and necessary range parameters
dataLocation = 'C:\Users\xjz6mk\Documents\work data\road_surface_monitoring\data\validation_datasets\validation_datasets';

%% Initialize and load system parameters
StreetTypes = dir(dataLocation);
StreetTypes = StreetTypes(~ismember({StreetTypes.name}, {'.', '..'}));

% Initialize a structure to store ADC data and labels
adcDataStruct = struct();

%% Main loop to extract ADC data
for iStreet = 1:length(StreetTypes)
    % Get the folder name and create a corresponding field in the structure
    classLabel = StreetTypes(iStreet).name;
    adcDataStruct.(classLabel) = [];

    % Define the path to the data files
    dataLoc = fullfile(StreetTypes(iStreet).folder, StreetTypes(iStreet).name, 'Radar_Fwd');
    radarFiles = dir(fullfile(dataLoc, '*.mat')); % Assuming radar data files are .mat

    % Loop through all radar files in the folder
    for iFile = 1:length(radarFiles)
        % Load the ADC data from the file
        adcDataFile = load(fullfile(dataLoc, radarFiles(iFile).name));
        ADCDataIn = adcDataFile.Look_A; % Adjust if the structure differs

        % Store the ADC data in the structure
        adcDataStruct.(classLabel)(end+1).ADCData = ADCDataIn;
        adcDataStruct.(classLabel)(end).fileName = radarFiles(iFile).name;
    end
end

%% Save the extracted ADC data structure
save('Data\ADC_Data\adcDataStruct_CNN_LSTM.mat', 'adcDataStruct', '-v7.3');
