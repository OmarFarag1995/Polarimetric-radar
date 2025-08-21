
clear
close all

%% Give the data Location and necessary range parameters
dataLocation = 'C:\Users\rjnmlb\Aptiv\Polarimetric_radar\validation_datasets';

%% Initialize and load system parameters
StreetTypes = dir(dataLocation);
StreetTypes = StreetTypes(~ismember({StreetTypes.name}, {'.', '..'}));

% Initialize a structure to store ADC data and labels
adcDataStruct = struct();

% Define mapping of prefixes to unified classes
classMapping = struct('dry', 'dry', 'sno', 'snow', 'wet', 'wet');

%% Main loop to extract ADC data
for iStreet = 1:length(StreetTypes)
    % Get the folder name
    folderName = StreetTypes(iStreet).name;
    % Determine the class based on the prefix
    prefix = regexp(folderName, '^[^_]*', 'match', 'once');
    if isfield(classMapping, prefix)
        classLabel = classMapping.(prefix);
    else
        classLabel = prefix;  % If the prefix is not mapped, use it as is
    end

    % Ensure the class field exists in adcDataStruct
    if ~isfield(adcDataStruct, classLabel)
        adcDataStruct.(classLabel) = [];
    end

    % Define the path to the data files
    dataLoc = fullfile(StreetTypes(iStreet).folder, StreetTypes(iStreet).name, 'Radar_Fwd');
    radarFiles = dir(fullfile(dataLoc, '*.mat')); % Assuming radar data files are .mat

    % Loop through all radar files in the folder
    for iFile = 1:length(radarFiles)
        % Load the ADC data from the file
        adcDataFile = load(fullfile(dataLoc, radarFiles(iFile).name));
        ADCDataIn = adcDataFile.Look_A; % Adjust if the structure differs

        % Store the ADC data in the corresponding class structure
        adcDataStruct.(classLabel)(end+1).ADCData = ADCDataIn;
        adcDataStruct.(classLabel)(end).fileName = radarFiles(iFile).name;
    end
end

%% Save the extracted ADC data structure
save('..\Data\ADC_Data\adcDataStruct.mat', 'adcDataStruct', '-v7.3');

fprintf('Data Extraction is complete. Data saved to adcDataStruct.mat\n');
