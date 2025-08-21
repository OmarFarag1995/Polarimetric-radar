clear; close all;

startBin = 12; endBin = 32; selectedBin = 22; % Change as needed

valLookNames = {'Look_A', 'Look_B', 'Look_C', 'Look_D'};
for iLook = 1:length(valLookNames)
    look = valLookNames{iLook};
    adcDataPath = fullfile('Data', 'ADC_Data', ['VAL_' look], ['adcDataStruct_Seq_Model_Validation_' look '.mat']);
    load(adcDataPath, 'adcDataStruct');

    nTXChannelsForAnalysis = 4;
    channelsToProcess = [1, 2, 5, 6];
    processedDataStruct = struct();
    fields = fieldnames(adcDataStruct);
    for iField = 1:length(fields)
        datasets = fields{iField};
        adcData = adcDataStruct.(datasets);
        processedDataStruct.(datasets) = struct();
        accumulatedRangeTimeMap = [];
        for iFile = 1:length(adcData)
            ADCDataIn = adcData(iFile).ADCData.Data;
            if isempty(ADCDataIn)
                warning('ADC data is empty for %s. Skipping...', adcData(iFile).fileName);
                continue;
            end
            [nRangeBins, nDopplerBins, nChannels] = size(ADCDataIn);
            channelData = double(ADCDataIn);
            hannWindowRange = repmat(hanning(nRangeBins), 1, nDopplerBins, nChannels);
            rangeFFT = fft(channelData .* hannWindowRange, [], 1) / nRangeBins;
            rangeFffTxSeperated = zeros(nRangeBins, nDopplerBins / 4, nChannels, nTXChannelsForAnalysis);
            rangeFffTxSeperated(:,:,:,1) = rangeFFT(:,1:4:end,:);
            rangeFffTxSeperated(:,:,:,2) = rangeFFT(:,2:4:end,:);
            rangeFffTxSeperated(:,:,:,3) = rangeFFT(:,3:4:end,:);
            rangeFffTxSeperated(:,:,:,4) = rangeFFT(:,4:4:end,:);
            accumulatedChannels = [];
            for chIdx = channelsToProcess
                rangeTimeData = abs(rangeFffTxSeperated(:,:,chIdx));
                accumulatedChannels = cat(3, accumulatedChannels, rangeTimeData);
            end
            concatenatedRangeTimeMap = permute(accumulatedChannels, [3, 2, 1]);
            accumulatedRangeTimeMap = cat(2, accumulatedRangeTimeMap, concatenatedRangeTimeMap);
        end
        processedDataStruct.(datasets).AccumulatedRangeTimeMap = accumulatedRangeTimeMap;
        processedDataStruct.(datasets).RangeBinsSubset = accumulatedRangeTimeMap(:,:, startBin+1:endBin);
        meanRangeBinsSubset = mean(processedDataStruct.(datasets).RangeBinsSubset, 3);
        processedDataStruct.(datasets).meanRangeBinsSubset = meanRangeBinsSubset;
        if endBin >= selectedBin
            processedDataStruct.(datasets).RangeBinSpecific = accumulatedRangeTimeMap(:,:, selectedBin);
        else
            processedDataStruct.(datasets).RangeBinSpecific = NaN(4, 1);
        end
    end
    saveFolder = fullfile('Data', 'Processed_Data', ['VAL_Range_Time_' look]);
    if ~exist(saveFolder, 'dir'), mkdir(saveFolder); end
    saveFile = fullfile(saveFolder, ['processedDataStruct_Validation_RangeTime_' look '.mat']);
    save(saveFile, 'processedDataStruct', '-v7.3');
    fprintf('Processed range-time maps for %s\n', look);
end
