clc;
clear;
close all;

%% Load the extracted features
load('..\Data\Feature_Vector_Data\Features_From_Full_ROI.mat', 'combinedFeatures', 'combinedLabels');

% Convert labels to categorical
Y = combinedLabels;

% Normalize features
mu = mean(combinedFeatures, 1);
sigma = std(combinedFeatures, [], 1);
X = (combinedFeatures - mu) ./ sigma;

% Shuffle the data
rng(0); % For reproducibility
shuffledIndices = randperm(size(X, 1));
X = X(shuffledIndices, :);
Y = Y(shuffledIndices, :);

%% Set up 5-fold cross-validation
k = 5;
cv = cvpartition(Y, 'KFold', k);
accuracies = zeros(k, 1);

for fold = 1:k
    % Get the training and testing indices for this fold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx, :);
    X_test = X(testIdx, :);
    Y_test = Y(testIdx, :);

    %% Define the neural network architecture
    numFeatures = size(X, 2);
    numClasses = numel(categories(Y));

    layers = [
        featureInputLayer(numFeatures, 'Name', 'input')
        fullyConnectedLayer(50, 'Name', 'fc1') % First hidden layer
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(25, 'Name', 'fc2') % Second hidden layer
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numClasses, 'Name', 'fc3') % Output layer
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')];

    %% Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {X_test, Y_test}, ...
        'ValidationFrequency', 10, ...
        'Verbose', false, ...
        'Plots', 'none'); % Disable training progress plot

    %% Train the neural network
    net = trainNetwork(X_train, Y_train, layers, options);

    %% Evaluate the network on the test data
    YPred = classify(net, X_test);
    accuracy = sum(YPred == Y_test) / numel(Y_test);
    accuracies(fold) = accuracy;
    disp(['Fold ', num2str(fold), ' - Test Classification Accuracy: ', num2str(accuracy * 100), '%']);
end

%% Calculate and display the average accuracy over all folds
meanAccuracy = mean(accuracies);
disp(['Average Classification Accuracy over ', num2str(k), '-fold cross-validation: ', num2str(meanAccuracy * 100), '%']);

%% Save the final trained model from the last fold (optional)
save('..\Data\Models\NN_Feature_Classification_Model.mat', 'net', 'mu', 'sigma');
fprintf('Neural network classification complete. Model saved to NN_Feature_Classification_Model.mat\n');
