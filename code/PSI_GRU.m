clc; clear; close all;

a = xlsread('GalwaySampleWaveData - final.xlsx');
rng('default');

numSteps = input('Enter number of forecast steps ahead (1–100): ');
if ~isscalar(numSteps) || numSteps < 1 || numSteps > 100
    error('Input must be a number between 1 and 100.');
end
forecastSteps = 1:numSteps;

d = input('Enter the last column index (d) for feature selection (>=1): ');
if ~isscalar(d) || d < 1 || d > size(a,2)
    error('Invalid value for d. Must be between 1 and %d.', size(a,2));
end

logFile = fopen('PSO_BiLSTM_Results.txt', 'w');

for j = forecastSteps
    fprintf('=========== Forecast Step Ahead: %d ===========\n', j);

    y = a(1:4300, 1:d);
    X = y(:, 2:d-1);
    Y = y(:, d);

    trainX = X(1:3200,:);
    trainY = Y(1+j:3200+j);
    testX  = X(3201:4000,:);
    testY  = Y(3201+j:4000+j);
    realheight = a(3201+j:4000+j, 1);

    trainX = trainX'; trainY = trainY';
    testX  = testX';  testY  = testY';

    valIdx = floor(0.8 * size(trainX,2));
    Xtrain = trainX(:,1:valIdx); Ytrain = trainY(:,1:valIdx);
    Xval   = trainX(:,valIdx+1:end); Yval = trainY(:,valIdx+1:end);

    % Run PSO optimization
 [bestParams, bestMSE, historyTable] = run_PSO(Xtrain, Ytrain, Xval, Yval);

    numHiddenUnits = round(bestParams(1));
    learnRate      = bestParams(2);
    maxEpochs      = round(bestParams(3));
    miniBatchSize  = round(bestParams(4));

    layers = [
        sequenceInputLayer(size(trainX,1))
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', learnRate, ...
        'GradientThreshold', 0.01, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

    net = trainNetwork(trainX, trainY, layers, options);
    YPred = predict(net, testX, 'MiniBatchSize', 1);
    prediction = YPred';

    err = prediction - testY;
    mse = mean(err.^2);

    save(sprintf('ForecastModel_Step_%d.mat', j), 'net', 'bestParams', 'mse');
    writetable(historyTable, sprintf('Optimization_History_Step_%d.xlsx', j));

    fprintf(logFile, 'Forecast Step: %d\n', j);
    fprintf(logFile, 'Hidden Units: %d\n', numHiddenUnits);
    fprintf(logFile, 'Learn Rate: %.6f\n', learnRate);
    fprintf(logFile, 'Epochs: %d\n', maxEpochs);
    fprintf(logFile, 'MiniBatch Size: %d\n', miniBatchSize);
    fprintf(logFile, 'Test MSE: %.6f\n\n', mse);

    figure('Name', sprintf('Forecast Step %d', j), 'NumberTitle','off');
    subplot(2,2,1); plot(testY, 'b'); hold on; plot(prediction, 'r--');
    title(sprintf('Actual vs Predicted (j = %d)', j));
    legend('Actual','Predicted'); xlabel('Time'); ylabel('GHI');

    subplot(2,2,2); histogram(err); title('Error Histogram');
    xlabel('Error'); ylabel('Frequency');

    subplot(2,2,3); boxplot(err); title('Prediction Error Boxplot');

    subplot(2,2,4); scatter(realheight, prediction, 'filled');
    title('Predicted vs Actual Max Height'); xlabel('Actual'); ylabel('Predicted');
    drawnow;
end

fclose(logFile);
sound(sin(1:9000));

%% === run_PSO.m ===
function [bestParams, bestFitness, historyTable] = run_PSO(trainX, trainY, valX, valY)
    rng('shuffle');
    lb = [20, 0.0001, 100, 8];  % Lower bounds
    ub = [70, 0.001, 300, 16];  % Upper bounds
    dim = length(lb);
    nPop = 10; MaxIt = 100;

    % Initialize swarm
    X = rand(nPop, dim).*(ub - lb) + lb;
    V = zeros(nPop, dim);
    fitness = zeros(nPop,1);

    % Evaluate initial fitness
    parfor i = 1:nPop
        fitness(i) = LSTM_Objective(X(i,:), trainX, trainY, valX, valY);
    end

    pBest = X; pBestFit = fitness;
    [gBestFit, gIdx] = min(fitness);
    gBest = X(gIdx,:);

    % History
    historyMSE = zeros(MaxIt,1);
    historyParams = zeros(MaxIt,dim);

    % Figure for live plot
    f = figure('Name','PSO Optimization Progress'); 
    msePlot = subplot(1,1,1);

    h = waitbar(0, 'Initializing PSO...');
    totalEval = nPop * MaxIt;
    startTime = tic;

    % PSO parameters
    w = 0.7; c1 = 1.5; c2 = 1.5;

    for t = 1:MaxIt
        parfor i = 1:nPop
            r1 = rand(1,dim); r2 = rand(1,dim);
            V(i,:) = w*V(i,:) + c1*r1.*(pBest(i,:) - X(i,:)) + c2*r2.*(gBest - X(i,:));
            X(i,:) = X(i,:) + V(i,:);
            X(i,:) = max(min(X(i,:), ub), lb);

            fit = LSTM_Objective(X(i,:), trainX, trainY, valX, valY);
            if fit < pBestFit(i)
                pBest(i,:) = X(i,:);
                pBestFit(i) = fit;
            end
        end

        [minFit, minIdx] = min(pBestFit);
        if minFit < gBestFit
            gBestFit = minFit;
            gBest = pBest(minIdx,:);
        end

        historyMSE(t) = gBestFit;
        historyParams(t,:) = gBest;

        % Live plot
        subplot(msePlot); cla;
        plot(1:t, historyMSE(1:t), 'b-o'); grid on;
        title('Best MSE vs Iteration'); xlabel('Iteration'); ylabel('MSE');
        drawnow;

        % Progress info
        percentDone = t / MaxIt;
        elapsed = toc(startTime);
        timePerIt = elapsed / t;
        estRemain = (MaxIt - t) * timePerIt;
        waitbar(percentDone, h, sprintf('Progress: %d%%%% - ETA: %.1f sec', ...
            round(percentDone*100), estRemain));

        fprintf('Iter %d/%d: BestFit=%.6f | Hidden=%d, LR=%.5f, Ep=%d, MB=%d\n', ...
            t, MaxIt, gBestFit, round(gBest(1)), gBest(2), round(gBest(3)), round(gBest(4)));
    end

    close(h);
    bestParams = gBest; bestFitness = gBestFit;

    historyTable = table((1:MaxIt)', historyMSE, historyParams(:,1), historyParams(:,2), ...
        historyParams(:,3), historyParams(:,4), ...
        'VariableNames', {'Iteration','BestMSE','HiddenUnits','LearnRate','Epochs','MiniBatch'});
end

%% === Objective Function ===
function mse = LSTM_Objective(params, Xtrain, Ytrain, Xval, Yval)
    numHiddenUnits = round(params(1));
    learnRate      = params(2);
    maxEpochs      = round(params(3));
    miniBatchSize  = round(params(4));

    layers = [
        sequenceInputLayer(size(Xtrain,1))
        gruLayer(numHiddenUnits)
        fullyConnectedLayer(1)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', learnRate, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ValidationData', {Xval, Yval}, ...
        'ValidationFrequency', 30, ...
        'GradientThreshold', 0.01);

    try
        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Ypred = predict(net, Xval, 'MiniBatchSize', 1);
        mse = mean((Ypred(:) - Yval(:)).^2);
        if isnan(mse) || isinf(mse)
            mse = 1e6;
        end
    catch
        mse = 1e6;
    end
end
