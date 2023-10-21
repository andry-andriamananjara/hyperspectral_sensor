clear all; clc; close all;

load('datasets/Mat_X.mat');

% Separate the Y trait with missing value (Ymiss)
% Separate the Y trait without missing value (Yfull)
[Xfull, Yfull, Xmiss, Ymiss] = separate_missing_value(X, Y);

% Apply PLS for filling Ymiss
Ymiss = fill_missing_value(Xfull, Yfull, Xmiss, Ymiss);

%combine all data
X = vertcat(Xfull, Xmiss);
Y = vertcat(Yfull, Ymiss);

% Split the data
[X1, Xtest, Y1, Ytest] = split_tain_test(X, Y);
[Xtrain, XVal, Ytrain, YVal] = split_tain_test(X1, Y1);

% Normalisation
XCal = normalization(Xtrain);
YCal = normalization(Ytrain);
[row col] = size(XCal);

numVariables = size(XCal, 2);

% PCR with No. PCs 5.
% PCRmsep = sum(crossval(@pcrsse, XCal, YCal,'KFold',4),1)/rows; % number 4 means number of validations

% PLS with No. PCs 5.
[Xload, Yload, XScore, YScore, betaPLS, PLSVar, PLSMSE, stats] = plsregress(XCal, YCal, 140, "cv", 10);

% Error MSE plot for PCR and PLS.
figure;
hold on
plot(1:141,PLSMSE(1,:),'-k.');
xlabel('No. components');
ylabel('Estimated MSE');
title('MSE Curve');
legend('PLS: MSE in X');

% Explained Variables
figure; 
plot(1:140, 100 * cumsum(PLSVar(1,:))/ sum(PLSVar(1,:)),'b.-');
xlabel("No. PCs in the model");
ylabel("Percent explained variance");
title("Cummulative explained variances by principal components");

%% Plot PCs
figure;
hold on;
ii = 1;
for i = 1:10
    subplot(10,2,ii);
    plot(XScore(:,i),XScore(:,i+1),"r.");
    text1 = "PC " + string(i);
    text2 = "PC " + string(i+1);
    xlabel(text1);
    ylabel(text2);
    ii = ii + 1;

    subplot(10,2,ii);    
    plot(YScore(:,i),YScore(:,i+1),"b.");
    xlabel(text1);
    ylabel(text2);
    ii = ii + 1;
end


function  [Xtrain, Xtest, Ytrain, Ytest] = split_tain_test(X,Y)
    [rows cols] = size(X);
    % Set the seed for reproducibility
    rng(10);
    
    % Define the proportion of data for training
    trainingProportion = 0.8;
    
    % Create a random partition
    c = cvpartition(rows, 'HoldOut', 1 - trainingProportion);
    
    % Indices for training and testing sets
    trainIdx = training(c);
    testIdx = ~trainIdx;
    
    % Split the data
    Xtrain = X(trainIdx, :);
    Xtest = X(testIdx, :);
    Ytrain = Y(trainIdx, :);
    Ytest = Y(testIdx, :);
end

%% Function pretreatment

function [Xfull Yfull Xmiss Ymiss] = separate_missing_value(X, Y)
    %This function remove all missing value
    %We remove the value if the Y row is empty
    data = horzcat(X,Y);

    % Create a logical matrix of non-missing values
    nonMissingRows = ~any(isnan(Y), 2);

    % Create a logical matrix of missing values
    MissingRows = any(isnan(Y), 2);

    % Extract rows without missing values
    cleanedData = data(nonMissingRows, :);

    % Extract rows without missing values
    emptyData = data(MissingRows, :);

    % Xfull Yfull Xmiss Ymiss
    Xfull = cleanedData(:,[1:end-1]); 
    Yfull = cleanedData(:,end);
    Xmiss = emptyData(:,[1:end-1]); 
    Ymiss = emptyData(:,end);
end

function [Ymiss] = fill_missing_value(Xfull, Yfull, Xmiss, Ymiss)

    display(['Filling the missing values of Ymiss ...']);
    [rows col] = size(Xmiss);
    % we use 140 components for filling the data
    % Because it is among the small MSE value
    [XL, Yl, XS, YS, beta, PCTVAR, MSE, stats] = plsregress(Xfull, Yfull,140,"cv",10);

    %fill the missing value
    Ymiss = [ones(rows,1) Xmiss]*beta;

    % Create a logical matrix of missing values
    missingValues = isnan(Ymiss);
    
    % Count the number of missing values in each column
    missingValuesPerColumn = sum(missingValues);
    display(['Ymiss has '+string(missingValuesPerColumn)+' missing values now.']);

end

function [XCal] = normalization(X)
    % We use Zscore for normalizing the data
    [XCal, muCal, sigmaCal] = zscore(X);
end
