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

%Save
save('datasets/Mat_X_clean.mat', 'X', 'col_x', 'Y', 'col_y');

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

