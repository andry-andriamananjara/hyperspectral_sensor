clear all; clc; close all;


%% DATA LOAD
load('Mat_X_clean.mat');


% DATA SPLIT: no need for validation and calibration separately if you do
% crossvalidation in the PLS: so just one partition
[XTrain, XTest, YTrain, YTest] = split_tain_test(X, Y);
[XCal, XVal, YCal, YVal] = split_tain_test(XTrain, YTrain);

%% DATA NORMALIZATION: no need to do this normalization inside the validation loop, each time the loop goes.

% Calibration-validation
[XCal, muCal, sigmaCal] = zscore(XCal); % No need for a separate function that just calls another function
[YCal, Meancal]         = centralised(YCal);

% Val
XVal                = normalize(XVal, 'center', muCal, 'scale', sigmaCal);
YVal                = YVal - Meancal;

% Test
XTest               = normalize(XTest, 'center', muCal, 'scale', sigmaCal);
YTest               = YTest - Meancal;


%% Deciding on the number of LVs
[row, col] = size(XCal);

nbcomp = 1:50;

for i=1:length(nbcomp)
    [Xload, Yload, XScore, YScore, betaPLS, PLSVar, PLSMSE, stats] = plsregress(XCal, YCal, nbcomp(i), "cv", 5);
    
    % R2 - You could have taken the R2 from the PLSVar variable that results from the PLS model as well.
   
    yfitPLS     = [ones(size(XCal,1),1) XCal] * betaPLS;
    TSS         = sum((YCal - mean(YCal)).^2); %% TSS = sum((YCal).^2); !Previous error.  TSS = sum((YCal - mean(YCal)).^2;
    RSS_PLS     = sum((YCal - yfitPLS).^2);
    R2PLS(i)    = 1 - (RSS_PLS/TSS); 


    % Calculate Q2 

    yfitPLSVal = [ones(size(XVal,1),1) XVal]*betaPLS;
    PRESS_PLS  = sum((YVal - yfitPLSVal).^2);
    Q2PLS(i)   = 1 - (PRESS_PLS/TSS);   
     
end

% Make decision on how many LVs you want to keep by plotting the R2, Q2 and
% MSECV plots

%%
figure;

nexttile;
bar(PLSMSE(2,:));
xlabel("No LVs in the PLS model.")
ylabel("Crossvalidation MSE");

nexttile;
plot(R2PLS);
xlabel("No LVs in the PLS model.");
ylabel("R^2(Y) cal");

nexttile; 
plot(Q2PLS);
xlabel("No LVs in the PLS model.");
ylabel("Q^2*Y) val");

%% Maybe you can select 21 LVs based on this plot, or around there. After we have done the calibration, we predict with the nu of chosen LVs the test set

[Xload, Yload, XScore, YScore, betaPLS, PLSVar, PLSMSE, stats] = plsregress(XCal, YCal, 21);
    

yfitPLSTest = [ones(size(XTest,1),1) XTest]*betaPLS;
PRESS_PLS_T  = sum((YTest - yfitPLSTest).^2);
Q2Test  = 1 - (PRESS_PLS_T/TSS);   


figure;
scatter(YTest, yfitPLSTest);
hold on
plot(YTest, YTest);
xlabel("True LMA value [mg] (scaled)");
ylabel("Predicted LMA value [mg] (scaled)");

% Except for a few bad fits, the model works pretty well.


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


function [XCal, meancal] = centralised(X)
    % We centralised
    meancal = mean(X);
    XCal = X - meancal; % If you already calculated mean(X) as a variable let's use it
end
