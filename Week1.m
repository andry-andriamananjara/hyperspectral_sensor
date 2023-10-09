clear all; clc; close all;

%% Import datasets

% filename = sprintf('ADAML_Project/datasets/data_part_%d.csv', 1);
df1 = readtable('ADAML-project/datasets/data_part_1.csv', 'Delimiter', ',', 'ReadVariableNames', true);
[row col] = size(df1);
display(["The fisrt dataset has ",num2str(row), " rows and ",num2str(col)," columns"]);

df2 = readtable('ADAML-project/datasets/data_part_2.csv', 'Delimiter', ',', 'ReadVariableNames', true);
[row col] = size(df2);
display(["The second dataset has ",num2str(row), " rows and ",num2str(col)," columns"]);

df1 = columns_process(df1);
df2 = columns_process(df2);

miss_col = miss_column_df2_func(df1,df2);
df1 = column_uniform_func(df1,miss_col);
miss_col = miss_column_df2_func(df2,df1);
df2 = column_uniform_func(df2,miss_col);

df_all =  vertcat(df1, df2);

%% split the input and the output data
[X col_x Y col_y] = split_input_output(df_all);
save('ADAML-project/datasets/Mat_X.mat', 'X', 'col_x', 'Y', 'col_y');

%% Visualisation
box_func(Y);

%% Plot matrix
figure;
plotmatrix(Y);
title(['Matrix plot']);

%% Function
function box_func(df1)
    boxplot(df1, 'Labels', 1:size(df1,2));
    title('Customized Box Plot');
    ylabel('Values');
    title(['Box plot']);
end

function [X col_x Y col_y] = split_input_output(df1)
    X = table2array(df1(:,[21:end]));
    col_x = df1(:,[21:end]).Properties.VariableNames;
    Y = table2array(df1(:,[1:20]));
    col_y = df1(:,[1:20]).Properties.VariableNames;
end

function [df] = columns_process(df1)
    % This function select the input (x values) and the output (traits) variables.
    col1 = df1.Properties.VariableNames;
    col_to_keep = [];
    
    % Remove column that contains concentration with units mg_g
    for i=1:length(col1)
        tf = contains(col1(i),"mg_g");
        % sf = contains(col1(i),"x");
        vf = contains(col1(i),"Var");
        if tf == 0 & vf == 0
            col_to_keep = [col_to_keep [col1(i)]];
        end
    end
    
    display(['No. Column to keep is ',num2str(length(col_to_keep))]);
    
    % Convert the reset of the colum into the right units
    for i=length(col_to_keep)
        list = ['BoronContent_mg_cm__','CaContent_mg_cm__','CopperContent_mg_cm__','MagnesiumContent_mg_cm__','ManganeseContent_mg_cm__', 'PhosphorusContent_mg_cm__', 'PotassiumContent_mg_cm__','SulfurContent_mg_cm__'];
        if ismember(col_to_keep(i), list)
            % convert to microgramme
            df1.(col_to_keep(i)) = df1.(col_to_keep(i))*1000;
        end
    end
    df = df1(:,col_to_keep);
end

function [miss_col] = miss_column_df2_func(df1,df2)
    % The dataframe must be loaded by readtable
    % This function checks the columns in df1 that are not in df2

    col1 = df1.Properties.VariableNames;
    col2 = df2.Properties.VariableNames;
    missing_column = [];

    for i = 1:length(col2)
        % Check if the column exists
        if ismember(col2(i), col1)
            a = 0;
        else
            missing_column = [missing_column col2(i)];
        end
    end
    miss_col = missing_column;
end

function [df] = column_uniform_func(df1,missing_column)
    % This function add missing_columns in dataframe df1
    for i = 1:length(missing_column)
        rd_value  = NaN(height(df1),1);
        df1 = addvars(df1,rd_value,'Before','x400','NewVariableNames', missing_column(i));
    end
    df = df1;
end