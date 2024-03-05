%% Script to generate csv/txt file from the dataset
% used variables: mCell and mInfo (if the ports numbers are needed)

clc
clear all
close all

%% DATA LOADING
% mCell = load('mCell_IP_extended.mat');
% mCell = mCell.mCell;

mCell = load('mCell_LTE','-mat');
mCell = mCell.mCellLte;

% mInfo = load('mInfo.mat');  % to use src and dst ports
% mInfo = mInfo.mInfo;

%% CSV FILE GENERATION - ONE COLUMN

array_patterns = cell(size(mCell,1),1);
array_patterns_sizes = zeros(size(mCell,1),1);
cont = 0;

for i = 1:size(mCell,1)
    
    current_cell = mCell{i};
    
    % uplink <= 0 in the dataset -> negative values in the csv file
    % downlink > 0 -> positive values in the csv file
    stream = -1*ones(size(current_cell,1),1);
    downlink_indexes = current_cell(:,3) > 0;
    stream(downlink_indexes) = 1;
    
    cell_pattern = current_cell(:,2);
    
%     if sum(cell_pattern > 1410) > 0
%         cell_pattern(cell_pattern > 1410) = 1410;
%         cont = cont +1;
%     end
    %cell_pattern = cell_pattern.*stream;
    array_patterns{i} = cell_pattern;
    array_patterns_sizes(i) = size(cell_pattern,1);
    
end

array_patterns = cell2mat(array_patterns);
reduced_patterns = unique(array_patterns);
histogram(array_patterns_sizes)

% csvwrite('dataset_full.csv', array_patterns);
% csvwrite('patterns_sizes.csv', array_patterns_sizes);
% array_patterns = cell2mat(array_patterns);
save('dataset_full', 'array_patterns');

%% CSV FILE GENERATION - MATRIX

maximum = 0;
index = 0;
for i = 1:size(mCell, 1)
    current_cell = mCell{i};
    cell_size = size(current_cell,1);
    
    if cell_size > maximum
        maximum = cell_size;
        index = i;
    end  
end

array_patterns = zeros(size(mCell,1),maximum);
sizes_array = zeros(size(array_patterns,1),1);

for i = 1:size(mCell,1)
    
    current_cell = mCell{i};
    
    % uplink <= 0 in the dataset -> negative values in the output file
    % downlink > 0 -> positive values in the output file
    stream = -1*ones(size(current_cell,1),1);
    downlink_indexes = current_cell(:,3) > 0;
    stream(downlink_indexes) = 1;
    
    cell_pattern = current_cell(:,2).*stream;
    sizes_array(i)=size(cell_pattern,1);
    array_patterns(i,1:size(cell_pattern,1)) = cell_pattern;
end

% figure, plot(array_patterns(1,:))
% hold on 
% plot(array_patterns(2,:))

% save('dataset_patterns', 'array_patterns');
% 
% array_patterns = array_patterns';
% save('dataset_transpose', 'array_patterns');


