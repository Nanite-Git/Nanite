%clear all;
clc;

Data = csvread('HistoryData.csv');

anz_pp_feature = 60;                        %Anzahl Punkte pro Feature
size_training = size(Data,1)/anz_pp_feature;     %Anzahl Trainingsexamples
%diff_values = 6;                            %Anzahl verschiedener Informationsquellen (z.B.: Open, Close, High, Low)

%for j=1:size_training  
%  X(j,1) = Data(anz_pp_feature*j,1);
%  for i=1:anz_pp_feature
%    for k=2:diff_values-1 
%      X(j,2+(k-2)+(i-1)*(diff_values)) = Data(i+(j-1)*anz_pp_feature,k);
%     endfor
%  endfor
%endfor

%  X = Data(1:anz_pp_feature,2:diff_values);
%  X = X(:);
%for i=2:size_training
%  X = X + Data(1+60*(i-1):60*i,2:diff_values);
%endfor
  

%x_new = zeros((length(x)/10)-1,10);
%for i=1:(length(x)/10)-1
%  
%  x_new(i,:) = x((1+10*(i-1)):10*i,:);
%  y(i) = x(1+i*10);
%  y_pre(i) = x(i*10);
%endfor
%
%rise_fall = (y-y_pre)>0