clear all;
clc;

x = csvread('HistoryData.csv');


%x_new = zeros((length(x)/10)-1,10);
%for i=1:(length(x)/10)-1
%  
%  x_new(i,:) = x((1+10*(i-1)):10*i,:);
%  y(i) = x(1+i*10);
%  y_pre(i) = x(i*10);
%endfor
%
%rise_fall = (y-y_pre)>0