%---------------------Reinigen------------------------%
clear all;
clc;

clear X, i;
%------------------------HistoryData einlesen----------------%
Data = csvread('HistoryData.csv');
%------------------------Konstanten definieren---------------%
anzfeat = 60;                        %Anzahl Punkte pro Feature
size_training = size(Data,1)/anz_pp_feature;     %Anzahl Trainingsexamples
diff_values = 6;                            %Anzahl verschiedener Informationsquellen (z.B.: Open, Close, High, Low)
max_diff = 0;     % Maximale Differenz zwischen zwei Open_Punkten
%-------------------Speicherallokierung und Initialisierung---------------%
X = zeros(size(Data,1)-anzfeat,anzfeat*(diff_values-1)+1);
arr_y  = zeros(size(X,1));        % Differenz zwischen aktuellem Punkt und zukünftigem Punkte
y = zeros(size(X,1));             % arr_y eingeteilt bzw. klassifiziert in x-Bereiche
%-------------------Berechnung x-----------------------%
for i=0:((size(Data,1)-1)-anzfeat)-1    %erstes -1 da von 0 iteriert, zweites -1 da die Calc für arr_y sonst übereläuft
    
  X(i+1,:) = [Data(i+anzfeat,1),reshape(Data((i+1):(anzfeat+i),2:diff_values),1,300)];
  
  arr_y(i) = Data(2,i+anzfeat)-Data(2,i+anzfeat+1)
  
%  if mod(i,50000)==0
%    fprintf('%i',i)
%    pause;
%  endif
endfor

%--------------------Berechnung y----------------------%
max_diff = max(abs(arr_y));

for i = -(max_diff/5):(max_diff/5)
y(arr_y == 0) = 

%x_new = zeros((length(x)/10)-1,10);
%for i=1:(length(x)/10)-1
%  
%  x_new(i,:) = x((1+10*(i-1)):10*i,:);
%  y(i) = x(1+i*10);
%  y_pre(i) = x(i*10);
%endfor
%
%rise_fall = (y-y_pre)>0