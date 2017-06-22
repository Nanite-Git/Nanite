clear X;
fprintf("Lese Daten aus CSV\n");

Data=csvread('HistoryData.csv');
m=size(Data,1)/2;

%Hälfte der Daten weil Laptop zu wenig Speicher hat 
Data=Data(1:m,:);


diff_values = 6;                      %Anzahl verschiedener Informationsquellen (z.B.: Open, Close, High, Low)
anz_features = 60;                        %Anzahl Punkte pro Feature
fprintf("Erstelle X aus Daten\n");
[X y]=FeatureSort(Data,anz_features,diff_values);