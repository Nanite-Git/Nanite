import numpy as np
from sklearn import svm 
from sklearn.metrics import *
from sklearn.externals import joblib
import Read_Write_MySQL.Com_MySQL_001 as sql
import time 
import calendar

import matplotlib.pyplot as plt
#from pybrain.tools.shortcuts import buildNetwork
#from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.structure import *
#from pybrain.datasets import *
#from pybrain.structure.modules import *

load_brain = 'False'

def read_data_csv(fp):
    all_features = []
    timestamp_list =[]
    close_list = []
    high_list = []
    low_list = []
    open_price_list =[]
    volume_list = []
    datasetname = fp
    for line in datasetname:
        l=line.split(',')
        x = list(l[len(l)-1])
        x = x[0:len(x)-1]
        x = ''.join(x)
        l[len(l)-1]=x
        all_features.append(l)
        timestamp, close, high, low, open_price , volume = l
        timestamp_list.append(int(timestamp))
        close_list.append(float(close))
        high_list.append(float(high))
        low_list.append(float(low))
        open_price_list.append(float(open_price))
        volume_list.append(float(volume))
    return timestamp_list, close_list, high_list, low_list, open_price_list, volume_list

def read_data_sql(sql_stream):
    timestamp_list_pred =[]
    close_list_pred = []
    high_list_pred = []
    low_list_pred = []
    open_price_list_pred =[]
    volume_list_pred = []
    datasetname = sql_stream
    
    for i in range(len(datasetname)-1):
        l = datasetname[i]
        timestamp_pred, close_pred, high_pred, low_pred, open_pred , volume_pred = l
        timestamp_pred = calendar.timegm(timestamp_pred.timetuple())
        
        timestamp_list_pred.append(int(timestamp_pred))
        close_list_pred.append(float(close_pred))
        high_list_pred.append(float(high_pred))
        low_list_pred.append(float(low_pred))
        open_price_list_pred.append(float(open_pred))
        volume_list_pred.append(float(volume_pred))
    return timestamp_list_pred, close_list_pred, high_list_pred, low_list_pred, open_price_list_pred, volume_list_pred
    
    return 0

def creating_binary_labels(close_list, open_price_list):
    label_pos = 'True'
    
    if label_pos == 'True':
        label_list = close_list - [x*1.0 for x in open_price_list]        #0018264
    else:
        label_list = open_price_list - [x*1.0 for x in close_list]
    label_list = label_list[2:-1]
    for i in range(len(label_list)):
        if(label_list[i]>0):
            label_list[i]=1
        else:
            label_list[i]=0
    return label_list



def fearure_creation(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x):
    #Initialising
    open_change_percentage_list=[]
    close_change_percentage_list=[]
    low_change_percentage_list=[]
    high_change_percentage_list=[]
    volume_change_percentage_list=[]    
    volume_diff_percentage_list=[]
    open_diff_percentage_list=[]
    Open_price_moving_average_list=[]
    Close_price_moving_average_list=[]
    High_price_moving_average_list=[]
    Low_price_moving_average_list=[]

    highest_open_price = open_price_list[0]
    lowest_open_price = open_price_list[0]
    highest_volume = volume_list[0]
    lowest_volume = volume_list[0]
    if(x>len(open_price_list)):
        x = len(open_price_list)
    for i in range(len(close_list)-x,len(close_list)):
        if(highest_open_price<open_price_list[i]):
            highest_open_price=open_price_list[i]
        if(lowest_open_price>open_price_list[i]):
            lowest_open_price=open_price_list[i]
        if(highest_volume<volume_list[i]):
            highest_volume=volume_list[i]
        if(lowest_volume>volume_list[i]):
            lowest_volume=volume_list[i]

    #Finding change percentage list/difference list
    opensum=open_price_list[0]
    closesum=close_list[0]
    highsum=high_list[0]
    lowsum=low_list[0]
    for i in range(1, len(close_list)-2):
        close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
        close_change_percentage_list.append(close_change_percentage)
        
        open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
        open_change_percentage_list.append(open_change_percentage)
        high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
        high_change_percentage_list.append(high_change_percentage)
        if volume_list[i-1]==0:
            volume_list[i-1] = volume_list[i-2]
        volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
        volume_change_percentage_list.append(volume_change_percentage)
        low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
        low_change_percentage_list.append(low_change_percentage)

        volume_diff = (volume_list[i] - volume_list[i-1])/(highest_volume-lowest_volume)
        volume_diff_percentage_list.append( volume_diff)
        open_diff = (open_price_list[i+1] - open_price_list[i])/(highest_open_price - lowest_open_price)
        open_diff_percentage_list.append(open_diff)
        opensum+=open_price_list[i]
        closesum+=close_list[i]
        highsum+=high_list[i]
        lowsum+=low_list[i]
        Open_price_moving_average = float(opensum/i+1) / open_price_list[i+1]
        Open_price_moving_average_list.append(Open_price_moving_average)
        High_price_moving_average = float(highsum/i+1) / high_list[i+1]
        High_price_moving_average_list.append(High_price_moving_average)
        Close_price_moving_average = float(closesum/i+1) / close_list[i+1]
        Close_price_moving_average_list.append(Close_price_moving_average)
        Low_price_moving_average = float(lowsum/i+1) / low_list[i+1]
        Low_price_moving_average_list.append(Low_price_moving_average)
            
    
    #Combining features
    close_change_percentage_list = np.array(close_change_percentage_list)
    high_change_percentage_list = np.array(high_change_percentage_list)
    low_change_percentage_list = np.array(low_change_percentage_list)
    volume_change_percentage_list = np.array(volume_change_percentage_list)
    open_price_list = np.array(open_price_list)
    close_list = np.array(close_list)
    open_diff_percentage_list=np.array(open_diff_percentage_list)
    volume_change_percentage_list=np.array(volume_change_percentage_list)
    
    feature1 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list))
    
    feature2 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                open_diff_percentage_list, 
                                volume_diff_percentage_list)) 
    
    feature3 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                Open_price_moving_average_list, 
                                Close_price_moving_average_list, 
                                High_price_moving_average_list, 
                                Low_price_moving_average_list)) 
    
    feature4 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                open_diff_percentage_list, 
                                volume_diff_percentage_list,
                                Open_price_moving_average_list, 
                                Close_price_moving_average_list, 
                                High_price_moving_average_list, 
                                Low_price_moving_average_list))
    
    label_list = creating_binary_labels(close_list, open_price_list)
    
    return feature1, feature2, feature3, feature4, label_list

def fearure_next(timestamp_list, close_list, high_list, low_list, open_price_list, volume_list, x):
    #Initialising
    open_change_percentage_list=[]
    close_change_percentage_list=[]
    low_change_percentage_list=[]
    high_change_percentage_list=[]
    volume_change_percentage_list=[]    
    volume_diff_percentage_list=[]
    open_diff_percentage_list=[]
    Open_price_moving_average_list=[]
    Close_price_moving_average_list=[]
    High_price_moving_average_list=[]
    Low_price_moving_average_list=[]

    highest_open_price = open_price_list[0]
    lowest_open_price = open_price_list[0]
    highest_volume = volume_list[0]
    lowest_volume = volume_list[0]
    if(x>len(open_price_list)):
        x = len(open_price_list)
    for i in range(len(close_list)-x,len(close_list)):
        if(highest_open_price<open_price_list[i]):
            highest_open_price=open_price_list[i]
        if(lowest_open_price>open_price_list[i]):
            lowest_open_price=open_price_list[i]
        if(highest_volume<volume_list[i]):
            highest_volume=volume_list[i]
        if(lowest_volume>volume_list[i]):
            lowest_volume=volume_list[i]

    #Finding change percentage list/difference list
    opensum=open_price_list[0]
    closesum=close_list[0]
    highsum=high_list[0]
    lowsum=low_list[0]
    for i in range(1, len(close_list)-1):
        close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
        close_change_percentage_list.append(close_change_percentage)
        
        open_change_percentage = (open_price_list[i+1] - open_price_list[i])/open_price_list[i]
        open_change_percentage_list.append(open_change_percentage)
        high_change_percentage = (high_list[i] - high_list[i-1])/high_list[i-1]
        high_change_percentage_list.append(high_change_percentage)
        if volume_list[i-1]==0:
            volume_list[i-1] = volume_list[i-2]
        volume_change_percentage = (volume_list[i] - volume_list[i-1])/volume_list[i-1]
        volume_change_percentage_list.append(volume_change_percentage)
        low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
        low_change_percentage_list.append(low_change_percentage)

        volume_diff = (volume_list[i] - volume_list[i-1])/(highest_volume-lowest_volume)
        volume_diff_percentage_list.append( volume_diff)
        open_diff = (open_price_list[i+1] - open_price_list[i])/(highest_open_price - lowest_open_price)
        open_diff_percentage_list.append(open_diff)
        opensum+=open_price_list[i]
        closesum+=close_list[i]
        highsum+=high_list[i]
        lowsum+=low_list[i]
        Open_price_moving_average = float(opensum/i+1) / open_price_list[i+1]
        Open_price_moving_average_list.append(Open_price_moving_average)
        High_price_moving_average = float(highsum/i+1) / high_list[i+1]
        High_price_moving_average_list.append(High_price_moving_average)
        Close_price_moving_average = float(closesum/i+1) / close_list[i+1]
        Close_price_moving_average_list.append(Close_price_moving_average)
        Low_price_moving_average = float(lowsum/i+1) / low_list[i+1]
        Low_price_moving_average_list.append(Low_price_moving_average)
            
    
    #Combining features
    close_change_percentage_list = np.array(close_change_percentage_list)
    high_change_percentage_list = np.array(high_change_percentage_list)
    low_change_percentage_list = np.array(low_change_percentage_list)
    volume_change_percentage_list = np.array(volume_change_percentage_list)
    open_price_list = np.array(open_price_list)
    close_list = np.array(close_list)
    open_diff_percentage_list=np.array(open_diff_percentage_list)
    volume_change_percentage_list=np.array(volume_change_percentage_list)
    
    feature1 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list))
    
    feature2 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                open_diff_percentage_list, 
                                volume_diff_percentage_list)) 
    
    feature3 = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                Open_price_moving_average_list, 
                                Close_price_moving_average_list, 
                                High_price_moving_average_list, 
                                Low_price_moving_average_list)) 
    
    feature_next = np.column_stack((open_change_percentage_list, 
                                close_change_percentage_list, 
                                high_change_percentage_list, 
                                low_change_percentage_list, 
                                volume_change_percentage_list, 
                                open_diff_percentage_list, 
                                volume_diff_percentage_list,
                                Open_price_moving_average_list, 
                                Close_price_moving_average_list, 
                                High_price_moving_average_list, 
                                Low_price_moving_average_list))    
    return feature_next



def svm_rbf(feature, label_list, C_val = 100000, gamma_val = 1):
    length_feature = len(feature)
    len_train = int(0.75*length_feature)
    train_feature = feature[0: len_train]
    test_feature = feature[len_train: ]
    train_label = label_list[0:len_train]
    test_label = label_list[len_train:]
   
    if load_brain =='True':
        # and later you can load it
        clf = joblib.load('svm_brain.pkl')
    else:
        clf = svm.SVC(C=C_val, gamma = gamma_val, kernel='rbf')
        
    clf.fit(train_feature, train_label)
    predicted = clf.predict(test_feature)
    print("Accuracy: ", accuracy_score(predicted, test_label)*100, "%")
    print("Precision Score :", precision_score(predicted, test_label)*100, "%")
    print("Recall Score :" ,recall_score(predicted, test_label)*100, "%")
    
    # now you can save it to a file
    joblib.dump(clf, 'svm_brain.pkl') 
    
    return predicted, test_label, train_feature, train_label, test_feature, clf



def plotting_svm(predicted, test_labels,name,clr):
    step = np.arange(0, len(test_labels))
    plt.subplot(211)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.ylabel('Actual Values')
    plt.plot(step, test_labels, drawstyle = 'steps-mid' ,color=clr)
    plt.subplot(212)
    plt.xlim(-1, len(test_labels) + 1)
    plt.ylim(-1, 2)
    plt.xlabel('minutes')
    plt.ylabel('Predicted Values')
    plt.plot(step, predicted, drawstyle = 'steps-mid',color=clr)
    plt.savefig(name)
    plt.close()

#def neural_networks(train_feature, train_label, test_features, test_labels):
#    net = buildNetwork(len(train_feature[0]), 30, 1, hiddenclass = TanhLayer, outclass = TanhLayer,recurrent = True)
#    ds = ClassificationDataSet(len(train_feature[0]), 1)
#    for i, j in zip(train_feature, train_label):
#        ds.addSample(i, j)
#    trainer = BackpropTrainer(net, ds)
#    epochs = 13
#    for i in range(epochs):
#        trainer.train()
#    predicted = list()
#    for i in test_features:
#        predicted.append(int(net.activate(i)>0.5))
#    predicted = np.array(predicted)
#
#    print( "Accuracy:", accuracy_score(test_labels, predicted)*100, "%")
#    return predicted

if __name__ == '__main__':
# =============================================================================
#     Open CSV-Data TO LOAD TRAINING DATA
# =============================================================================
    fp= open("dataset/HistoryData.csv",'r+', encoding = 'utf-16')
    timestamp_list, close_list, high_list, low_list, open_price_list, volume_list = read_data_csv(fp)
    fp.close()
   
    x = 5
    feature1, feature2, feature3, feature4, label_list = fearure_creation(timestamp_list, close_list, 
                                                                          high_list, low_list, open_price_list, 
                                                                          volume_list, x )
#    print()
#    print( "-----------------------------------------------------------------------")
#    print( "SVM - RBF Kernel with Features : ")
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
#    predicted1, test_label1, train_feature1, train_label1, test_feature1 = svm_rbf(feature1, label_list)
#    print( "-----------------------------------------------------------------------")
#    
#    
#    print()
#    print( "-----------------------------------------------------------------------")
#    print( "SVM - RBF Kernel with Features : ")
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
#    print( "Open Difference% , Volume Difference%, ")
#    predicted2, test_label2, train_feature2, train_label2, test_feature2= svm_rbf(feature2, label_list)
#    print( "-----------------------------------------------------------------------")
#
#    print()
#    print( "-----------------------------------------------------------------------")
#    print( "SVM - RBF Kernel with Features : ")
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
#    print( "Open MovingAvg, Close MovingAvg, High MovingAvg, Low MovingAvg")
#    predicted3, test_label3, train_feature3, train_label3, test_feature3= svm_rbf(feature3, label_list)
#    print( "-----------------------------------------------------------------------")
#    
    
    print()
    print( "-----------------------------------------------------------------------")
    print( "SVM - RBF Kernel with Features : ")
        
    # Parameter für BTCUSD
    C_val = 52000               # 60000 + i*4000
    gamma_val = 0.028            # 0.7 + 0.2*(j/10)
    
    print("SVM-Parameter")
    print("C: ", C_val,"\nGamma: ",gamma_val)
#    for i in range(20):
#        C_val = 52000 #20000 + i*4000
#        gamma_val = 0.028
    print("...calculating...")
    predicted4, test_label4, train_feature4, train_label4, test_feature4, clf =  svm_rbf(feature4, label_list,C_val, gamma_val)

    #    precision_sum.append(precision_score(predicted4, test_label4)*100)
    #    
    #    print(np.where(precision_sum == max(precision_sum)))
    #print("Pred: ",predicted4[-10:],"\n","Real: ",test_label4[-10:])
    #    input("Wait key:")
    print( "-----------------------------------------------------------------------")


# =============================================================================
#     CONNECTION TO DATABASE
# =============================================================================
    #Get parameter to connect to Database and
    #Establishing connection to database
    config, cnx = sql.my_configparser()
    
    #Get cursorinformation from database
    cnx_cursor = cnx.cursor()
    

    if(1): #read_flag_and_reset()):
        query = ("SELECT timestamp, close, high, low, open, volume FROM test.historydata")
        #data extraction is a list
        data_extraction = sql.read_from_sql(cnx_cursor, query)
    time.sleep(10)
        
    cnx.commit()
    
    #CLOSE Database
    cnx_cursor.close()
    cnx.close()

    ############# UNDER CONSTRUCTION: NEED TO DISSOLVE THE LIST INTO THE FEED-DATA-TO-FEATURES ################
    
    timestamp_list_pred, close_list_pred, high_list_pred, low_list_pred, open_price_list_pred, volume_list_pred = read_data_sql(data_extraction)

    feature_next = fearure_next(timestamp_list_pred, close_list_pred, 
                                high_list_pred, low_list_pred, 
                                open_price_list_pred, volume_list_pred, x )
    
    predicted_next = clf.predict(feature_next)
    print("New: ",predicted_next[-10:])
    
    
    
#    plotting_svm(predicted1, test_label1,"Plots/hcl/hcltech1_22_apr_rbf.jpg",'r')
#    plotting_svm(predicted2, test_label2,"Plots/hcl/hcltech2_22_apr_rbf.jpg",'g')
#    plotting_svm(predicted3, test_label3,"Plots/hcl/hcltech3_22_apr_rbf.jpg",'b')
#    plotting_svm(predicted4, test_label4,"Plots/hcl/hcltech4_22_apr_rbf.jpg",'y')
#    print()
#    print( "*******************************RNN*************************************")
#    print()
#    print( "-----------------------------------------------------------------------")
#    #print( "SVM - RBF Kernel with Features : "
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
#    predicted_NN_1=neural_networks(train_feature1, train_label1, test_feature1, test_label1)
#    print( "-----------------------------------------------------------------------")
#
#
#    print()
#    print( "-----------------------------------------------------------------------")
#    #print( "SVM - RBF Kernel with Features : "
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
#    print( "Open Difference% , Volume Difference%, ")
#    predicted_NN_2=neural_networks(train_feature2, train_label2, test_feature2, test_label2)
#    print( "-----------------------------------------------------------------------")
#   
#    print()
#    print( "-----------------------------------------------------------------------")
#    #print( "SVM - RBF Kernel with Features : "
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%,")
#    print( "Open MovingAvg, Close MovingAvg, High MovingAvg, Low MovingAvg")
#    predicted_NN_3=neural_networks(train_feature3, train_label3, test_feature3, test_label3)
#    print( "-----------------------------------------------------------------------")
#
#    print()
#    print( "-----------------------------------------------------------------------")
#    #print( "SVM - RBF Kernel with Features : "
#    print( "Open Change%, Close Change%, High Change%, Low Change%, Volume Change%")
#    print( "Open Difference% , Volume Difference%, Open Price Moving Avg")
#    print( "Close Price Moving Avg, High Price Moving Avg, Low Price Moving Avg")
#    predicted_NN_4=neural_networks(train_feature4, train_label4, test_feature4, test_label4)
#    print( "-----------------------------------------------------------------------")
#   
#    plotting_svm(predicted_NN_1, test_label1,"Plots/hcl/hcltech1_22_apr_NN.jpg",'r')
#    plotting_svm(predicted_NN_2, test_label2,"Plots/hcl/hcltech2_22_apr_NN.jpg",'g')
#    plotting_svm(predicted_NN_2, test_label3,"Plots/hcl/hcltech3_22_apr_NN.jpg",'b')
#    plotting_svm(predicted_NN_2, test_label4,"Plots/hcl/hcltech4_22_apr_NN.jpg",'y')
