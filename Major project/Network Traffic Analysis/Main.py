from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

global filename

global X,Y
global dataset
global main
global text
accuracy = []
global X_train, X_test, y_train, y_test

main = tkinter.Tk()
main.title("Network Traffic Analysis Using Machine Learning") #designing main screen
main.geometry("1300x1200")

#traffic names VPN and NON-VPN
class_labels = ['BROWSING', 'CHAT', 'FT', 'MAIL', 'P2P', 'STREAMING', 'VOIP', 'VPN-BROWSING', 'VPN-CHAT', 'VPN-FT', 'VPN-MAIL', 'VPN-P2P',
                'VPN-STREAMING', 'VPN-VOIP']


#fucntion to upload dataset
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('class1').size()
    label.plot(kind="bar")
    plt.show()
    
#function to perform dataset preprocessing
def DataPreprocessing():
    global X,Y
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    le = LabelEncoder()
    #traffic type selection
    dataset['class1'] = pd.Series(le.fit_transform(dataset['class1'].astype(str)))
    text.insert(END,"Dataset after preprocessing\n\n")
    text.insert(END,str(dataset.head()))

    temp = dataset.values
    X = temp[:,0:dataset.shape[1]-1] #taking X and Y from dataset for training
    Y = temp[:,dataset.shape[1]-1]
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")

def runKNN():
    global X,Y
    global X_train, X_test, y_train, y_test
    global accuracy
    accuracy.clear()
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors = 2) 
    cls.fit(X, Y) 
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fmeasure = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"KNN Accuracy  :  "+str(a)+"\n")
    text.insert(END,"KNN Precision : "+str(precision)+"\n")
    text.insert(END,"KNN Recall    : "+str(recall)+"\n")
    text.insert(END,"KNN FScore    : "+str(fmeasure)+"\n\n")

def runNB():
    global X,Y
    global X_train, X_test, y_train, y_test
    global accuracy
    cls = GaussianNB() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fmeasure = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Naive Bayes Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Naive Bayes Precision : "+str(precision)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(recall)+"\n")
    text.insert(END,"Naive Bayes FScore    : "+str(fmeasure)+"\n\n")
    
def runDT():
    global X_train, X_test, y_train, y_test
    global accuracy
    cls = DecisionTreeClassifier() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fmeasure = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Decision Tree Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Decision Tree Precision : "+str(precision)+"\n")
    text.insert(END,"Decision Tree Recall    : "+str(recall)+"\n")
    text.insert(END,"Decision Tree FScore    : "+str(fmeasure)+"\n\n")

def runSVM():
    global X_train, X_test, y_train, y_test
    global accuracy
    cls = svm.SVC() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fmeasure = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"SVM Accuracy  :  "+str(a)+"\n")
    text.insert(END,"SVM Precision : "+str(precision)+"\n")
    text.insert(END,"SVM Recall    : "+str(recall)+"\n")
    text.insert(END,"SVM FScore    : "+str(fmeasure)+"\n\n")

    

def graph():
    height = accuracy
    bars = ('KNN Accuracy','Naive Bayes Accuracy','Decision Tree Accuracy','SVM Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Accuracy Comparison Graph")
    plt.show()

def GUI():
    global main
    global text
    font = ('times', 16, 'bold')
    title = Label(main, text='Network Traffic Analysis Using Machine Learning')
    title.config(bg='darkviolet', fg='gold')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 12, 'bold')
    text=Text(main,height=30,width=110)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=100)
    text.config(font=font1)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload Network Traffic Dataset", command=uploadDataset, bg='#ffb3fe')
    uploadButton.place(x=900,y=100)
    uploadButton.config(font=font1)  

    processButton = Button(main, text="Data Preprocessing", command=DataPreprocessing, bg='#ffb3fe')
    processButton.place(x=900,y=150)
    processButton.config(font=font1) 

    knnButton = Button(main, text="Run KNN Algorithm", command=runKNN, bg='#ffb3fe')
    knnButton.place(x=900,y=200)
    knnButton.config(font=font1) 

    nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB, bg='#ffb3fe')
    nbButton.place(x=900,y=250)
    nbButton.config(font=font1)

    dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDT, bg='#ffb3fe')
    dtButton.place(x=900,y=300)
    dtButton.config(font=font1) 

    svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
    svmButton.place(x=900,y=350)
    svmButton.config(font=font1)

    graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
    graphButton.place(x=900,y=400)
    graphButton.config(font=font1)

    main.config(bg='forestgreen')
    main.mainloop()
    
if __name__ == "__main__":
    GUI()


    
