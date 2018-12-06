import csv
import math
import operator
import nltk
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

trainphrase = []
trainset = []
testphrase = []
testphraseID = []
testset = []
label = []

def loadDataset(trainfile, testfile):
    with open(trainfile, 'r') as csvfile:
        trainlines = csv.reader(csvfile)
        trainset = list(trainlines)
        for x in range(1, len(trainset)):
            trainphrase.append(trainset[x][2])
            label.append(int(trainset[x][3]))
    with open(testfile, 'r') as csvfile:
        testlines = csv.reader(csvfile)
        testset = list(testlines)
        for x in range(1, len(testset)):
            testphrase.append(testset[x][2])
            testphraseID.append(testset[x][0])


def trainData():
    vectorizer = CountVectorizer(stop_words='english')
    Vector_train = vectorizer.fit_transform(trainphrase)
    Vector_test = vectorizer.transform(testphrase)

    with open('Prediction.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['PhraseID', 'Sentiment']);
        classifier = LogisticRegression(multi_class='auto')
        LR = classifier.fit(Vector_train, label)
        prediction = LR.predict(Vector_test)
        for i in range(0, len(prediction)):
            filewriter.writerow([testphraseID[i],prediction[i]])


def main():
    loadDataset('train.csv', 'testset_1.csv')
    trainData()

main()