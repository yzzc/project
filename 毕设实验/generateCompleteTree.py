# Copyright 2016-2024, Yanlin Duan, dyling2016@163.com; Shuyin Xia, xia_shuyin @ outlook;
# Generate a completely random tree: for noise detection

import numpy
from numpy import *
from collections import Counter
from sklearn.neighbors import kNeighborsClassifier as kNN
from keras.models import Sequential # A CNN is more commonly used to the sequential network structure
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier # classification
from sklearn.linear_model import LogisticRegression

class BinaryTree:
    def __init__(self, labels = array([]), datas = array([])):
        self.label = labels
        self.data = datas
        self.leftChild = None
        self.rightChild = None
    def set_rightChild(self, rightObj):
        self.rightChild = rightObj
    def set_leftChild(self, leftObj):
        self.leftChild = leftObj
    def get_rightChild(self):
        return self.rightChild
    def get_leftChild(self):
        return self.leftChild
    def get_data(self):
        return self.data
    def get_label(self):
        return self.label

# Export Chinese for python2
def print_(hanzi):
    print((hanzi).decode('utf-8'))

# Divide the data with the splitValue of the splitAttribute column element into two parts: leftData and rightData
def splitData(data, splitAttribute, splitValue):
    leftData = array([])
    rightData = array([])
    for c in data [:,]:
        if c [splitAttribute]>splitValue:
            if len(rightData) == 0:
                rightData = c
            else:
                rightData = vstack((rightData, c))
        else:
            if len(leftData) == 0:
                leftData = c
            else:
                leftData = vstack((leftData, c))

    return leftData, rightData

#data is a two-dimensional matrix data
# The first column is the label [0,1], or [-1,1]
# The last column is the sample ordinal
# Returns the root node of a tree
minNumSample = 10
def generateTree(data, uplabels = []):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size
    
    if numberAttribute == 0:
        return None
    
    numberAttribute = numberAttribute-2
    
    # The current data category, also called the node category
    labelNumKey = []
    if numberSample == 1:
        labelvalue = data [0]
        rootdata = data [numberAttribute + 1]
    else:
        # labelAttribute = data [:, 0]
        labelNum = Counter(data [:, 0])
        labelNumKey = list(labelNum.keys())
        labelNumValue = list(labelNum.values())
        labelvalue = labelNumKey [labelNumValue.index(max(labelNumValue))]
        rootdata = data [:, numberAttribute + 1]
    
    rootlabel = hstack((labelvalue, uplabels))
    
    CRTree = BinaryTree(rootlabel, rootdata)
    
    # Tree stops growing at least two conditions: 1 number of samples limit; 2 first column all equal
    if numberSample <minNumSample or len(labelNumKey) <2:
        return CRTree
    else:
        splitAttribute = 0 # randomly get the partition attribute
        splitValue = 0 # randomly get the value in the partition attribute
        maxCycles = 1.5 * numberAttribute #Maximum number of cycles
        i = 0
        while True: # Once the data exception occurs: In addition to the above two kinds of conditions to stop the tree growth conditions, that is, the wrong data, where the cycle will not stop
            i += 1
            splitAttribute = random.randint(1, numberAttribute) # function Returns an integer that includes a range boundary
            if splitAttribute>0 and splitAttribute <numberAttribute + 1: # Matches the attribute column of the matrix request
                dataSplit = data [:, splitAttribute]
                # uniquedata = list(Counter(dataSplit) .keys()) # acts the same as the following line
                uniquedata = list(set(dataSplit))
                if len(uniquedata)>1:
                    break
            if i>maxCycles: # data exception caused by the tree to stop growing
                print('data exception')
                return CRTree
        sv1 = random.choice(uniquedata)
        i = 0;
        while True:
            i += 1
            sv2 = random.choice(uniquedata)
            if sv2!=sv1:
                break
            if i>maxCycles:
                print('find split point timeout')
                return CRTree
        splitValue = mean([sv1, sv2])
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree

#Call functions
def CRT(data):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        return None
    orderAttribute = arange(numberSample) .reshape(numberSample, 1)
    data = hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree

# Returns the matrix of two rows of N columns, the first row is the sample label, and the second row is the noise threshold
def visitCRT(tree):
    if tree.get_leftChild() == None and tree.get_rightChild() == None:
        data = tree.get_data()
        labels = checkLabelSequence(tree.get_label())
        try:
            labels = zeros(len(data)) + labels
        except TypeError:
            pass
        result = vstack((data, labels))
        return result
    else:
        resultLeft = visitCRT(tree.get_leftChild())
        resultRight = visitCRT(tree.get_rightChild())
        result = hstack((resultLeft, resultRight))
        return result


# Returns the number of changes between a sequence of the last two times
def checkLabelSequence(labels):
    index1 = 0
    for i in range(1, len(labels)):
        if labels [index1]!= labels[i]:
            index1 = i
            break
    if index1 == 0:
        return 0

    index2 = 0
    for i in range(index1 + 1, len(labels)):
        if labels [index1]!= labels [i]:
            index2 = i
            break
    if index2 == 0:
        index2 = len(labels)
    return index2-index1    

# Returns whether the sequence of noise data is a tree
def filterNoise(data, tree = None, niThreshold = 3):
    if tree == None:
        tree = CRT(data)
    visiTree = visitCRT(tree)
    visiTree = visiTree [:, argsort(visiTree [0 ,:])]
    for i in range(len(visiTree [0 ,:])):
        if visiTree [1, i]>= niThreshold: # is noise
            visiTree [1, i] = 1
        else:
            visiTree [1, i] = 0
    return visiTree [1 ,:]

# Returns whether it is a sequence of noise data - forest
def CRFNFL(data, ntree = 100, niThreshold = 3):
    m, n = data.shape
    result = zeros((m, ntree))
    for i in range(ntree):
        visiTree = filterNoise(data, niThreshold = niThreshold)
        result [:, i] = visiTree

    noiseData = []
    for i in result:
        if sum(i)>= 0.5 * ntree:
            noiseData.append(1)
        else:
            noiseData.append(0)

    return array(noiseData)

# Delete exception data
def deleteNoiseData(data, noiseOrder):
    flag = 0;
    for i in range(noiseOrder.size):
        if noiseOrder [i] == 0:
            if flag == 0:
                redata = data [i ,:]
                flag = 1
            else:
                redata = vstack((redata, data [i ,:]))
    return redata

def kNNFunc(traindata,testdata):
    traindatalabel=traindata[:,0]
    traindata=traindata[:,1:]
    testdatalabel=testdata[:,0]
    testdata=testdata[:,1:]
    model=kNN(n_neighbors=3,algorithm='brute')
    model.fit(traindata,traindatalabel)
    precision=model.score(testdata,testdatalabel)
    return precision

def bpNNFunc(traindata=None,testdata=None):
    try:
        m,numberAttributes=traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    #load data
    traindatalabel=traindata[:,0]
    traindata=traindata[:,1:]
    testdatalabel=testdata[:,0]
    testdata=testdata[:,1:]
    
# If the tag type is [-1 1] -> [0 1]
    for i in range(traindatalabel.size):
        if traindatalabel [i] == - 1:
            traindatalabel [i] = 0
    for i in range(testdatalabel.size):
        if testdatalabel [i] == - 1:
            testdatalabel [i] = 0
    
    #Creata a model
    model = Sequential()
    # Add the input layer to hide the layer's connection
    # Add Dense full connection of the nerve layer, there are two parameters, one is the input data and output data dimension
    # If you need to add the next nerve layer, do not need to define the input latitude, because it defaults to the output of the previous layer as the current layer of input
    model.add(Dense(20, input_dim = numberAttributes-1, init = 'uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)) # to prevent over-fitting, with a certain probability of some neurons do not work

    model.add(Dense(10, init = 'uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)) # to prevent over-fitting, with a certain probability of some neurons do not work

    model.add(Dense(1, init = 'uniform')) # The Sigmoid function is the activation function
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25)) # to prevent over-fitting, with a certain probability of some neurons do not work
    
    # Compile the model, the loss function is, with adam solution
    model.compile(loss = 'binary_crossentropy', # binary_crossentropy: logloss(logarithmic loss)
                optimizer = 'adam', #optimizers: this is used to optimize the method: adam(gradient descent algorithm) sgd
                metrics = ['accuracy']) #metrics: Model Evaluation(Accuracy) accuracy

    # Default saved in the current directory
    # model.save('my_model.h5')

    # Start training
    model.fit(traindata, traindatalabel, epochs = 10, batch_size = 32, verbose = 1)

    # Return to the prediction accuracy of three methods
    # Method 1: evaluafe()
    precision = model.evaluate(testdata, testdatalabel)
    precision = precision [1]
    
    ## Method 2: predict()
    # prelabel = model.predict(testdata)
    # prelabel2 = [round_(i) for i in prelabel]
    # precision2 = 0
    #for i in range(len(prelabel2)):
    # if prelabel2 [i] == testdatalabel [i]:
    # precision2 += 1
    # precision2 = precision2 / len(prelabel2)

    ## Method 3: predict_classes()
    # prelabel3 = model.predict_classes(testdata)
    # precision3 = 0
    #for i in range(len(prelabel3)):
    # if prelabel3 [i] == testdatalabel [i]:
    # precision3 += 1
    # precision3 = precision3 / len(prelabel3)

    #print('precision1, precision2, precision3 =:', precision, precision2, precision3)
    return precision

def svmFunc(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata [:, 0]
    traindata = traindata [:, 1:]
    testdatalabel = testdata [:, 0]
    testdata = testdata [:, 1:]
    
    model = svm.SVC(kernel = 'rbf') # linear
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel) # prediction accuracy
    return precision

def cartFunc(traindata, testdata):
#from sklearn.ensemble import RandomForestClassifier # classification
#from sklearn.ensemble import RandomForestRegressor # Regression
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata [:, 0]
    traindata = traindata [:, 1:]
    testdatalabel = testdata [:, 0]
    testdata = testdata [:, 1:]
    
    model = RandomForestClassifier(n_estimators = 1) # The situation of a tree
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel) # prediction accuracy
    return precision

def lrFunc(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata [:, 0]
    traindata = traindata [:, 1:]
    testdatalabel = testdata [:, 0]
    testdata = testdata [:, 1:]

    model = LogisticRegression() # KNN, k = 3
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel) # prediction accuracy
    return precision

# Combine the noise detection classification algorithm:
def CRFNFL_kNN(traindata, testdata, ntree = 100, niThreshold = 11):
    # Create ntree trees
    if ntree <1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m, n = traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest = array([]) # m The matrix of the row ntree column
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree [:, argsort(visiTree [0 ,:])]
        visiTree = visiTree [1 ,:]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))
    
    precision = 0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
        print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):
                    if forest [j, k]>= subNi:
                        noiseForest [j] += 1;

                if noiseForest [j]>= 0.5 * ntree: #votes
                    noiseForest [j] = 1
                else:
                    noiseForest [j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            #print('denoiseTraindata.shape', denoiseTraindata.shape)
            preTemp = kNNFunc(denoiseTraindata, testdata) # call the system basic classification algorithm
            #print("preTemp:", preTemp)
            if precision <preTemp:
                precision = preTemp

    else: # forest size is greater than 10, every 10 trees as a interval traversal of different sizes of precision
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # # SubForest as a forest for different niThreshold traversal
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest [j, k]>= subNi:
                            noiseForest [j] += 1;

                    if noiseForest [j]>= 0.5 * subNtree:
                        noiseForest [j] = 1
                    else:
                        noiseForest [j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                #print('denoiseTraindata.shape', denoiseTraindata.shape)
                preTemp = kNNFunc(denoiseTraindata, testdata) # Basic classifier classification results                
                #print("preTemp:",preTemp)
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=kNNFunc(denoiseTraindata,testdata)
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def CRFNFL_BPNN(traindata,testdata,ntree=100,niThreshold=11):
    #build ntree trees as a forest
    if ntree<1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m,n=traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest=array([]) # m rows of ntree columns
    for i in range(ntree):
        tree=CRT(traindata)
        visiTree=visitCRT(tree)
        visiTree=visiTree[:,argsort(visiTree[0,:])]
        visiTree=visiTree[1,:]
        if forest.size==0:
            forest=visiTree.reshape(m,1)
        else:
            forest=hstack((forest,visiTree.reshape(m,1)))
    
    precision=0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
        print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):  
                    if forest[j,k]>=subNi:
                        noiseForest[j]+=1

                if noiseForest[j]>=0.5*ntree: #votes
                    noiseForest[j]=1
                else:
                    noiseForest[j]=0
            
            denoiseTraindata=deleteNoiseData(traindata,noiseForest)
            #print('denoiseTraindata:',denoiseTraindata.shape)
            preTemp=bpNNFunc(denoiseTraindata,testdata) #Call the system basic classification algorithm
            #print("preTemp:",preTemp)
            if precision<preTemp:
                precision=preTemp
    else: # forest size is greater than 10, every 10 trees as a interval traversal of different sizes of precision
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # # SubForest as a forest for different niThreshold traversal            
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*subNtree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata:',denoiseTraindata.shape)
                preTemp=bpNNFunc(denoiseTraindata,testdata) #Basic classifier classification results
                #print("preTemp:",preTemp)
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=bpNNFunc(denoiseTraindata,testdata)
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def CRFNFL_kMeansTree(traindata,testdata,ntree=100,niThreshold=11):
    #Build ntree trees
    if ntree<1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m,n=traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest=array([]) # m rows of ntree columns
    for i in range(ntree):
        tree=CRT(traindata)
        visiTree=visitCRT(tree)
        visiTree=visiTree[:,argsort(visiTree[0,:])]
        visiTree=visiTree[1,:]
        if forest.size==0:
            forest=visiTree.reshape(m,1)
        else:
            forest=hstack((forest,visiTree.reshape(m,1)))
    
    precision=0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
        print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):  
                    if forest[j,k]>=subNi:
                        noiseForest[j]+=1;

                if noiseForest[j]>=0.5*ntree: #votes
                    noiseForest[j]=1
                else:
                    noiseForest[j]=0;

            denoiseTraindata=deleteNoiseData(traindata,noiseForest);
            preTemp=0 #Call the system basic classification algorithm
            if precision<preTemp:
                precision=preTemp

    else:  #Forest size is greater than 10, every 10 trees as a traversal of the accuracy of different sizes
        startNtree=1;
        endNtree=ntree//10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # # SubForest as a forest for different niThreshold traversal
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1;

                    if noiseForest[j]>=0.5*subNtree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                preTemp=0 #Basic classifier classification results
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=0 #basic classifier classification
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def CRFNFL_SVM(traindata,testdata,ntree=100,niThreshold=11):
    #create ntree trees as a forest
    if ntree<1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m,n=traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest=array([]) # m rows ntree columns of matrix
    for i in range(ntree):
        tree=CRT(traindata)
        visiTree=visitCRT(tree)
        visiTree=visiTree[:,argsort(visiTree[0,:])]
        visiTree=visiTree[1,:]
        if forest.size==0:
            forest=visiTree.reshape(m,1)
        else:
            forest=hstack((forest,visiTree.reshape(m,1)))
    
    precision=0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
         print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
         for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):  
                    if forest[j,k]>=subNi:
                        noiseForest[j]+=1;

                if noiseForest[j]>=0.5*ntree: #votes
                    noiseForest[j]=1
                else:
                    noiseForest[j]=0;

            denoiseTraindata=deleteNoiseData(traindata,noiseForest);
            #print('denoiseTraindata.shape',denoiseTraindata.shape)
            preTemp=svmFunc(denoiseTraindata,testdata) #Call the system basic classification algorithm
            #print('preTemp',preTemp)
            if precision<preTemp:
                precision=preTemp

    else: # forest size is greater than 10, every 10 trees as a interval traversal of different sizes of precision
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            #print('subNtree:', subNtree)
            # # SubForest as a forest for different niThreshold traversal
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*subNtree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=svmFunc(denoiseTraindata,testdata) #basic classifier classification result
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=svmFunc(denoiseTraindata,testdata)
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def CRFNFL_LR(traindata,testdata,ntree=100,niThreshold=11):
    #create ntree trees
    if ntree<1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m,n=traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest=array([]) # m rows of ntree columns
    for i in range(ntree):
        tree=CRT(traindata)
        visiTree=visitCRT(tree)
        visiTree=visiTree[:,argsort(visiTree[0,:])]
        visiTree=visiTree[1,:]
        if forest.size==0:
            forest=visiTree.reshape(m,1)
        else:
            forest=hstack((forest,visiTree.reshape(m,1)))
    
    precision=0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
        print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):  
                    if forest[j,k]>=subNi:
                        noiseForest[j]+=1;

                if noiseForest[j]>=0.5*ntree: #votes
                    noiseForest[j]=1
                else:
                    noiseForest[j]=0;

            denoiseTraindata=deleteNoiseData(traindata,noiseForest)
            #print('denoiseTraindata.shape',denoiseTraindata.shape)
            preTemp=lrFunc(denoiseTraindata,testdata) #Call the system basic classification algorithm
            if precision<preTemp:
                precision=preTemp

    else: # forest size is greater than 10, every 10 trees as a interval traversal of different sizes of precision
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # # SubForest as a forest for different niThreshold traversal
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1;

                    if noiseForest[j]>=0.5*subNtree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=lrFunc(denoiseTraindata,testdata) #basic system method result
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=lrFunc(denoiseTraindata,testdata)
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def CRFNFL_Cart(traindata,testdata,ntree=100,niThreshold=11):
    #create ntree trees as a forest
    if ntree<1:
        print('The value of ntree at least is 1.')
        return 0
    
    try:
        m,n=traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest=array([]) # m rows of ntree columns
    for i in range(ntree):
        tree=CRT(traindata)
        visiTree=visitCRT(tree)
        visiTree=visiTree[:,argsort(visiTree[0,:])]
        visiTree=visiTree[1,:]
        if forest.size==0:
            forest=visiTree.reshape(m,1)
        else:
            forest=hstack((forest,visiTree.reshape(m,1)))
    
    precision=0
    if ntree <10: # forest size is less than 10, only discuss niThreshold changes in the case of the optimal accuracy
        print('forest size is less than 10, only discuss niThreshold changes in the case of optimal precision.')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # Use different niThreshold to detect noise data and update the optimum accuracy
            noiseForest = zeros(m) # Save the noise detection result
            # Start traversing the forest matrix
            for j in range(m): # A data detection process
                for k in range(ntree):  
                    if forest[j,k]>=subNi:
                        noiseForest[j]+=1;

                if noiseForest[j]>=0.5*ntree: #votes
                    noiseForest[j]=1
                else:
                    noiseForest[j]=0;

            denoiseTraindata=deleteNoiseData(traindata,noiseForest)
            #print('denoiseTraindata.shape',denoiseTraindata.shape)
            preTemp=cartFunc(denoiseTraindata,testdata) #call system classifier method
            if precision<preTemp:
                precision=preTemp

    else: # forest size is greater than 10, every 10 trees as a interval traversal of different sizes of precision
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree% 10; # If there is a remainder, and finally with all trees update once
        # Every 10 trees as a distance traversal of different sizes of precision
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # # SubForest as a forest for different niThreshold traversal
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1;

                    if noiseForest[j]>=0.5*subNtree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=cartFunc(denoiseTraindata,testdata) #system method result
                if precision<preTemp:
                    precision=preTemp

        if remainderNtree>0:
            #print('remainderNtree:',remainderNtree)
            for subNi in range(2,niThreshold+1):
                noiseForest=zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j,k]>=subNi:
                            noiseForest[j]+=1

                    if noiseForest[j]>=0.5*ntree:
                        noiseForest[j]=1
                    else:
                        noiseForest[j]=0

                denoiseTraindata=deleteNoiseData(traindata,noiseForest)
                #print('denoiseTraindata.shape',denoiseTraindata.shape)
                preTemp=cartFunc(denoiseTraindata,testdata)
                #print('preTemp',preTemp)
                if precision<preTemp:
                    precision=preTemp

    return precision

def crfnfl_all(traindata,testdata):
    m,n=traindata.shape
    print('traindata:',m,n)
    m,n=testdata.shape
    print('testdata:',m,n)
    print("Original accuracy")
    pre1=kNNFunc(traindata,testdata)
    pre2=bpNNFunc(traindata,testdata)
    pre3=svmFunc(traindata,testdata)
    pre4=lrFunc(traindata,testdata)
    pre5=cartFunc(traindata,testdata)

    print("knn = :",pre1)
    print("bpnn = :",pre2)
    print("svm = :",pre3)
    print("lr = :",pre4)
    print("cart = :",pre5)

    print("De-noising accuracy")
    pre1=CRFNFL_kNN(traindata,testdata,ntree=23,niThreshold=6)
    pre2=CRFNFL_BPNN(traindata,testdata,ntree=5,niThreshold=2)
    pre3=CRFNFL_SVM(traindata,testdata,ntree=24,niThreshold=6)
    pre4=CRFNFL_LR(traindata,testdata,ntree=23,niThreshold=6)
    pre5=CRFNFL_Cart(traindata,testdata,ntree=32,niThreshold=6)

    print("knn = :",pre1)
    print("bpnn = :",pre2)
    print("svm = :",pre3)
    print("lr = :",pre4)
    print("cart = :",pre5)
