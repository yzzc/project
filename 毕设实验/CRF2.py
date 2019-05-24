# Copyright 2016-2024, Yanlin Duan, comdyling2016@163.com; Shuyin Xia,xia_shuyin@outlook;
# 生成完全随机树：用于噪声检测

import numpy
from numpy import *
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier as kNN
##from keras.models import Sequential  # 一种是CNN比较常用到的sequential网络结构
##from keras.layers.core import Dense, Dropout, Activation
##from keras.models import load_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier  # 分类
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

class BinaryTree:
    def __init__(self, labels=array([]), datas=array([])):
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


# 输出中文,用于python2
def print_(hanzi):
    print((hanzi).decode('utf-8'))


# 将data 以第splitAttribute列元素的splitValue为界划分成leftData 和rightData两部分
def splitData(data, splitAttribute, splitValue):
    leftData = array([])
    rightData = array([])
    for c in data[:, ]:
        if c[splitAttribute] > splitValue:
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


# data 为二维矩阵数据
# 第一列为标签[0,1]，或者[-1,1]
# 最后一列为样本序数
# 返回一个树的根节点
minNumSample = 10


def generateTree(data, uplabels=[]):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        numberSample = 1
        numberAttribute = data.size

    if numberAttribute == 0:
        return None

    numberAttribute = numberAttribute - 2

    # 当前数据的类别，也叫节点类别
    labelNumKey = []
    if numberSample == 1:
        labelvalue = data[0]
        rootdata = data[numberAttribute + 1]
    else:
        # labelAttribute=data[:,0]
        labelNum = Counter(data[:, 0])
        labelNumKey = list(labelNum.keys())
        labelNumValue = list(labelNum.values())
        labelvalue = labelNumKey[labelNumValue.index(max(labelNumValue))]
        rootdata = data[:, numberAttribute + 1]

    rootlabel = hstack((labelvalue, uplabels))

    CRTree = BinaryTree(rootlabel, rootdata)

    # 树停止增长的条件至少有两个：1样本个数限制；2第一列全部相等
    if numberSample < minNumSample or len(labelNumKey) < 2:
        return CRTree
    else:
        splitAttribute = 0  # 随机得到划分属性
        splitValue = 0  # 随机得到划分属性中的值
        maxCycles = 1.5 * numberAttribute  # Maximum number of cycles
        i = 0
        while True:  # 一旦出现数据异常：除了上面两种停止树增长的条件外的异常情况，即为错误数据，这里的循环将不发停止
            i += 1
            splitAttribute = random.randint(1, numberAttribute)  # 函数返回包括范围边界的整数
            if splitAttribute > 0 and splitAttribute < numberAttribute + 1:  # 符合矩阵要求的属性列
                dataSplit = data[:, splitAttribute]
                # uniquedata=list(Counter(dataSplit).keys()) #作用同下面一行
                uniquedata = list(set(dataSplit))
                if len(uniquedata) > 1:
                    break
            if i > maxCycles:  # 数据异常导致的树停止增长
                # print('数据异常')
                return CRTree
        sv1 = random.choice(uniquedata)
        i = 0;
        while True:
            i += 1
            sv2 = random.choice(uniquedata)
            if sv2 != sv1:
                break
            if i > maxCycles:
                # print('查找划分点超时')
                return CRTree
        splitValue = mean([sv1, sv2])
        leftdata, rightdata = splitData(data, splitAttribute, splitValue)
        CRTree.set_leftChild(generateTree(leftdata, rootlabel))
        CRTree.set_rightChild(generateTree(rightdata, rootlabel))
        return CRTree


# 调用函数
def CRT(data):
    try:
        numberSample, numberAttribute = data.shape
    except ValueError:
        return None
    orderAttribute = arange(numberSample).reshape(numberSample, 1)
    data = hstack((data, orderAttribute))
    completeRandomTree = generateTree(data)
    return completeRandomTree


# 返回两行N列的矩阵，第一行是样本标签，第二行是判断噪声阈值
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


# 返回一个序列最近两次变化之间的个数
def checkLabelSequence(labels):
    index1 = 0
    for i in range(1, len(labels)):
        if labels[index1] != labels[i]:
            index1 = i
            break
    if index1 == 0:
        return 0

    index2 = 0
    for i in range(index1 + 1, len(labels)):
        if labels[index1] != labels[i]:
            index2 = i
            break
    if index2 == 0:
        index2 = len(labels)
    return index2 - index1


# 返回是否是噪声数据的序列——树
def filterNoise(data, tree=None, niThreshold=3):
    if tree == None:
        tree = CRT(data)
    visiTree = visitCRT(tree)
    visiTree = visiTree[:, argsort(visiTree[0, :])]
    for i in range(len(visiTree[0, :])):
        if visiTree[1, i] >= niThreshold:  # 是噪声
            visiTree[1, i] = 1
        else:
            visiTree[1, i] = 0
    return visiTree[1, :]


# 返回是否是噪声数据的序列——森林
def CRFNFL(data, ntree=100, niThreshold=3):
    m, n = data.shape
    result = zeros((m, ntree))
    for i in range(ntree):
        visiTree = filterNoise(data, niThreshold=niThreshold)
        result[:, i] = visiTree

    noiseData = []
    for i in result:
        if sum(i) >= 0.5 * ntree:
            noiseData.append(1)
        else:
            noiseData.append(0)

    return array(noiseData)


# 删除异常数据
def deleteNoiseData(data, noiseOrder):
    flag = 0;
    for i in range(noiseOrder.size):
        if noiseOrder[i] == 0:
            if flag == 0:
                redata = data[i, :]
                flag = 1
            else:
                redata = vstack((redata, data[i, :]))
    return redata

'''
标Func的是未去噪方法
'''
def kNNFunc(traindata, testdata):
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]
    model = kNN(n_neighbors=3, algorithm='brute')
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)
    return precision


def bpNNFunc(traindata=None, testdata=None):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    # 如果标签种类为[-1 1]——>[0 1]
    for i in range(traindatalabel.size):
        if traindatalabel[i] == -1:
            traindatalabel[i] = 0
    for i in range(testdatalabel.size):
        if testdatalabel[i] == -1:
            testdatalabel[i] = 0

    # Creata a model
    model = Sequential()
    # 添加输入层、隐藏层的连接
    # 添加的是 Dense 全连接神经层,参数有两个，一个是输入数据和输出数据的维度
    # 如果需要添加下一个神经层的时候，不用再定义输入的纬度，因为它默认就把前一层的输出作为当前层的输入
    model.add(Dense(20, input_dim=numberAttributes - 1, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # 防止过拟合，以一定的概率让某些神经元不起作用

    model.add(Dense(10, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # 防止过拟合，以一定的概率让某些神经元不起作用

    model.add(Dense(1, init='uniform'))  # 以Sigmoid函数为激活函数
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))  # 防止过拟合，以一定的概率让某些神经元不起作用

    # 编译模型，损失函数为，用adam法求解
    model.compile(loss='binary_crossentropy',  # binary_crossentropy: logloss(对数损失)
                  optimizer='adam',  # optimizers: 这个是用来选用优化方法的:adam(梯度下降算法) sgd
                  metrics=['accuracy'])  # metrics：模型评估（精度） accuracy

    # 默认保存在当前目录
    # model.save('my_model.h5')

    # 开始训练
    model.fit(traindata, traindatalabel, epochs=10, batch_size=32, verbose=1)

    # 返回预测精度三种方法
    # 方法1：evaluata()
    precision = model.evaluate(testdata, testdatalabel)
    precision = precision[1]

    ##方法2：predict()
    # prelabel=model.predict(testdata)
    # prelabel2=[round_(i) for i in prelabel]
    # precision2=0
    # for i in range(len(prelabel2)):
    #    if prelabel2[i]==testdatalabel[i]:
    #        precision2+=1
    # precision2=precision2/len(prelabel2)

    ##方法3：predict_classes()
    # prelabel3=model.predict_classes(testdata)
    # precision3=0
    # for i in range(len(prelabel3)):
    #    if prelabel3[i]==testdatalabel[i]:
    #        precision3+=1
    # precision3=precision3/len(prelabel3)

    # print('precision1,precision2,precision3=:',precision,precision2,precision3)
    return precision


def svmFunc(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = svm.SVC(kernel='rbf')  # linear
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision


def cartFunc(traindata, testdata):
    # from sklearn.ensemble import RandomForestClassifier#分类
    # from sklearn.ensemble import RandomForestRegressor#回归
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = RandomForestClassifier(n_estimators=1)  # 一棵树的情况
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision


def lrFunc(traindata, testdata):
    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = LogisticRegression()  # KNN,k=3
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision

def gbdtFunc(traindata, testdata):

    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = GradientBoostingClassifier()
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision

def xgbFunc(traindata, testdata):

    try:
        m, numberAttributes = traindata.shape
    except ValueError as e:
        print(str(e))
        return -1
    # load data
    traindatalabel = traindata[:, 0]
    traindata = traindata[:, 1:]
    testdatalabel = testdata[:, 0]
    testdata = testdata[:, 1:]

    model = XGBClassifier()
    model.fit(traindata, traindatalabel)
    precision = model.score(testdata, testdatalabel)  # 预测精度
    return precision


'''
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
'''
# 结合噪声检测的分类算法：
def CRFNFL_kNN(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # # 验证精度
            preTemp = kNNFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = kNNFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print("preTemp:",preTemp)
            if precision < preTemp:
                precision = preTemp

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = kNNFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = kNNFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print("preTemp:",preTemp)
                if precision < preTemp:
                    precision = preTemp

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = kNNFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = kNNFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

    return precision


def CRFNFL_BPNN(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata:',denoiseTraindata.shape)
            # 验证精度
            preTemp = bpNNFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = bpNNFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print("preTemp:",preTemp)
            if precision < preTemp:
                precision = preTemp

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata:',denoiseTraindata.shape)
                # 验证精度
                preTemp = bpNNFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = bpNNFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print("preTemp:",preTemp)
                if precision < preTemp:
                    precision = preTemp

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = bpNNFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = bpNNFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

    return precision



def CRFNFL_SVM(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest);
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # # 验证精度
            preTemp = svmFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = svmFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print('preTemp',preTemp)
            if precision < preTemp:
                precision = preTemp

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # print('subNtree:',subNtree)
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = svmFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = svmFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # # 验证精度
                preTemp = svmFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = svmFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

    return precision


def CRFNFL_LR(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # # 验证精度
            preTemp = lrFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = lrFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print('preTemp',preTemp)
            if precision < preTemp:
                precision = preTemp

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = lrFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = lrFunc(denoiseTraindata, Validationdata)
                # print('preTemp',preTemp)
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

    return precision


def CRFNFL_GBDT(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    best_niThreshold=0
    bset_subNtree = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # 验证精度
            preTemp = gbdtFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            if precision < preTemp:
                precision = preTemp
                best_niThreshold = subNi
                bset_subNtree = ntree

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = gbdtFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp
                    best_niThreshold = subNi
                    bset_subNtree = subNtree

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = gbdtFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp
                    best_niThreshold = subNi

    return precision,bset_subNtree,best_niThreshold

def CRFNFL_XGB(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(str(e))
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    best_niThreshold=0
    bset_subNtree = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # 验证精度
            preTemp = gbdtFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = lrFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print('preTemp',preTemp)
            if precision < preTemp:
                precision = preTemp
                best_niThreshold = subNi
                bset_subNtree = ntree

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = xgbFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp
                    best_niThreshold = subNi
                    bset_subNtree = subNtree

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = xgbFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp
                    best_niThreshold = subNi

    return precision,bset_subNtree,best_niThreshold

def CRFNFL_Cart(traindata, Validationdata, testdata, ntree=100, niThreshold=11):
    # 建立ntree棵树
    if ntree < 1:
        print('The value of ntree at least is 1.')
        return 0

    try:
        m, n = traindata.shape
    except ValueError as e:
        print(e)
        return 0

    forest = array([])  # m行ntree列的矩阵
    for i in range(ntree):
        tree = CRT(traindata)
        visiTree = visitCRT(tree)
        visiTree = visiTree[:, argsort(visiTree[0, :])]
        visiTree = visiTree[1, :]
        if forest.size == 0:
            forest = visiTree.reshape(m, 1)
        else:
            forest = hstack((forest, visiTree.reshape(m, 1)))

    precision = 0
    if ntree < 10:  # 森林规模小于10，只讨论niThreshold的变化情况下的最优精度
        print('森林规模小于10，只讨论niThreshold的变化情况下的最优精度。')
        for subNi in range(2, niThreshold + 1):
            # sub niThreshold = i
            # 用不同的niThreshold检测噪声数据，更新最优精度
            noiseForest = zeros(m)  # 保存噪声检测结果
            # 开始遍历forest矩阵
            for j in range(m):  # 一个数据的检测过程
                for k in range(ntree):
                    if forest[j, k] >= subNi:
                        noiseForest[j] += 1;

                if noiseForest[j] >= 0.5 * ntree:  # votes
                    noiseForest[j] = 1
                else:
                    noiseForest[j] = 0;

            denoiseTraindata = deleteNoiseData(traindata, noiseForest)
            # print('denoiseTraindata.shape',denoiseTraindata.shape)
            # 验证精度
            preTemp = cartFunc(denoiseTraindata, Validationdata)  # 调用系统基本分类算法
            # 测试精度
            preTest = lrFunc(denoiseTraindata, testdata)
            # print("preTest:",preTest)
            # print('preTemp',preTemp)
            if precision < preTemp:
                precision = preTemp

    else:  # 森林规模大于10，每10棵树作为一个间隔遍历不同规模下的精度
        startNtree = 1;
        endNtree = ntree // 10;
        remainderNtree = ntree % 10;  # 如果有余数，最后在用所有树更新一次精度
        # 每10棵树作为一个间隔遍历不同规模下的精度
        for i in range(startNtree, endNtree + 1):
            subNtree = i * 10
            # 将subForest 作为森林进行不同niThreshold的遍历
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(subNtree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1;

                    if noiseForest[j] >= 0.5 * subNtree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = cartFunc(denoiseTraindata, Validationdata)  # 基本分类器分类结果
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

        if remainderNtree > 0:
            # print('remainderNtree:',remainderNtree)
            for subNi in range(2, niThreshold + 1):
                noiseForest = zeros(m)
                for j in range(m):
                    for k in range(ntree):
                        if forest[j, k] >= subNi:
                            noiseForest[j] += 1

                    if noiseForest[j] >= 0.5 * ntree:
                        noiseForest[j] = 1
                    else:
                        noiseForest[j] = 0

                denoiseTraindata = deleteNoiseData(traindata, noiseForest)
                # print('denoiseTraindata.shape',denoiseTraindata.shape)
                # 验证精度
                preTemp = cartFunc(denoiseTraindata, Validationdata)
                # 测试精度
                preTest = lrFunc(denoiseTraindata, testdata)
                # print("preTest:",preTest)
                # print('preTemp',preTemp)
                # print('preTemp',preTemp)
                if precision < preTemp:
                    precision = preTemp

    return precision

'''''
接口
'''''
def crfnfl_all(traindata, testdata, Validationdata):
    m, n = traindata.shape
    print('traindata:', m, n)
    m, n = testdata.shape
    print('testdata:', m, n)
    print("原始精度")
    pre1 = kNNFunc(traindata, testdata)
    pre2 = bpNNFunc(traindata, testdata)
    pre3 = svmFunc(traindata, testdata)
    pre4 = lrFunc(traindata, testdata)
    pre5 = cartFunc(traindata, testdata)
    gbdt_pre0 = gbdtFunc(traindata, testdata)
    xgb_pre0 = xgbFunc(traindata, testdata)

    print("knn = :", pre1)
    print("bpnn = :", pre2)
    print("svm = :", pre3)
    print("lr = :", pre4)
    print("cart = :", pre5)
    print("gbdt = :", gbdt_pre0)
    print("xgboost = :", xgb_pre0)

    print("去噪后精度")
    pre6 = CRFNFL_kNN(traindata, Validationdata, ntree=23, niThreshold=6)
    pre7 = CRFNFL_BPNN(traindata, Validationdata, ntree=5, niThreshold=2)
    pre8 = CRFNFL_SVM(traindata, Validationdata, ntree=24, niThreshold=6)
    pre9 = CRFNFL_LR(traindata, Validationdata, ntree=23, niThreshold=6)
    pre10 = CRFNFL_Cart(traindata, Validationdata, ntree=32, niThreshold=6)
    gbdt_pre1 = CRFNFL_GBDT(traindata, Validationdata)
    xgb_pre1 = CRFNFL_XGB(traindata, Validationdata)

    print("knn = :", pre1)
    print("bpnn = :", pre2)
    print("svm = :", pre3)
    print("lr = :", pre4)
    print("cart = :", pre5)
    print("gbdt = :", gbdt_pre1)
    print("xgb = :", xgb_pre1)

    return pre1, pre2, pre3, pre4, pre5, gbdt_pre0, xgb_pre0, pre6, pre7, pre8, pre9, pre10, gbdt_pre1, xgb_pre1





