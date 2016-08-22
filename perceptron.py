import sys
import random
import math


def readData(data):
    f1 = open(data)
    data = []
    line1 = f1.readline()
    i = 0
    while (line1 != ''):
        splitLine = line1.split()
        tempLine = []
        for j in range(0, len(splitLine), 1):
            tempLine.append(int(splitLine[j]))
        tempLine.append(1)
        data.append(tempLine)
        line1 = f1.readline()
    f1.close()
    return data


def readLabels(trainlabels, length):
    classification = [None] * length
    f2 = open(trainlabels)
    dataRow = f2.readline()
    while (dataRow != ''):
        splitLine = convertElementsToInt(dataRow.split())
        classification[splitLine[1]] = splitLine[0]
        if classification[splitLine[1]] == 0:
            classification[splitLine[1]] = -1
        dataRow = f2.readline()
    f2.close()
    return classification


def convertElementsToInt(dataRow):
    temp = []
    for i in range(0, len(dataRow), 1):
        temp.append(int(dataRow[i]))
    return temp


def dotProduct(weights, dataRow, cols):
    dp = 0
    for j in range(0, cols, 1):
        dp += weights[j] * dataRow[j]
    return dp


def initWeights(cols):
    weights = [None] * cols
    for j in range(0, cols, 1):
        weights[j] = .02 * random.random() - 0.01
    return weights


def computeDellF(classification, data, weights, cols):
    ## Gradient descent iteration
    eta = 0.001
    prev_error = 0
    for k in range(0, 100000, 1):
        ## Compute dellF
        dellf = [0] * (cols)
        for i in range(0, len(data), 1):
            if (classification[i] is not None):
                dp = dotProduct(weights, data[i], cols)
                for j in range(0, cols, 1):
                    dellf[j] += (classification[i] - dp) * data[i][j]
        for j in range(0, cols, 1):
            weights[j] += eta * dellf[j]
        error = 0
        for i in range(0, len(data), 1):
            if (classification[i] is not None):
                error += (classification[i] - dotProduct(weights, data[i], cols)) ** 2
        if (abs(prev_error - error) < 0.001):  ## Stopping Condition
            print "finished"
            break
        prev_error = error
        print "error = ", error
    return weights


def printNormalizedWeight(weights, cols):
    print "w = ",
    normw = 0
    for j in range(0, cols - 1, 1):
        normw += (weights[j]) ** 2
        print weights[j],
    normw = math.sqrt(normw)
    print "\n||w|| = ", normw
    d_origin = abs(weights[len(weights) - 1] / normw)
    print "distance to origin = ", d_origin
    return 0


def predict(classification, weights, data, rows):
    f3 = open('output', 'w')
    for i in range(0, rows, 1):
        if (classification[i] is None):
            dp = dot_product(weights, data[i])
            if (dp > 0):
                f3.write("1 " + str(i) + "\n")
            else:
                f3.write("0 " + str(i) + "\n")
    print "Prediction complete"
    return 0


def main(data, trainLabels):
    dataList = readData(data)
    labels = readLabels(trainLabels, len(dataList))
    cols = len(dataList[0])
    weights = initWeights(cols)
    weights = computeDellF(labels, dataList, weights, cols)
    printNormalizedWeight(weights, cols)


# predict(labels, weights, dataList, len(dataList))

if __name__ == '__main__':
    data = sys.argv[1]
    trainLabels = sys.argv[2]
    main(data, trainLabels)
