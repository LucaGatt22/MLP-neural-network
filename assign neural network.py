import pandas
df = pandas.read_csv('5XOR.csv')
totalNoRows = 32
testingSetNoRows = 6 # 32-26 # 32/5
trainingSetNoRows = 26 # 32/5*4 rounded to nearest integer
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
trainingSet = df.iloc[:26]
testingSet = df.iloc[26:] # identified trainingSet and testingSet

# x = (trainingSet['XOR'][6]) # numpy.int64
# x=int(x) # able to typecast
# # x= (x==0) # numpy.int64 = (int)
# print(x)

# inputs from user
seed = int(input('Input random seed: '))
# seed = 0
import numpy as np
np.random.seed(seed)
hiddenLayerNeuronsNo = int(input('Enter number of hidden layer neurons: '))
# hiddenLayerNeuronsNo = 4


# randomise trainingSet
np.random.shuffle(trainingSet.values) # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows

# random numbers between -1 and 1

# wH = [[-1,0,1,1],
#         [-1,0,1,1],
#         [-1,0,1,1],
#         [-1,0,1,1],
#         [-1,0,1,1] ]
# wO = [-1,0,1,1]

## working with close reference to slide 12 of JA105
# generate wH and wO for first time with random numbers
wH= np.random.uniform(-1, 1, size=(5, hiddenLayerNeuronsNo))
wO= np.random.uniform(-1, 1, size=hiddenLayerNeuronsNo)
# print('wH= '+str(wH)+ '   wO= '+str(wO))
# print(wH)
    # wH changes according to hiddenLayerNeuronsNo

sigmoid = lambda x: 1 / (1 + np.exp(-x))
def feedForward(inp,wH,wO):
    netH = np.matmul(inp,wH)
    # print('netH='+str(netH))
    global outH
    # https://www.geeksforgeeks.org/apply-function-to-each-element-of-a-list-python/
    outH = list(map(sigmoid, netH))
    # print('outH='+str(outH))
    netO = np.matmul(outH,wO)
    # print('netO='+str(netO))
    outO = sigmoid(netO)
    
    return outO
    


badFactsToEpoch = [['Bad Facts','Epoch']]

mu=eta=0.2

epochs = 2000
for epoch in range(epochs):
    badFactsCount=0
    for trainingInputIndex in range(len(trainingSet)):
        trainingInput = [trainingSet['A'][trainingInputIndex],
                            trainingSet['B'][trainingInputIndex],
                            trainingSet['C'][trainingInputIndex],
                            trainingSet['D'][trainingInputIndex],
                            trainingSet['E'][trainingInputIndex] ] # list
        
        # print(trainingInputIndex)
        # print('wH= '+str(wH)+ ' wO= '+str(wO))
        # if trainingInputIndex==6:
        #     wH6 = wH.copy()
        # elif trainingInputIndex == 11:
        #     if (wH[2][0] == wH6[2][0]):
        #         print('weights not changing')
                
        # print('wH6='+str(wH6))
        
        
        out = feedForward(trainingInput, wH, wO) # weights change with every epoch
        # print(out) # float or some numpy.float
        
        target = trainingSet['XOR'][trainingInputIndex]
        error = target - out
        
        if abs(error) > mu:
            # error back propagation
            # output layer weights
            deltaO = out*(1-out)*error
            dW = list(map(lambda x: eta*deltaO*x,outH)) # dW and wO are (1,4)
            wO = np.add(wO,dW)
            # print('wO= '+str(wO))
            
            # hidden layer weights
            deltaH = np.zeros(hiddenLayerNeuronsNo)
            # print('outH= '+str(outH)+ '\ndeltaO= '+str(deltaO) + '\nwO= '+str(wO))
            for i in range(hiddenLayerNeuronsNo):
                deltaH[i] = outH[i]*(1-outH[i])*deltaO*wO[i]
            # print(deltaH) # (1,hiddenLayerNeuronsNo)
            
            dW = np.zeros((5,hiddenLayerNeuronsNo))
            for i in range(5):
                for j in range(hiddenLayerNeuronsNo):
                    dW[i][j] = eta * deltaH[j] * trainingInput[i]
            wH = np.add(wH, dW)
            
            # code from here onwards copied from my first attempt
            # bad fact count
            badFactsCount+=1
            
    badFactsToEpoch.append([badFactsCount, epoch]) # add to dataset for graph
    if badFactsCount==0:
        print("Training completed since all are good facts")
        
        break # terminate training

# # bad facts to epoch results export to csv
# # to_csv - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
# from pandas import DataFrame
# # df = DataFrame(badFactsToEpoch)
# ## save to xlsx file
# DataFrame(badFactsToEpoch).to_csv('BadFactsVsEpochResults.csv', index=False)


# testing phase
goodFactNumTesting = 0
from numpy import round
for testingIndex in range(len(testingSet['A'])):
    testingInput=testingSet.iloc[testingIndex,0:5]
    testingTarget=testingSet.iloc[testingIndex,5]
    if round(feedForward(testingInput,wH,wO)) == testingTarget: # round is used for the output to be either 0 or 1
        goodFactNumTesting += 1
    # else:
    #     # print("Test failed. :(")
    #     # success = False
    #     pass
        #break # terminate program. Restart it or there is logical error in code

accuracyTesting = goodFactNumTesting/(6) # 6 is length of testingSet
print("Testing accuracy is "+ str(accuracyTesting*100) + "%.")
#if success: print("Testing Completed. The neural network works well. :)")


# plot graph of BadFactsVsEpoch and save it
def plotBadFactsVsEpochGraph():
    import matplotlib.pyplot as plt
    
    # Extracting data
    epochs = [data[1] for data in badFactsToEpoch[1:]]
    bad_facts = [data[0] for data in badFactsToEpoch[1:]]
    
    # Creating a line graph
    plt.plot(epochs, bad_facts, linestyle='-')
    plt.title('Bad Facts vs. Epoch for '+str(hiddenLayerNeuronsNo)+' H.L. neurons and r.seed set to '+str(seed))
    plt.xlabel('Epochs')
    plt.ylabel('Bad Facts')
    plt.grid(True)
    plt.savefig("graphs/BadFactsVsEpochGraph - "+str(hiddenLayerNeuronsNo)+" hidden l. neurons_ r.seed "+str(seed)+".jpg") # savefig() - https://towardsdatascience.com/save-plots-matplotlib-1a16b3432d8a
plotBadFactsVsEpochGraph()

print() # blank line to differentiate between a program run and another