import cython
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from numpy import loadtxt
import time
import sys
import itertools
from technical_indicators import technical_indicators as tec
import multiprocessing



totalStart = time.time()
sys.setrecursionlimit(30000000)

ask = np.loadtxt('option_data345.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )

ask2 = np.loadtxt('5Years.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )





patFound = 0
average = 0
patternPrice = 0
PredictionLag = 60
patternSize = 240



def percentChange(startPoint,currentPoint):
    try:
        x = ((float(currentPoint)-startPoint)/abs(startPoint))*100.00
        if x == 0.0:
            return 0.000000001
        else:
            return x
    except:
        return 0.0001


def patternStorage():
    '''
    The goal of patternFinder is to begin collection of %change patterns
    in the tick data. From there, we also collect the short-term outcome
    of this pattern. Later on, the length of the pattern, how far out we
    look to compare to, and the length of the compared range be changed,
    and even THAT can be machine learned to find the best of all 3 by
    comparing success rates.'''
    threshold2 = 0.0045
    startTime = time.time()
    
    
    #x = len(avgLine)-30
    x = len(avgLineFull)-patternSize
    y = 61
    currentStance = 'none'
    p1 = 0
    while y < x:
        pattern = []
        for i in xrange(patternSize):
            p1 = percentChange(avgLineFull[y-patternSize], avgLineFull[y-patternSize-(i+1)])
            pattern.append(p1)

                



        #outcomeRange = avgLineFull[y+20:y+30]
        #currentPoint = avgLineFull[y]
        outcomeRange = avgLineFull[y+25:y+35]
        currentPoint = avgLineFull[y]


        try:
            avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
        except Exception, e:
            print str(e)
            avgOutcome = 0
        #futureOutcome = percentChange(currentPoint, avgOutcome)
        futureOutcome = float(avgLineFull[y+PredictionLag])-currentPoint

        '''
        print 'where we are historically:',currentPoint
        print 'soft outcome of the horizon:',avgOutcome
        print 'This pattern brings a future change of:',futureOutcome
        print '_______'
        print p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
        '''
        if ( abs(futureOutcome) > threshold2):
       

            patternAr.append(pattern)
            performanceAr.append(futureOutcome)
        
        y+=1

    endTime = time.time()
   # print len(patternAr)
   # print len(performanceAr)
    print 'Pattern storing took:', endTime-startTime


def patternOptimizer(patternArray, performArray):
   # global d
    global counter
    global ThresholdFailure
    ThresholdFailure = 5
    global PercentFailure
    counter = 0
    d = {}
    matches = []
    matchThreshhold  = 0.0003
    targetMatchCount = 200
    global indexCurrent
    global originLength
    originLength = len(patternArray)
    print('pattern length')
    print(originLength)
    global IndexMatch
    global matchFound
    matchFound = False
    indexCurrent = 0
    if (len(patternArray) == 0):
        # the recursion did not complete, so we keep storing optimal patterns into array and then recurse through list of updated performArrayray and pattern arrays
        return False 
    for eachPattern in patternArray:   
        indexMatch = 0
        NumofMatch = 0
        Success= 0
        Failure = 0
        matches = []
        matchings = []
        for otherPattern in patternArray[indexCurrent+1:]:
            sim1 = 100.00 - abs(percentChange(eachPattern[0], otherPattern[0]))
            sim2 = 100.00 - abs(percentChange(eachPattern[1], otherPattern[1]))
            sim3 = 100.00 - abs(percentChange(eachPattern[2], otherPattern[2]))
            sim4 = 100.00 - abs(percentChange(eachPattern[3], otherPattern[3]))
            sim5 = 100.00 - abs(percentChange(eachPattern[4], otherPattern[4]))
            sim6 = 100.00 - abs(percentChange(eachPattern[5], otherPattern[5]))
            sim7 = 100.00 - abs(percentChange(eachPattern[6], otherPattern[6]))
            sim8 = 100.00 - abs(percentChange(eachPattern[7], otherPattern[7]))
            sim9 = 100.00 - abs(percentChange(eachPattern[8], otherPattern[8]))
            sim10 = 100.00 - abs(percentChange(eachPattern[9], otherPattern[9]))
            
            sim11 = 100.00 - abs(percentChange(eachPattern[10], otherPattern[10]))
            sim12 = 100.00 - abs(percentChange(eachPattern[11], otherPattern[11]))
            sim13 = 100.00 - abs(percentChange(eachPattern[12], otherPattern[12]))
            sim14 = 100.00 - abs(percentChange(eachPattern[13], otherPattern[13]))
            sim15 = 100.00 - abs(percentChange(eachPattern[14], otherPattern[14]))
            sim16 = 100.00 - abs(percentChange(eachPattern[15], otherPattern[15]))
            sim17 = 100.00 - abs(percentChange(eachPattern[16], otherPattern[16]))
            sim18 = 100.00 - abs(percentChange(eachPattern[17], otherPattern[17]))
            sim19 = 100.00 - abs(percentChange(eachPattern[18], otherPattern[18]))
            sim20 = 100.00 - abs(percentChange(eachPattern[19], otherPattern[19]))
            
            sim21 = 100.00 - abs(percentChange(eachPattern[20], otherPattern[20]))
            sim22 = 100.00 - abs(percentChange(eachPattern[21], otherPattern[21]))
            sim23 = 100.00 - abs(percentChange(eachPattern[22], otherPattern[22]))
            sim24 = 100.00 - abs(percentChange(eachPattern[23], otherPattern[23]))
            sim25 = 100.00 - abs(percentChange(eachPattern[24], otherPattern[24]))
            sim26 = 100.00 - abs(percentChange(eachPattern[25], otherPattern[25]))
            sim27 = 100.00 - abs(percentChange(eachPattern[26], otherPattern[26]))
            sim28 = 100.00 - abs(percentChange(eachPattern[27], otherPattern[27]))
            sim29 = 100.00 - abs(percentChange(eachPattern[28], otherPattern[28]))
            sim30 = 100.00 - abs(percentChange(eachPattern[29], otherPattern[29]))

            sim31 = 100.00 - abs(percentChange(eachPattern[30], otherPattern[30]))
            sim32 = 100.00 - abs(percentChange(eachPattern[31], otherPattern[31]))
            sim33 = 100.00 - abs(percentChange(eachPattern[32], otherPattern[32]))
            sim34 = 100.00 - abs(percentChange(eachPattern[33], otherPattern[33]))
            sim35 = 100.00 - abs(percentChange(eachPattern[34], otherPattern[34]))
            sim36 = 100.00 - abs(percentChange(eachPattern[35], otherPattern[35]))
            sim37 = 100.00 - abs(percentChange(eachPattern[36], otherPattern[36]))
            sim38 = 100.00 - abs(percentChange(eachPattern[37], otherPattern[37]))
            sim39 = 100.00 - abs(percentChange(eachPattern[38], otherPattern[38]))
            sim40 = 100.00 - abs(percentChange(eachPattern[39], otherPattern[39]))
            
            sim41 = 100.00 - abs(percentChange(eachPattern[40], otherPattern[40]))
            sim42 = 100.00 - abs(percentChange(eachPattern[41], otherPattern[41]))
            sim43 = 100.00 - abs(percentChange(eachPattern[42], otherPattern[42]))
            sim44 = 100.00 - abs(percentChange(eachPattern[43], otherPattern[43]))
            sim45 = 100.00 - abs(percentChange(eachPattern[44], otherPattern[44]))
            sim46 = 100.00 - abs(percentChange(eachPattern[45], otherPattern[45]))
            sim47 = 100.00 - abs(percentChange(eachPattern[46], otherPattern[46]))
            sim48 = 100.00 - abs(percentChange(eachPattern[47], otherPattern[47]))
            sim49 = 100.00 - abs(percentChange(eachPattern[48], otherPattern[48]))
            sim50 = 100.00 - abs(percentChange(eachPattern[49], otherPattern[49]))
        
            sim51 = 100.00 - abs(percentChange(eachPattern[50], otherPattern[50]))
            sim52 = 100.00 - abs(percentChange(eachPattern[51], otherPattern[51]))
            sim53 = 100.00 - abs(percentChange(eachPattern[52], otherPattern[52]))
            sim54 = 100.00 - abs(percentChange(eachPattern[53], otherPattern[53]))
            sim55 = 100.00 - abs(percentChange(eachPattern[54], otherPattern[54]))
            sim56 = 100.00 - abs(percentChange(eachPattern[55], otherPattern[55]))
            sim57 = 100.00 - abs(percentChange(eachPattern[56], otherPattern[56]))
            sim58 = 100.00 - abs(percentChange(eachPattern[57], otherPattern[57]))
            sim59 = 100.00 - abs(percentChange(eachPattern[58], otherPattern[58]))
            sim60 = 100.00 - abs(percentChange(eachPattern[59], otherPattern[59]))

            howSim = (sim1+sim2+sim3+sim4+sim5+sim6+sim7+sim8+sim9+sim10
                  +sim11+sim12+sim13+sim14+sim15+sim16+sim17+sim18+sim19+sim20
                  +sim21+sim22+sim23+sim24+sim25+sim26+sim27+sim28+sim29+sim30+sim31+sim32+sim33+sim34+sim35+sim36+sim37+sim38+sim39+sim40
                  +sim41+sim42+sim43+sim44+sim45+sim46+sim47+sim48+sim49+sim50
                  +sim51+sim52+sim53+sim54+sim55+sim56+sim57+sim58+sim59+sim60)/patternSize

            if howSim > 75:
                 matches.append(otherPattern)
                 matchings.append(performArray[indexMatch])
                 # found a matching pattern, now we compare future value of this pattern with the similar one
     
               
                 futureValueofCurrent = performArray[indexCurrent]
                 futureValueofMatch = performArray[indexMatch]
                 #remove B from pattern and future outcome list to avoid further comparison of B with other elements in loop since B is similar to A and we do not need to compare B with rest of loop
                 # but when we remove B we fuck the order and when we try and get the index of 
                
                 # next we check whether the future matches
                 if futureValueofCurrent * futureValueofMatch > 0:
                      Success = Success + 1
                      NumofMatch = NumofMatch + 1
                 else:
                   Failure =Failure + 1
                   NumofMatch = NumofMatch + 1

                 PercentFailure=(Failure/NumofMatch)*100

                 if NumofMatch>targetMatchCount and PercentFailure > ThresholdFailure:
                    del patternArray[indexCurrent]    #removed  all matches
                    print(performArray[indexCurrent])
                    del performArray[indexCurrent]    #removed A performance
                    patternArray = [x for x in patternArray if x not in matches]
                    #performArray = [x for x in performArray if x not in matchings]
                    #patternArray = np.delete(patternArray, matches, axis=1)
                    performArray = np.delete(performArray, matchings).tolist()
                    #patternArray.tolist()
                   # performArray.tolist()
                    print('removed')
                    return patternOptimizer(patternArray, performArray)
                   
                       
                 elif NumofMatch>targetMatchCount and PercentFailure < ThresholdFailure:
                    optimalPatterns.append(patternArray[indexCurrent])     # added A to list of best patterns
                    updatedPerformArray.append(performArray[indexCurrent]) # added A to list of best outcomes
                    del patternArray[indexCurrent]    #removed  A pattern
                    print(performArray[indexCurrent])
                    del performArray[indexCurrent]    #removed A performance
                    arraylen = len(patternArray)
                    patternArray = [x for x in patternArray if x not in matches]
                   # performArray = [x for x in performArray if x not in matchings]
                    #patternArray = np.delete(patternArray, matches, axis=1)
                    performArray = np.delete(performArray, matchings).tolist()
                   # patternArray.tolist()
                    #performArray.tolist()
                   # patternArray = patternArray.reshape(arraylen,60)
                   # performArray = patternArray.reshape(arraylen,60)
                    print('success')
                    return patternOptimizer(patternArray, performArray)
                   
                     
          
            indexMatch = indexMatch + 1
                      
        counter = 0
        indexCurrent = indexCurrent+1
        
        
                 

def currentPattern():

    global patternPrice 
    mostRecentPoint = avgLineFull[-1]

    
    #for i in xrange(patternSize):
    cp1 = np.array(avgLine[-patternSize:], dtype='float32')
    a = np.empty(patternSize)
    a.fill(avgLine[-1-patternSize])
    #array1 = np.array([-5, -5, -5, -5, -5], dtype='float32')
    #array2 = np.array([2, 2, 2, 2, 2], dtype='float32')
    diff = ((a - cp1) / abs(a))*100
    #cp1 = percentChange(avgLine[-patternSize-1],avgLine[-patternSize+i])
    #patForRec.append(cp1)
    return diff






def graphRawFX():
    
    fig=plt.figure(figsize=(10,7))
    ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
    ax1.plot(ask)
    ax1.plot(ask)
   # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.grid(True)
    for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    ax1_2 = ax1.twinx()
    #ax1_2.fill_between(date, 0, (ask-ask), facecolor='g',alpha=.3)

    plt.subplots_adjust(bottom=.23)
    plt.show()



    
def patternRecognition():

    plotPatAr = []
    patFound = 0
    global average
    global patFound
    global realMovement
  
    simArray = []
    for eachPattern in patternAr:
        #for i in xrange(patternSize):
        ar1 = np.array(eachPattern, dtype='float32')
        ar2 = np.array(patForRec, dtype='float32')
        diff = 100 - (((ar1 - ar2) / abs(ar1)) * 100)
        #sim1 = 100.00 - abs(percentChange(eachPattern[i], patForRec[i]))
        #simArray.append(sim1)
        howSim = np.average(diff)
        if howSim > 80:
            
            patdex = patternAr.index(eachPattern)
            patFound = 1
           # print('Pattern Found')
            #xp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
            #############

            plotPatAr.append(eachPattern)



    if patFound == 1:
        fig = plt.figure(figsize=(10,6))
        futureP = []

        #print("to what")
        #print(toWhat)
        
        for eachPatt in plotPatAr:
            futurePoints = patternAr.index(eachPatt)
            futureP.append(performanceAr[futurePoints])
            if performanceAr[futurePoints] > patForRec[9]:
                pcolor = '#24bc00'
            else:
                pcolor = '#d40000'
                #pcolor = '#000000'
            
            #plt.plot(xp, eachPatt)
            ####################
            plt.scatter(35, performanceAr[futurePoints],c=pcolor,alpha=.4)
        realOutcomeRange = allData[toWhat+25:toWhat+35]
        realAvgOutcome = reduce(lambda x, y: x + y, realOutcomeRange) / len(realOutcomeRange)
        average =  np.average(futureP)

        realMovement = float(allData[toWhat+PredictionLag])-allData[toWhat]
        rMove = realMovement
        #print('the real movement')
        #print(realMovement)
        plt.scatter(40, realMovement, c='#54fff7',s=25)
        #plt.plot(xp, patForRec, '#54fff7', linewidth = 3)
        plt.grid(True)
        plt.title('Pattern Recognition.\nCyan line is the current pattern. Other lines are similar patterns from the past.\nPredicted outcomes are color-coded to reflect a positive or negative prediction.\nThe Cyan dot marks where the pattern went.\nOnly data in the past is used to generate patterns and predictions.')
        #plt.show()

            
            

array1 =np.array([5, 5, 5, 5, 5], dtype='float32')
array2 = np.array([-2, -2 ,- 2 ,-2 ,-2])
diff = ((array1-array2) / array2)
print(diff)

dataLength = int(ask.shape[0])
print 'data length is', dataLength

allData = ((ask+ask)/2)
allData = allData[240:]
avgLineFull = ((ask2+ask2)/2)
avgLineFull = avgLineFull[:300000]

#toWhat = 53500
toWhat = 3000
threshold = 0
win = 0
loss = 0
numTrades = 0
patternAr = []
performanceAr = []
global patForRec
updatedPerformArray = []
optimalPatterns = []
patternStorage()
print('optimization complete')
print('size of original pattern array')
print(len(patternAr))
#patternOptimizer(patternAr,performanceAr)
#print('size of optimized pattern array')
#print(len(optimalPatterns))
#print('size of optimized performance array')
#print(len(updatedPerformArray))
errorList = []
#patternAr = optimalPatterns
#performanceAr = updatedPerformArray

#while toWhat < dataLength:
for x in xrange(1000):
    avgLine = ((ask+ask)/2)
    avgLine = avgLine[:toWhat]
    rsi = tec.rsi(np.array(avgLine[-patternSize:]))

    
    

    

    #avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
    
    
  
    patForRec = currentPattern()
    patternRecognition()
    #print('pattern actual price')
    #print(patternPrice)
    #print('runnign to what')
    #print(toWhat)
    #lastPrice = ask[toWhat]
    #futurePrice = ask[toWhat+30]
    #prcChange = percentChange(lastPrice,futurePrice)

    #print('our movement is')
    #print(prcChange)


    
    if patFound == 1 and abs(average) > threshold and rsi[-1] < 30:
        if average> 0 and ask[toWhat+PredictionLag] - ask[toWhat] > 0:
            win = win + 1
            print('trade won !!!')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
        else:
            loss = loss + 1
            print('trade Lost on up prediction:(')
            numTrades = numTrades + 1
            error = abs(ask[toWhat + PredictionLag] - ask[toWhat] - average)
            errorList.append(error)
    elif patFound == 1 and abs(average) > threshold and rsi[-1] > 70:
        if average < 0 and ask[toWhat+PredictionLag] - ask[toWhat] < 0:
            win = win + 1
            print('trade won !!!')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
        else:
            loss = loss + 1
            print('trade Lost on down prediction:(')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)

            # get the last price, get the future price. find percentage change and compare to predicted
                                     
   
    
    totalEnd = time.time()-totalStart
    print 'Entire processing took:',totalEnd,'seconds'
    toWhat += 1
print('number of trades:')
print(numTrades)
print('accuracy is')
print(100*float(win)/float(numTrades))
print('max error for win is')
print(np.amax(errorList))
    
