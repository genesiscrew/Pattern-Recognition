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
import multiprocessing
from scipy import signal
import pandas as pd
from numpy import inf
from sklearn import linear_model
from sklearn.cluster import MeanShift, estimate_bandwidth
import trendy




totalStart = time.time()
sys.setrecursionlimit(30000000)

ask = np.loadtxt('5Years.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )
ask = ask[1440000:1700000]

ask2 = np.loadtxt('5Years.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )





#global patFound
average = 0
patternPrice = 0
PredictionLag = 5
patternSize = 60
rsiStack = 0
rsiUp = 0
rsiDown = 0


def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({ 
        'data' : y_sine} ,index=t_sine)
    return result

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def fir_highpass(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y




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
    The goal of patternFinder is to begin collection of %chang e patterns
    in the tick data. From there, we also collect the short-term outcome
    of this pattern. Later on, the length of the pattern, how far out we
    look to compare to, and the length of the compared range be changed,
    and even THAT can be machine learned to find the best of all 3 by
    comparing success rates.'''
    threshold2 = 0.000
    startTime = time.time()
    
    
    #x = len(avgLine)-30
    x = len(avgLineFull)-patternSize
    y = patternSize + 1
    currentStance = 'none'
    p1 = 0
    while y < x:
        pattern = []
        for i in xrange(patternSize):
            p1 = percentChange(avgLineFull[y-patternSize], avgLineFull[y-patternSize-(i+1)])
            pattern.append(p1)

                

        outcomeRange = avgLineFull[y+PredictionLag-5:y+PredictionLag+5]
        try:
            avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
        except Exception, e:
            print str(e)
            avgOutcome = 0
        #outcomeRange = avgLineFull[y+20:y+30]
        #currentPoint = avgLineFull[y]
        
        currentPoint = avgLineFull[y]



        futureOutcome = percentChange(currentPoint, avgOutcome)
        #futureOutcome = avgOutcome-currentPoint
        #futureOutcome = float(avgLineFull[y+PredictionLag])-currentPoint 

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
        
        y+=30

    endTime = time.time()
   # print len(patternAr)
   # print len(performanceAr)
    print 'Pattern storing took:', endTime-startTime

def addPattern(newPattern,  outcomeRange):
    '''
    The goal of patternFinder is to begin collection of %chang e patterns
    in the tick data. From there, we also collect the short-term outcome
    of this pattern. Later on, the length of the pattern, how far out we
    look to compare to, and the length of the compared range be changed,
    and even THAT can be machine learned to find the best of all 3 by
    comparing success rates.'''
    threshold2 = 0.0005
    startTime = time.time()
    
    
    #x = len(avgLine)-30
    x = len(avgLineFull)-patternSize
    y = patternSize + 1
    currentStance = 'none'
   
  
    pattern = []
    for i in xrange(len(newPattern)):
        p1 = percentChange(newPattern[0], newPattern[patternSize-i-1])
        pattern.append(p1)

            
    startPoint = patternSize-1
    #outcomeRange = futureOutcome
    try:
        avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
    except Exception, e:
        print str(e)
        avgOutcome = 0
    #outcomeRange = avgLineFull[y+20:y+30]
    #currentPoint = avgLineFull[y]
    
    currentPoint = newPattern[startPoint]



    futureOutcome = percentChange(currentPoint, avgOutcome)
    #futureOutcome = avgOutcome-currentPoint
    #futureOutcome = float(avgLineFull[y+PredictionLag])-currentPoint 

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
    
    

    endTime = time.time()
   # print len(patternAr)
   # print len(performanceAr)
   # print 'Pattern storing took:', endTime-startTime

def patternOptimizer(patternArray, performArray):
   # global d
    global counter
    global ThresholdFailure
    ThresholdFailure = 5
    global PercentFailure
    counter = 0
    d = {}
    matches = []
    matchThreshhold  = 0.000
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
    diff = ((cp1-a) / abs(a))*100
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
        diff = 100 - abs((((ar1 - ar2) / abs(ar1)) * 100))
        #sim1 = 100.00 - abs(percentChange(eachPattern[i], patForRec[i]))
        #simArray.append(sim1)
        howSim = np.average(diff)
        if howSim > 75:
            
            patdex = patternAr.index(eachPattern)
            patFound = 1
            #print('Pattern Found')
           # xp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
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
            
          #  plt.plot(xp, eachPatt)
            ####################
           
        
        realOutcomeRange = allData[toWhat+PredictionLag-5:toWhat+PredictionLag+5]
        realAvgOutcome = reduce(lambda x, y: x + y, realOutcomeRange) / len(realOutcomeRange)
        average =  np.average(futureP)
        #plt.scatter(35, average,c=pcolor,alpha=.4)
        #realMovement = float(allData[toWhat+PredictionLag])-allData[toWhat]
        realOutcome = percentChange(allData[toWhat],realAvgOutcome)
        #rMove = realMovement
        #print('the real movement')
        #print(realMovement)
        #plt.scatter(40, realOutcome, c='#54fff7',s=25)
       # plt.plot(xp, patForRec, '#54fff7', linewidth = 3)
        #plt.grid(True)
        
        #plt.title('Pattern Recognition.\nCyan line is the current pattern. Other lines are similar patterns from the past.\nPredicted outcomes are color-coded to reflect a positive or negative prediction.\nThe Cyan dot marks where the pattern went.\nOnly data in the past is used to generate patterns and predictions.')
        #plt.show()

def sma(prices, period):
    """
    Simple Moving Average (SMA) are used to smooth the data in an array to help
    eliminate noise and identify trends.
    In SMA, each value in the time period carries equal weight.
    They do not predict price direction, but can be used to identify the
    direction of the trend or define potential support and resistance levels.
    SMA = (P1 + P2 + ... + Pn) / K
    where K = n and Pn is the most recent price
    http://www.financialwebring.org/gummy-stuff/MA-stuff.htm
    http://www.csidata.com/?page_id=797
    http://stockcharts.com/school/doku.php?st=moving+average&id=chart_school:technical_indicators:moving_averages
    Input:
      prices ndarray
      period int > 1 and < len(prices)
    Output:
      smas ndarray
    Test:
    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23,
    ... 22.43, 22.24, 22.29, 22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75,
    ... 23.83, 23.95, 23.63, 23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68,
    ... 23.10, 22.40, 22.17])
    >>> period = 10
    >>> print(tai.sma(prices, period))
    [ 22.221  22.209  22.229  22.259  22.303  22.421  22.613  22.765  22.905
      23.076  23.21   23.377  23.525  23.652  23.71   23.684  23.612  23.505
      23.432  23.277  23.131]
    """
   # period = PredictionLag
    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    sma_range = num_prices - period + 1

    smas = np.zeros(sma_range)

    # only required for the commented code below
    #k = period

    for idx in range(sma_range):
        # this is the code, but using np.mean below is faster and simpler
        #for period_num in range(period):
        #    smas[idx] += prices[idx + period_num]
        #smas[idx] /= k

        smas[idx] = np.mean(prices[idx:idx + period])

    return smas

def rsi(prices, period=14):
    """
    The Relative Strength Index (RSI) is a momentum oscillator.
    It oscillates between 0 and 100.
    It is considered overbought/oversold when it's over 70/below 30.
    Some traders use 80/20 to be on the safe side.
    RSI becomes more accurate as the calculation period (min_periods)
    increases.
    This can be lowered to increase sensitivity or raised to decrease
    sensitivity.
    10-day RSI is more likely to reach overbought or oversold levels than
    20-day RSI. The look-back parameters also depend on a security's
    volatility.
    Like many momentum oscillators, overbought and oversold readings for RSI
    work best when prices move sideways within a range.
    You can also look for divergence with price.
    If the price has new highs/lows, and the RSI hasn't, expect a reversal.
    Signals can also be generated by looking for failure swings and centerline
    crossovers.
    RSI can also be used to identify the general trend.
    The RSI was developed by J. Welles Wilder and was first introduced in his
    article in the June, 1978 issue of Commodities magazine, now known as
    Futures magazine. It is detailed in his book New Concepts In Technical
    Trading Systems.
    http://www.csidata.com/?page_id=797
    http://stockcharts.com/help/doku.php?id=chart_school:technical_indicators:relative_strength_in
    Input:
      prices ndarray
      period int > 1 and < len(prices) (optional and defaults to 14)
    Output:
      rsis ndarray
    Test:
    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([44.55, 44.3, 44.36, 43.82, 44.46, 44.96, 45.23,
    ... 45.56, 45.98, 46.22, 46.03, 46.17, 45.75, 46.42, 46.42, 46.14, 46.17,
    ... 46.55, 46.36, 45.78, 46.35, 46.39, 45.85, 46.59, 45.92, 45.49, 44.16,
    ... 44.31, 44.35, 44.7, 43.55, 42.79, 43.26])
    >>> print(tai.rsi(prices))
    [ 70.02141328  65.77440817  66.01226849  68.95536568  65.88342192
      57.46707948  62.532685    62.86690858  55.64975092  62.07502976
      54.39159393  50.10513101  39.68712141  41.17273382  41.5859395
      45.21224077  37.06939108  32.85768734  37.58081218]
    """
    #period = rsiStack
    #period = 60
    period = 20
    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    # this could be named gains/losses to save time/memory in the future
    changes = prices[1:] - prices[:-1]
    #num_changes = len(changes)

    rsi_range = num_prices - period

    rsis = np.zeros(rsi_range)

    gains = np.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0

    losses = np.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    # convert all negatives into positives
    losses *= -1

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsis[0] = 100
    else:
        rs = avg_gain / avg_loss
        rsis[0] = 100 - (100 / (1 + rs))

    for idx in range(1, rsi_range):
        avg_gain = ((avg_gain * (period - 1) + gains[idx + (period - 1)]) /
                    period)
        avg_loss = ((avg_loss * (period - 1) + losses[idx + (period - 1)]) /
                    period)

        if avg_loss == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain / avg_loss
            rsis[idx] = 100 - (100 / (1 + rs))

    return rsis

def bb(prices, period, num_std_dev=2.0):
    """
    Bollinger bands (BB) are volatility bands placed above and below a moving
    average.
    Volatility is based on the standard deviation, which changes as volatility
    increases and decreases.
    The bands automatically widen when volatility increases and narrow when
    volatility decreases.
    This dynamic nature of Bollinger Bands also means they can be used on
    different securities with the standard settings.
    For signals, Bollinger Bands can be used to identify M-Tops and W-Bottoms
    or to determine the strength of the trend.
    Signals derived from narrowing BandWidth are also important.
    Bollinger BandWidth is an indicator that derives from Bollinger Bands, and
    measures the percentage difference between the upper band and the lower
    band.
    BandWidth decreases as Bollinger Bands narrow and increases as Bollinger
    Bands widen.
    Because Bollinger Bands are based on the standard deviation, falling
    BandWidth reflects decreasing volatility and rising BandWidth reflects
    increasing volatility.
    %B quantifies a security's price relative to the upper and lower Bollinger
    Band. There are six basic relationship levels:
    %B equals 1 when price is at the upper band
    %B equals 0 when price is at the lower band
    %B is above 1 when price is above the upper band
    %B is below 0 when price is below the lower band
    %B is above .50 when price is above the middle band (20-day SMA)
    %B is below .50 when price is below the middle band (20-day SMA)
    They were developed by John Bollinger.
    Bollinger suggests increasing the standard deviation multiplier to 2.1 for
    a 50-period SMA and decreasing the standard deviation multiplier to 1.9 for
    a 10-period SMA.
    http://www.csidata.com/?page_id=797
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_width
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_perce
    Input:
      prices ndarray
      period int > 1 and < len(prices)
      num_std_dev float > 0.0 (optional and defaults to 2.0)
    Output:
      bbs ndarray with upper, middle, lower bands, bandwidth, range and %B
    Test:
    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([86.16, 89.09, 88.78, 90.32, 89.07, 91.15, 89.44,
    ... 89.18, 86.93, 87.68, 86.96, 89.43, 89.32, 88.72, 87.45, 87.26, 89.50,
    ... 87.90, 89.13, 90.70, 92.90, 92.98, 91.80, 92.66, 92.68, 92.30, 92.77,
    ... 92.54, 92.95, 93.20, 91.07, 89.83, 89.74, 90.40, 90.74, 88.02, 88.09,
    ... 88.84, 90.78, 90.54, 91.39, 90.65])
    >>> period = 20
    >>> print(tai.bb(prices, period))
    [[  9.12919107e+01   8.87085000e+01   8.61250893e+01   5.82449423e-02
        5.16682146e+00   6.75671306e-03]
     [  9.19497209e+01   8.90455000e+01   8.61412791e+01   6.52300429e-02
        5.80844179e+00   5.07661263e-01]
     [  9.26132536e+01   8.92400000e+01   8.58667464e+01   7.55995881e-02
        6.74650724e+00   4.31816571e-01]
     [  9.29344497e+01   8.93910000e+01   8.58475503e+01   7.92797873e-02
        7.08689946e+00   6.31086945e-01]
     [  9.33114122e+01   8.95080000e+01   8.57045878e+01   8.49848539e-02
        7.60682430e+00   4.42420124e-01]
     [  9.37270110e+01   8.96885000e+01   8.56499890e+01   9.00563838e-02
        8.07702198e+00   6.80945403e-01]
     [  9.38972812e+01   8.97460000e+01   8.55947188e+01   9.25117832e-02
        8.30256250e+00   4.63143909e-01]
     [  9.42636418e+01   8.99125000e+01   8.55613582e+01   9.67861377e-02
        8.70228361e+00   4.15826692e-01]
     [  9.45630193e+01   9.00805000e+01   8.55979807e+01   9.95225220e-02
        8.96503854e+00   1.48579313e-01]
     [  9.47851634e+01   9.03815000e+01   8.59778366e+01   9.74461225e-02
        8.80732672e+00   1.93266744e-01]
     [  9.50411874e+01   9.06575000e+01   8.62738126e+01   9.67087637e-02
        8.76737475e+00   7.82660026e-02]
     [  9.49062071e+01   9.08630000e+01   8.68197929e+01   8.89956780e-02
        8.08641429e+00   3.22789193e-01]
     [  9.49015375e+01   9.08830000e+01   8.68644625e+01   8.84332063e-02
        8.03707509e+00   3.05526266e-01]
     [  9.48939343e+01   9.09040000e+01   8.69140657e+01   8.77834713e-02
        7.97986867e+00   2.26311285e-01]
     [  9.48594576e+01   9.09880000e+01   8.71165424e+01   8.50982021e-02
        7.74291521e+00   4.30661576e-02]
     [  9.46722663e+01   9.11525000e+01   8.76327337e+01   7.72280810e-02
        7.03953265e+00  -5.29486389e-02]
     [  9.45543042e+01   9.11905000e+01   8.78266958e+01   7.37753219e-02
        6.72760849e+00   2.48722001e-01]
     [  9.46761721e+01   9.11200000e+01   8.75638279e+01   7.80546993e-02
        7.11234420e+00   4.72660054e-02]
     [  9.45733946e+01   9.11670000e+01   8.77606054e+01   7.47286754e-02
        6.81278915e+00   2.01003516e-01]
     [  9.45322396e+01   9.12495000e+01   8.79667604e+01   7.19508503e-02
        6.56547911e+00   4.16304661e-01]
     [  9.45303313e+01   9.12415000e+01   8.79526687e+01   7.20906879e-02
        6.57766250e+00   7.52141243e-01]
     [  9.43672335e+01   9.11660000e+01   8.79647665e+01   7.02286710e-02
        6.40246702e+00   7.83328285e-01]
     [  9.41460689e+01   9.10495000e+01   8.79529311e+01   6.80194599e-02
        6.19313782e+00   6.21182512e-01]]
    """

    num_prices = len(prices)
   

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    bb_range = num_prices - period + 1

    # 3 bands, bandwidth, range and %B
    bbs = np.zeros((bb_range, 6))

    simple_ma = sma(prices, period)

    for idx in range(bb_range):
        std_dev = np.std(prices[idx:idx + period])

        # upper, middle, lower bands, bandwidth, range and %B
        bbs[idx, 0] = simple_ma[idx] + std_dev * num_std_dev
        bbs[idx, 1] = simple_ma[idx]
        bbs[idx, 2] = simple_ma[idx] - std_dev * num_std_dev
        bbs[idx, 3] = (bbs[idx, 0] - bbs[idx, 2]) / bbs[idx, 1]
        bbs[idx, 4] = bbs[idx, 0] - bbs[idx, 2]
        bbs[idx, 5] = (prices[idx] - bbs[idx, 2]) / bbs[idx, 4]

    return bbs


def roc(prices, period=10):
    """
    The Rate-of-Change (ROC) indicator, a.k.a. Momentum, is a pure momentum
    oscillator that measures the percent change in price from one period to the
    next.
    The plot forms an oscillator that fluctuates above and below the zero line
    as the Rate-of-Change moves from positive to negative.
    ROC signals include centerline crossovers, divergences and
    overbought-oversold readings. Identifying overbought or oversold extremes
    comes natural to the Rate-of-Change oscillator.
    It can be used to measure the ROC of any data series, such as price or
    another indicator.
    Also known as PROC when used with price.
    ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    http://www.csidata.com/?page_id=797
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rate_of_change_roc_a
    Input:
      prices ndarray
      period int > 1 and < len(prices) (optional and defaults to 21)
    Output:
      rocs ndarray
    Test:
    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([11045.27, 11167.32, 11008.61, 11151.83, 10926.77,
    ... 10868.12, 10520.32, 10380.43, 10785.14, 10748.26, 10896.91, 10782.95,
    ... 10620.16, 10625.83, 10510.95, 10444.37, 10068.01, 10193.39, 10066.57,
    ... 10043.75])
    >>> print(tai.roc(prices, period=12))
    [-3.84879682 -4.84888048 -4.52064339 -6.34389154 -7.85923013 -6.20834146
     -4.31308173 -3.24341092]
    """

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    roc_range = num_prices - period

    rocs = np.zeros(roc_range)

    for idx in range(roc_range):
        rocs[idx] = ((prices[idx + period] - prices[idx]) / prices[idx]) * 100

    return rocs

fps = 500
sine_fq = 10 #Hz
duration = 10 #seconds
sine_5Hz = sine_generator(fps,sine_fq,duration)
sine_fq = 1 #Hz
duration = 10 #seconds
sine_1Hz = sine_generator(fps,sine_fq,duration)

sine = sine_5Hz + sine_1Hz



#ddddddddddddddddddd

            
            

dataLength = int(ask.shape[0])
print 'data length is', dataLength
startingPoint = 1000

allData = ((ask+ask)/2)
myData = ((ask+ask)/2)
allData = allData[startingPoint :-5]
avgLineFull = ((ask2+ask2)/2)
avgLineFull = avgLineFull[:startingPoint]
fullData = ((ask2+ask2)/2)
#toWhat = 53500
toWhat = startingPoint
threshold = 0.015
win = 0
loss = 0
numTrades = 0
patternAr = []
performanceAr = []
global patForRec
updatedPerformArray = []
optimalPatterns = []
#patternStorage()
pastWinStatus = 0
maxLossCounter = 0
goUp = False
goDown = False
direction = 0
continueUp = False
continueDown = False
counter4 = 0
max_range = 0
max_range2 = 0
plsUp = 0
plsDown = 0
trend_found = 0
counttrend = 0


filtered_sine = butter_highpass_filter(allData,20,fps)
for i in range(len(filtered_sine)):
    filtered_sine[i] = 0.5 * np.log((1+filtered_sine[i])/(1-filtered_sine[i]))

plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('generated signal')
plt.subplot(212)
norm_data = np.histogram(filtered_sine, bins=10, density=True)
plt.hist(filtered_sine,500)
plt.title('filtered signal')
plt.show()


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
bandwidthList = []
totalLossCounter=0
maxLossCounter=0
countRange = 0
rangesum = []
counterela = 0
counterula = 0
lossUp = 0
lossDown = 0
previousDiffMax = 0
previousDiffMin = 0
counterUpTrend = 0
counterDownTrend = 0
totalUpCounts = 0
totalDownCounts = 0
upTrendAv = []
downTrendAv = []
tradeCounter = 0
doTrade = 0
lastMinute = 30

for x in xrange(100000):
    
    upmomentum = False
    downmomentum = False
    tradeUp = False
    tradeDown = False
    Trade = False
    okUp = 0
    okDown = 0
    minIndex = 0
    maxIndex = 0
    histPrice = 0
    indexDiff = 0
    decision = 0
    stdDev = 0
    dontTrade = 0
    keyLag = 0
    slopeRange = []
    if totalUpCounts > 250 or totalDownCounts > 250:
        totalUpCounts = 0
        totalDownCounts = 0
        
  
    newPattern = []
    avgLine = ((ask+ask)/2)
    avgLine = avgLine[:toWhat]
    #avgLine = ((ask2+ask2)/2)
    myData = ((ask+ask)/2)
    counter = patternSize+PredictionLag+5
    #avgLine = avgLine[650000:660000]
    start = startingPoint +x-counter
    end = startingPoint +x-PredictionLag-5
    start2 = startingPoint +x-PredictionLag-5
    end2 = startingPoint+x
    newPattern = fullData[start:end]
    futureOutcome = [fullData[start2:end2]]
    #addPattern(newPattern, futureOutcome)
    print("size of norm is", len(allData))
   
    print('complete')
    input_Data = myData[counter4:toWhat]
    counter4 = counter4+1
    
    #med = np.median(norm)
    filtered_sine = butter_highpass_filter(input_Data,20,fps)
    min_sine = np.amin(filtered_sine)
    max_sine = np.amax(filtered_sine)
    print('max',max_sine)
    print('min',min_sine)

    filtered_sine = [(((i-min_sine)/(max_sine-min_sine))*2)-1 for i in filtered_sine]
    filtered_sine = fir_highpass(filtered_sine,20,fps)
    for i in range(len(filtered_sine)):
        filtered_sine[i] = 0.5 * np.log((1+filtered_sine[i])/(1-filtered_sine[i]))
           
    filtered_sine = np.nan_to_num(filtered_sine)
   # filtered_sine[filtered_sine == -inf] = 0
    #filtered_sine[filtered_sine == inf] = 0
   
 
##    plt.figure(figsize=(20,10))
##    plt.subplot(211)
##    plt.plot(range(len( input_Data)), input_Data)
##    plt.title('generated signal')
##    plt.subplot(212)
##   # norm_data = np.histogram(filtered_sine, bins=10, density=True)
##    plt.hist(filtered_sine,10)
##    plt.title('filtered signal')
##    plt.show()
    
    mean_value = np.mean(filtered_sine)
    max_value = np.amax(filtered_sine)*0.5
    min_value = np.amin(filtered_sine)*0.5
    down_values = filtered_sine[np.where(filtered_sine < min_value)]
    up_values = filtered_sine[np.where(filtered_sine > max_value)]
    if len(up_values) != 0 and len(down_values) != 0:      
        print('length of up is', len(up_values))
        print('length of down is', len(down_values))
        len_down = len(down_values)
        len_up = len(up_values)
        len_diff = abs(len(down_values)-len(up_values))
        up_rangemax = np.amax(up_values)
        up_rangemin = np.amin(up_values)
        down_rangemax = np.amax(down_values)
        down_rangemin = np.amin(down_values)
        up_range = up_rangemax-up_rangemin
        down_range = down_rangemax-down_rangemin
        print('up range is', up_range)
        print('down range', down_range)
        range_diff = len_up-len_down
        range_diffr = up_range - down_range
    
##    if  abs(range_diff) > 25 and  abs(range_diff) <= 100 and trend_found == 0:
##        rangesum.append(range_diff)
##        meanrange = np.mean(rangesum)
##        countRange = countRange+1
##        
##    if  abs(range_diff) >= 10 and trend_found == 0 and countRange == 1:
##        trend_found = 1
##        time.sleep(5)
##        if np.sign(meanrange) >= 1:
##            plsUp = 1
##            plsDown = 0
##        elif np.sign(meanrange) >= -1:
##            plsDown = 1
##            plsUp = 0
    counterela = counterela + 1
    if  abs(max_range) <= abs(filtered_sine[-1]):
        max_range = filtered_sine[-1]

    if counterela >= 100 and abs(max_range) >= 1:
        trend_found = 1
        if np.sign(max_range) >= 1:
            plsDown = 1
            plsUp = 0
        elif np.sign(max_range) >= -1:
            plsUp = 1
            plsDown = 0

    if trend_found == 1:
        counterula = counterula + 1
           
        print('range difference is', range_diff)

    min_fish = 1
    max_fish = 5
    fish_value = 100*(abs(filtered_sine[-1]))/(max_fish-min_fish)
    min_trend = 0
    max_trend = 0.7
    trend_value = 100*((abs(range_diffr)-min_trend)/(max_trend-min_trend))
    print('fisher value is', fish_value)
    print('trend value is', trend_value)
    print('max range is' , max_range)
##    boilinger = bb(np.array(myData[(toWhat-patternSize):toWhat]), 7)
##   # print('fucking length ' , myData[(startingPoint-patternSize):startingPoint])
##    
## 
##    momentum = roc(np.array(myData[(toWhat-patternSize):toWhat]))
##
##    bandwidth =  [ x[-3] for x in boilinger]
##    percentBand = boilinger[-1, 5]
##    averageB = np.average(bandwidth)
##    #print(percentBand)
##    #print(averageB)
##    #print(boilinger[-1][-3])
##   # bandwidthList.append(bandwidth)
##    baby = boilinger[-3][-1]
##  
##    averageBandwidth = np.average(bandwidthList)
##    averageMomentum = abs(np.average(momentum))
##
##    currentMom = momentum[-9:-1]
##    currentMomo = momentum[-1]
##    currentMomentum = np.amax(currentMom)
##    currentMomentumDown = np.amin(currentMom)
##    currentMomentumAverage = np.average(currentMom)
##    signUp = np.sign(currentMomentum)
##    signDown = np.sign(currentMomentumDown)
##    if (signUp == 1 and currentMomentum >= 0.01) :
##        upmomentum = True
##        print("momentum is saying it will go up")
##        
##    if (signDown == -1 and abs(currentMomentumDown) >= 0.01)  :
##        downmomentum = True
##        print("momentum is saying it will go down")
##        
##    if averageMomentum < currentMomentum:
##        PredictionLag = 1
##    else:
##        PredictionLag = 1

 
  


##    if np.sign(currentMomentum) == np.sign(currentMomentumDown):
##        if np.sign(currentMomentum) == 1 and currentMomentum > 0.04 and abs(currentMomentumDown) > 0.04 and currentMomentumAverage > 0  :
##            tradeUp = True
##            tradeDown = False
##     
##        elif np.sign(currentMomentum) == -1 and abs(currentMomentum) > 0.04 and abs(currentMomentumDown) > 0.04 and currentMomentumAverage < 0 :
##            tradeDown = True
##            tradeUp = False
##    
##    else:
##        if currentMomentum > abs(currentMomentumDown):
##            goUp = True
##            goDown = False
##        else:
##            goDown = True
##            goUp = False


        
    

    #avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
    
    
  
    #patForRec = currentPattern()
    #patternRecognition()
    #print('pattern actual price')time.sleep(5);
    #print(patternPrice)
    #print('runnign to what')x
    #print(toWhat)
    #lastPrice = ask[toWhat]
    #futurePrice = ask[toWhat+30]
    #prcChange = percentChange(lastPrice,futurePrice)
    print('current price' , filtered_sine[-1])
   

    #outcomeRange = ask[toWhat-5:toWhat-1]
    #try:
     #   avgOutcome = np.average(outcomeRange)
    #except Exception, e:
     #   print str(e)
      #  avgOutcome = 0
        #outcomeRange = avgLineFull[y+20:y+30]
        #currentPoint = avgLineFull[y]
    #currentOutcome = percentChange(ask[toWhat], avgOutcome)
    stdDev = np.std(input_Data[-1000:-1])
    histPrice = input_Data[-60:-2]
    histPrice2 = input_Data[-100:-2]
    minIndex = np.argmin(histPrice)
    maxIndex = np.argmax(histPrice)
    priceDiffMin = ((input_Data[-1]-histPrice[minIndex])/histPrice[minIndex])*100
    priceDiffMax = ((input_Data[-1]-histPrice[maxIndex])/histPrice[maxIndex])*100
    priceAverage = np.mean(histPrice)
    priceDiffbetween = ((input_Data[-1]-priceAverage)/priceAverage)*100
    signDiff = np.sign(priceDiffbetween)   
    print('average price' , priceAverage)
    print('price now is' , input_Data[-1])
    print('standard deviation is' , stdDev)
    if stdDev >= 0.001:
        if  priceDiffMax < 0 and priceDiffMax > -0.9  and previousDiffMax < -0.1 and signDiff == -1:       
            counterUpTrend = counterUpTrend+1
            totalUpCounts  =  totalUpCounts  +1
            previousDiffMax = priceDiffMax
            upTrendAv.append(previousDiffMax)
        
        else:
            counterUpTrend = 0
            previousDiffMax = 0
            upTrendAv = []
        if priceDiffMin > 0 and priceDiffMin < 0.9 and previousDiffMin > 0.1 and signDiff == 1:
            counterDownTrend = counterDownTrend+1
            totalDownCounts  =  totalDownCounts  +1
            previousDiffMin = priceDiffMin
            downTrendAv.append(previousDiffMin)
        else:
            counterDownTrend = 0
            previousDiffMin = 0
            downTrendAv = []
    #priceDiffbetween = ((input_Data[-1]-input_Data[-60])/input_Data[-60])*100
    previousDiffMax =  priceDiffMax
    previousDiffMin =  priceDiffMin
    if filtered_sine[-1] <= -1 and priceDiffMax < -0.2 and signDiff == -1 :
        
       
       indexDiff = 59 - maxIndex
       
       print('index diff  is', indexDiff)
       print("price diff is", priceDiffMax)
       okUp = 1
       if indexDiff < 5:
           PredictionLag = 5
       elif indexDiff > 1 and indexDiff <= 5:
           PredictionLag = 10
       elif indexDiff >5 and indexDiff <= 10:
           PredictionLag = 15
       elif indexDiff >10 and indexDiff <= 15:
           PredictionLag = 30
       elif indexDiff >15 and indexDiff <= 30:
           PredictionLag = 30
       elif indexDiff >30 and indexDiff <= 60:
           PredictionLag = 30
       else:
           PredictionLag = 60
       
      

    if filtered_sine[-1] >= 1 and priceDiffMin > 0.2 and signDiff == 1 :
     
       indexDiff = 59 - minIndex
       
       print("price diff is", priceDiffMin)
       print('min price is',histPrice[maxIndex])
       okDown = 1
       if indexDiff < 5:
           PredictionLag = keyLag
       elif indexDiff > 1 and indexDiff <= 5:
           PredictionLag = 5
       elif indexDiff >5 and indexDiff <= 10:
           PredictionLag = 10
       elif indexDiff >10 and indexDiff <= 15:
           PredictionLag = 15
       elif indexDiff >15 and indexDiff <= 30:
           PredictionLag = 30
       elif indexDiff >30 and indexDiff <= 60:
           PredictionLag = 30
       else:
           PredictionLag = 60
    
    #if abs(filtered_sine[-1]) <= 0.7:
    PredictionLag = 60

    print("down trend  count is", counterUpTrend)
    print("up trend  count is", counterDownTrend)
    print("total down trend  count is",  totalUpCounts)
    print("total up trend  count is",  totalDownCounts)
    if len(upTrendAv) != 0:
        print("down trend  signal is", np.amin(upTrendAv))
    if len(downTrendAv) != 0:
        print("up trend  signal is", np.amax(downTrendAv))
    trendStrength =  totalDownCounts-totalUpCounts
    print("trade counter is", tradeCounter)
    if abs(trendStrength) >= 99:
        doTrade = 1

    regr = linear_model.LinearRegression()
    slope_data = input_Data[-lastMinute:toWhat]
    time_interval = list(range(0,lastMinute))
    x1 = np.array(time_interval).reshape(len(time_interval), 1)
    y1 = np.array(slope_data).reshape(len(slope_data), 1)
    #print(len(buy_data), len(time_interval))
    regr.fit(x1, y1)
    #print('Coefficients: \n', regr.coef_[0][0])
    slope_buy = regr.coef_[0][0]
    for i in range(lastMinute):
        slope_ok = ((input_Data[-1]-input_Data[-lastMinute-i])/lastMinute)
        slopeRange.append(slope_ok)
       # print("slope is",  slope_ok)
    data2 = input_Data
    data2 = np.reshape(data2,(-1,25))
    bandwidth = estimate_bandwidth(data2, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # fit the data
    ms.fit(data2)
    ml_results = []
    for k in range(len(np.unique(ms.labels_))):
        my_members = ms.labels_ == k
        values = data2[my_members, 0]    
        #print values
        
        # find the edges
        ml_results.append(min(values))
        ml_results.append(max(values))



    ml_fuck = np.array(ml_results)
    ml_results =  ml_fuck.flatten()
    maxValue = np.amax(ml_results)
    minValue = np.amin(ml_results)
    currentShit = input_Data[-1]
    lessthanCurrent = []
    morethanCurrent = []
    lessthanCurrent = ml_results[np.where(ml_results <currentShit)]
    morethanCurrent = ml_results[np.where(ml_results >  currentShit)]
    print('Mnimu support is:' ,morethanCurrent)
    if len(morethanCurrent) > 0 and len(lessthanCurrent) > 0:
        minMaxDiff = morethanCurrent[0]-lessthanCurrent[-1]*0.25
    if input_Data[-1] < minValue or input_Data[-1] > maxValue:
        dontTrade = 1 
    elif len(ml_results) > 2 and input_Data[-1]-np.mean(lessthanCurrent) <=  minMaxDiff:
        dontTrade = 1
    #nearestValue = np.find_nearest( ml_results, input_Data[-1] )
    
    #print("interval data")
##    if okUp ==1:
##        plt.figure(figsize=(20,10))
##        plt.subplot(211)
##        plt.plot(range(len( input_Dataneeds	to	keep	)), input_Data)
##        plt.title('generated signal')
##        plt.subplot(212)
##        # norm_data = np.histogram(filtered_sine, bins=10, density=True)
##        plt.hist(filtered_sine,10)
##        plt.title('filtered signal')
##        plt.show()    
##        print('current decision is approved and details are as follows', filtered_sine[-1])
##        decision = input('enter input')
##    elif okDown ==1:
##        plt.figure(figsize=(20,10))
##        plt.subplot(211)
##        plt.plot(range(len( input_Data)), input_Data)
##        plt.title('generated signal')
##        plt.subplot(212)
##        # norm_data = np.histogram(filtered_sine, bins=10, density=True)
##        plt.hist(filtered_sine,10)
##        plt.title('filtered signal')
##        plt.show()
##    plt.figure(figsize=(20,10))
##    plt.subplot(211)
##    plt.plot(range(len( input_Data)), input_Data)
##    plt.title('generated signal')
##    for k in ml_results:
##        plt.axhline(y=k)
##    plt.subplot(212)

##   # norm_data = np.histogram(filtered_sine, bins=10, density=True)
##    plt.hist(filtered_sine,10)
##    plt.title('filtered signal')
##    plt.show()

   # print('current decision is approved and details are as follows', filtered_sine[-1])
   # decision = input('enter input')
   # time.sleep(5);
    #
    #(downmomentum == True and tradeDown == True and goDown == True and bandwidth >= 0.0012) or

   # if  ((filtered_sine[-1] <= -1 and range_diff < 0.1)  or (filtered_sine[-1] >= 1 and up_range>down_range and range_diff > 0.1) ):
      
    #if ((fish_value > trend_value and filtered_sine[-1] < -1) or (trend_value > fish_value and filtered_sine[-1] > 0.8 and up_range>down_range and len_up>len_down and len_diff > 3)) and abs(trend_value-fish_value) > 20 :
   # if  (filtered_sine[-1] > 1 and up_range>down_range and down_range > 0 and range_diff > 0.15  and len_up>len_down and len_diff > 1):
   # if (filtered_sine[-1] <= -1 and okUp == 1 and stdDev < 0.001) or (filtered_sine[-1] <= -0.5 and  totalUpCounts>totalDownCounts and trendStrength > 99):
   # if abs(trendStrength) >= 50 and np.sign(trendStrength) == 1 and doTrade == 1 and filtered_sine[-1] > 0:
   #if (filtered_sine[-1] <= -1 and okUp == 1 and stdDev < 0.001) or (filtered_sine[-1] <= -0.5 and  totalUpCounts>totalDownCounts and trendStrength > 50):
    if filtered_sine[-1] <= -0.5 and  counterUpTrend>counterDownTrend and abs(trendStrength) > 15 and sum(n < 0 for n in slopeRange) == lastMinute and len(ml_results) > 2 and dontTrade == 0:
        tradeCounter = tradeCounter + 1
##        if tradeCounter == 5:
##            doTrade = 0
##            tradeCounter = 0
##            totalUpCounts = 0
##            totalDownCounts = 0
        if  allData[toWhat+PredictionLag] - allData[toWhat] >= 0 :
            win = win + 1
            pastWinStatus = 0
            print('trade won on up !!!, predictionLag was', PredictionLag)
            numTrades = numTrades + 1
            error=abs(allData[toWhat+PredictionLag] - allData[toWhat]-average)
            errorList.append(error)
            plt.figure(figsize=(20,10))
            plt.subplot(211)
            plt.plot(range(len( input_Data)),input_Data)
            plt.title('generated signal')
            for k in ml_results:
                plt.axhline(y=k)
            plt.subplot(212)
           # norm_data = np.histogram(filtered_sine, bins=10, density=True)
            plt.hist(filtered_sine,10)
            plt.title('filtered signal')
            plt.show()


            #Change font size of x-axis    
            time.sleep(5);
            if maxLossCounter>totalLossCounter :
               totalLossCounter=maxLossCounter
               maxLossCounter=0;
            else:
               totalLossCounter=totalLossCounter
               maxLossCounter=0;
            
             
        else:
            loss = loss + 1
            direction = -1
            pastWinStatus = -1
            lossUp = lossUp + 1
            print('trade Lost on up prediction:(, prediction lag ', PredictionLag)
            numTrades = numTrades + 1
            error = abs(allData[toWhat + PredictionLag] - allData[toWhat] - average)
            errorList.append(error)
            maxLossCounter = maxLossCounter+1
            plt.figure(figsize=(20,10))
            plt.subplot(211)
            plt.plot(range(len( input_Data)), input_Data)
            plt.title('generated signal')
            for k in ml_results:
                plt.axhline(y=k)
            plt.subplot(212)
           # norm_data = np.histogram(filtered_sine, bins=10, density=True)
            plt.hist(filtered_sine,10)
            plt.title('filtered signal')
            plt.show()


            #Change font size of x-axis    
            time.sleep(5);
            
   # elif ((filtered_sine[-1] >= 1 and range_diff < 0.1) or ( filtered_sine[-1] <= -1 and down_range>up_range and range_diff > 0.1) ):
    #elif ((fish_value > trend_value and filtered_sine[-1] > 1) or (trend_value > fish_value and filtered_sine[-1] < -0.8 and up_range<down_range and len_down>len_up and len_diff > 3)) and abs(trend_value-fish_value) > 20:
    #elif (filtered_sine[-1] < -1 and up_range<down_range and up_range > 0 and abs(range_diff) > 0.15 and len_down>len_up and len_diff > 1):
    #elif (filtered_sine[-1] >= 1 and okDown == 1 and stdDev < 0.001) or  (filtered_sine[-1] >= 0.5 and  totalDownCounts>totalUpCounts and trendStrength > 50):
    elif  filtered_sine[-1] >= 0.5 and  counterDownTrend>counterUpTrend and abs(trendStrength) > 15 and (sum(n > 0 for n in slopeRange)) == lastMinute and len(ml_results) > 2 and dontTrade == 0:
    #elif abs(trendStrength) >= 50 and np.sign(trendStrength) == -1 and doTrade == 1 and filtered_sine[-1] < 0:
##        tradeCounter = tradeCounter + 1
##        if tradeCounter == 5:
##            doTrade = 0
##            tradeCounter = 0
##            totalUpCounts = 0
##            totalDownCounts = 0
            
        if  allData[toWhat+PredictionLag] - allData[toWhat] <= 0:
            win = win + 1
            pastWinStatus = 0
            print('trade won on down!!! prediction lag was', PredictionLag)
            numTrades = numTrades + 1
            error=abs(allData[toWhat+PredictionLag] - allData[toWhat]-average)
            errorList.append(error)
            plt.figure(figsize=(20,10))
            plt.subplot(211)
            plt.plot(range(len( input_Data)), input_Data)
            plt.title('generated signal')
            for k in ml_results:
                plt.axhline(y=k)
            plt.subplot(212)

           # norm_data = np.histogram(filtered_sine, bins=10, density=True)
            plt.hist(filtered_sine,10)
            plt.title('filtered signal')
            plt.show()


            #Change font size of x-axis    
            time.sleep(5);
            if maxLossCounter>totalLossCounter :
               totalLossCounter=maxLossCounter
               maxLossCounter=0;
            else:
               totalLossCounter=totalLossCounter
               maxLossCounter=0;
        else:
            direction = 1
            lossDown = lossDown + 1
            pastWinStatus = -1
            loss = loss + 1
            print('trade Lost on fdown prediction:(, prediction lag was' , PredictionLag)
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
            maxLossCounter = maxLossCounter+1
            plt.figure(figsize=(20,10))
            plt.subplot(211)
            plt.plot(range(len( input_Data)), input_Data)
            plt.title('generated signal')
            for k in ml_results:
                plt.axhline(y=k)
            plt.subplot(212)
           # norm_data = np.histogram(filtered_sine, bins=10, density=True)
            plt.hist(filtered_sine,10)
            plt.title('filtered signal')
            plt.show()
            time.sleep(5);
            maxLossCounter = maxLossCounter+1

            # get the last price, get the future price. find percentage change and compare to predicted
                                     
    else:
         pastMovement = 0
         pastWinStatus = 0
    print("wins is")
    print(win)
    print((win*0.71*600)-(loss*600))
    print("trades is")
    print(numTrades)
    print("TotalContineusLoss",totalLossCounter)
    totalEnd = time.time()-totalStart
    print 'Entire processing took:',totalEnd,'seconds'
    toWhat += 1
print('number of trades:')
print(numTrades)
print('accuracy is')
print(100*float(win)/float(numTrades))
print('max error for win is')
print(np.amax(errorList))
    
