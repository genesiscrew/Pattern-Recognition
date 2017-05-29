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



totalStart = time.time()
sys.setrecursionlimit(30000000)

ask = np.loadtxt('option_data135.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )

ask2 = np.loadtxt('5Years.txt', unpack=True,
                              delimiter=',', usecols = (0) 
                             )





global patFound
average = 0
patternPrice = 0
PredictionLag = 15
patternSize = 120
rsiStack = 0
rsiUp = 0
rsiDown = 0



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
    threshold2 = 0.0025
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
        
        y+=20

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
        plt.scatter(35, average,c=pcolor,alpha=.4)
        #realMovement = float(allData[toWhat+PredictionLag])-allData[toWhat]
        realOutcome = percentChange(allData[toWhat],realAvgOutcome)
        #rMove = realMovement
        #print('the real movement')
        #print(realMovement)
        plt.scatter(40, realOutcome, c='#54fff7',s=25)
        #plt.plot(xp, patForRec, '#54fff7', linewidth = 3)
        plt.grid(True)
        
        plt.title('Pattern Recognition.\nCyan line is the current pattern. Other lines are similar patterns from the past.\nPredicted outcomes are color-coded to reflect a positive or negative prediction.\nThe Cyan dot marks where the pattern went.\nOnly data in the past is used to generate patterns and predictions.')
        plt.show()

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
    period = PredictionLag
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


def roc(prices, period=21):
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

            
            

dataLength = int(ask.shape[0])
print 'data length is', dataLength

allData = ((ask+ask)/2)
allData = allData[240:-5]
avgLineFull = ((ask2+ask2)/2)
avgLineFull = avgLineFull[:200000]

#toWhat = 53500
toWhat = 300
threshold = 0.02
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
bandwidthList = []
for x in xrange(1500):
    avgLine = ((ask+ask)/2)
    avgLine = avgLine[:toWhat]
    #avgLine = ((ask2+ask2)/2)
    #avgLine = avgLine[650000:660000]
    movingAverage = sma(np.array(avgLine[-patternSize:]), PredictionLag)
    
    boilinger = bb(np.array(avgLine[-patternSize:]), patternSize)
    #print(boilinger)

    momentum = roc(np.array(avgLine[-patternSize:]))

    bandwidth =  boilinger[-1][-3]

    bandwidthList.append(bandwidth)
  
    averageBandwidth = np.average(bandwidthList)
    averageMomentum =  abs(np.average(momentum))

    currentMom = momentum[-5:-1]
    currentMomentum = np.amax(currentMom)
    if ((currentMomentum >= 0.02 and currentMomentum < 0.03) or currentMomentum <-0.03) :
        upmomentum = True
        
    if ((currentMomentum <= -0.02 and currentMomentum > -0.03) or currentMomentum >0.03) :
        downmomentum = True
        
    if averageMomentum < currentMomentum:
        PredictionLag = 15
    else:
        PredictionLag = 30

  
    rsiup = rsi(np.array(avgLine[-patternSize:]))
    rsidown = rsi(np.array(avgLine[-patternSize:]))

    
    

    

    #avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
    
    
  
    patForRec = currentPattern()
    patternRecognition()
    #print('pattern actual price')
    #print(patternPrice)
    #print('runnign to what')x
    #print(toWhat)
    #lastPrice = ask[toWhat]
    #futurePrice = ask[toWhat+30]
    #prcChange = percentChange(lastPrice,futurePrice)

    #print('our movement is')
    #print(prcChange)

    outcomeRange = ask[toWhat-5:toWhat-1]
    try:
        avgOutcome = np.average(outcomeRange)
    except Exception, e:
        print str(e)
        avgOutcome = 0
        #outcomeRange = avgLineFull[y+20:y+30]
        #currentPoint = avgLineFull[y]
    currentOutcome = percentChange(ask[toWhat], avgOutcome)
   # print(currentOutcome)
    print(currentMomentum)
    
    if   (average > currentOutcome and average != 0 and abs(average) > threshold and upmomentum == True and patFound == 1):
        average = 0
        patFound = 0
        upmomentum = False
        if  ask[toWhat+PredictionLag] - ask[toWhat] > 0:
            win = win + 1
            print('trade won !!!')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
            time.sleep(5);
        else:
            loss = loss + 1
            print(rsidown[-1])
            print('trade Lost on up prediction:(')
            numTrades = numTrades + 1
            error = abs(ask[toWhat + PredictionLag] - ask[toWhat] - average)
            errorList.append(error)
            time.sleep(5);
    elif (average < currentOutcome and average != 0 and abs(average) > threshold and downmomentum == True and patFound == 1):
        patFound = 0
        average = 0
        downmomentum = False
        if  ask[toWhat+PredictionLag] - ask[toWhat] < 0:
            win = win + 1
            print('trade won !!!')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
            time.sleep(5);
        else:
            loss = loss + 1
            print(rsiup[-1])
            print('trade Lost on fdown prediction:(')
            numTrades = numTrades + 1
            error=abs(ask[toWhat+PredictionLag] - ask[toWhat]-average)
            errorList.append(error)
            time.sleep(5);

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
    
