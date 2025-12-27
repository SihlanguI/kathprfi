#%matplotlib inline

#%matplotlib widget

#from matplotlib import rcParams
import matplotlib.pylab as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 10),
         'axes.labelsize': 16,
         'axes.titlesize':16,
         'xtick.labelsize':16,
         'ytick.labelsize':16,
         'axes.labelweight':'bold',
          'legend.fontsize': 16,
          'font.size':13,
         'figure.max_open_warning': 0}
         
plt.rcParams.update(params)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colorss

import time
from datetime import date
from os import path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import pandas as pd

import dask.array as da
from dask.distributed import Client
import xarray as xr
from dask import compute
from dask import config
from dask import delayed

from tqdm.notebook import tqdm

from scipy.stats import t

# define all the function to be imported in the KATHPRFI analysis script
def plot_RFI_mask(pltobj,axis,alpha=0.1):
    
    '''
    This function will plot a massk around all know RFI soucres,
    '''
    
    if axis == 'v':
        objfunc = pltobj.axvspan
    else:
        objfunc = pltobj.axhspan
    objfunc(470,854, alpha=alpha, color='cyan')#UHF TV  (470 - 854 MHz)
    objfunc(925,960, alpha=alpha, color='yellow')#GSM 900 Mobile
    objfunc(960,1164, alpha=alpha, color='purple')#Aeronautical Nav
    objfunc(1674,1677, alpha=alpha, color='grey')#Meteosat
    objfunc(1667,1667, alpha=alpha, color='grey')#Fengun
    objfunc(1682,1682, alpha=alpha, color='grey')#Meteosat
    objfunc(1685,1687, alpha=alpha, color='grey')#Meteosat
    objfunc(1687,1687, alpha=alpha, color='grey')#Fengun
    objfunc(1690,1690, alpha=alpha, color='grey')#Meteosat
    objfunc(1699,1699, alpha=alpha, color='grey')#Meteosat
    objfunc(1702,1702, alpha=alpha, color='grey')#Fengyun
    objfunc(1705,1706, alpha=alpha, color='grey')#Meteosat
    objfunc(1709,1709, alpha=alpha, color='grey')#Fengun
    objfunc(1501,1570, alpha=alpha, color='blue')#Inmarsat
    objfunc(1496,1585, alpha=alpha, color='blue')#Inmarsat
    objfunc(1574,1576, alpha=alpha, color='blue')#Inmarsat
    objfunc(1509,1572, alpha=alpha, color='blue')#Inmarsat
    objfunc(1574,1575, alpha=alpha, color='blue')#Inmarsat
    objfunc(1512,1570, alpha=alpha, color='yellow')#Thuraya
    objfunc(1450,1498, alpha=alpha, color='red')#Afristar
    objfunc(1652,1694, alpha=alpha, color='red')#Afristar
    objfunc(1542,1543, alpha=alpha, color='cyan')#Express AM1
    objfunc(1554,1554, alpha=alpha, color='cyan')#Express AM 44
    objfunc(1190,1215, alpha=alpha, color='green')#Galileo
    objfunc(1260,1300, alpha=alpha, color='green')#Galileo
    objfunc(1559,1591, alpha=alpha, color='green')#Galileo
    objfunc(1544,1545, alpha=alpha, color='green')#Galileo
    objfunc(1190,1217, alpha=alpha, color='green')#Beidou
    objfunc(1258,1278, alpha=alpha, color='green')#Beidou
    objfunc(1559,1563, alpha=alpha, color='green')#Beidou
    objfunc(1555,1596, alpha=alpha, color='green')#GPS L1  1555 -> 1596
    objfunc(1207,1238, alpha=alpha, color='green')#GPS L2  1207 -> 1248
    objfunc(1378,1384, alpha=alpha, color='green')#GPS L3
    objfunc(1588,1615, alpha=alpha, color='green')#GLONASS  1588 -> 1615 L1
    objfunc(1232,1259, alpha=alpha, color='green')#GLONASS  1232 -> 1259 L2
    objfunc(1616,1630, alpha=alpha, color='grey')#IRIDIUM    
    #objfunc(1722,1739, alpha=alpha, color='grey')#IRIDIUM
    
    
    
def OpenZarr(file_list):
    """
    This function will load each file in file_list and concatenate them into a single xarray Dataset
    """
    datasets = []
    for file in file_list:
        zarrf = xr.open_zarr(file, group='arr', consolidated=False)
        datasets.append(zarrf)
    return datasets

#@delayed
def ProcessZarr(ZarrFile,Dim):
    '''
    This function will use the zarr file to return the probability on selected dimension but will compute on demand
    '''
    
    np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
    
    m = ZarrFile.master.sum(dim=Dim)
    c = ZarrFile.counter.sum(dim=Dim)
    
    p = (m.astype(float)/c.astype(float)).values

    return(p)



#@delayed
def ProcessZarrSelect(ZarrFile,freq_select, Dim):
    '''
    This function will use the zarr file to return the probability on selected dimension but will compute on demand
    '''
    
    np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
    
    m = ZarrFile.master.sel(frequency=freq_select).sum(dim=Dim)
    c = ZarrFile.counte.sel(frequency=freq_select).sum(dim=Dim)
    
    p = (m.astype(float)/c.astype(float)).values
#     print('process done')
    return(p)

def ReturnProb(FileList, Dimension):
    '''
    
        This function will return the probability after calling the Openzarr function and process the zarr file
    
    '''

    data = OpenZarr(FileList)
    data = ProcessZarr(data, Dimension)
        
    return data


def ReturnProbFreSlice(FileList, Dimension):
    '''
    
    This function will return the probability for the frequency selected
    
    '''

    data = OpenZarr(FileList)
  
    data = ProcessZarrSelect(data, Dimension)
    
    return data

        
def FreqCut(FreqLo, FreqHi, FreqData):
    '''
    Input: 
        FreqLo: Lower frequest selection
        FreqHi: Upper frequency selection
        Data: Data set under investigation
    Output:
        Return Indices of selected frequency range
    '''

    ChanIndx = np.where((FreqData >= FreqLo*1.e6) & 
                        (FreqData <= FreqHi*1.e6))[0]

    if len(ChanIndx) == 0:
        print("Selection resulted in empty indices, check your inputs")
    else:
        return ChanIndx
    
def OpenZarrSelectFreq(ZarrList, StartFreq, EndFreq, freq):
    
    '''
    This function is similar to OpenZarrSelect, except that it will produce a probability dataset as function of Time 
    for a frequency slice
    '''
    
    ChanIndx = FreqCut(StartFreq, EndFreq,freq)
    if len(ChanIndx) != 0:
        ProbData = []
#     zarrf = xr.open_zarr(FileList, group = 'arr')

        # ProbData = []
        for i in range(len(ZarrList)):
            try:
                zarrf = xr.open_zarr(ZarrList[i], group = 'arr')
                Dim = ['frequency','baseline','azimuth','elevation']

                with config.set(**{'array.slicing.split_large_chunks': True}):
                    tm = zarrf.master[:,ChanIndx,:].sum(dim=Dim)
                    tc = zarrf.counter[:,ChanIndx,:].sum(dim=Dim)
                    np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
                    p = compute(tm.astype(float)/tc.astype(float))

                    ProbData.append(p)
                    print('Added file {} : {}'.format(i,ZarrList[i][32:]))
            except Exception as e:
                print(e)
                continue
    else:
        print("Frequency selection had some issues")
        
    return ProbData

def OpenZarrSelectFreqTest(ZarrList, StartFreq, EndFreq, freq, Dim_to_Sum):
    
    '''
    The function takes in four arguments: ZarrList, which is a list of the Zarr files to be opened, StartFreq and EndFreq, which are the start and end frequencies of the frequency slice to be selected, freq, which is the frequency array, and Dim_to_Sum, which is the dimension to sum over.
    The FreqCut function is called to extract the index of the channels that fall within the selected frequency slice.
    If channels are found within the frequency slice, the function creates an empty list called ProbData.
    For each file in ZarrList, the function tries to open the Zarr file and extract the necessary data. If it encounters an error, it prints the error message and continues to the next file.
    The function sums the data along the dimension specified by Dim_to_Sum for the selected frequency channels.
    The function calculates the probability data by dividing the summed data from step 5 by the corresponding counts and passes it to the compute function.
    The resulting probability data and the filename are appended to the ProbData list.
    If no channels are found within the selected frequency slice, the function prints a message to that effect.
    Finally, the function returns the ProbData list.
    '''
    
    ChanIndx = FreqCut(StartFreq, EndFreq,freq)
    if len(ChanIndx) != 0:
        ProbData = []
#     zarrf = xr.open_zarr(FileList, group = 'arr')

        # ProbData = []
        for i in range(len(ZarrList)):
            try:
                zarrf = xr.open_zarr(ZarrList[i], group = 'arr')
                # Dim = [Dim_to_Sum,'baseline','azimuth','elevation']

                with config.set(**{'array.slicing.split_large_chunks': True}):
                    tm = zarrf.master[:,ChanIndx,:].sum(dim=Dim_to_Sum)
                    tc = zarrf.counter[:,ChanIndx,:].sum(dim=Dim_to_Sum)
                    np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
                    p = compute(tm.astype(float)/tc.astype(float))

                    ProbData.append((ZarrList[i], p))
                    print('Added file {} : {}'.format(i,ZarrList[i][32:]))
            except Exception as e:
                print(e)
                continue
    else:
        print("Frequency selection had some issues")
        
    return ProbData

def OpenZarrSelectFreqTime(ZarrList, StartFreq, EndFreq, freq, Dim_to_Sum):
    
    '''
    This function is similar to OpenZarrSelect, except that it will produce a probability dataset as function of Time 
    for a frequency slice
    '''
    
    ChanIndx = FreqCut(StartFreq, EndFreq,freq)
    if len(ChanIndx) != 0:
        ProbData = []
#     zarrf = xr.open_zarr(FileList, group = 'arr')

        # ProbData = []
        for i in range(len(ZarrList)):
            try:
                zarrf = xr.open_zarr(ZarrList[i], group = 'arr')
                Dim = [Dim_to_Sum,'baseline','azimuth','elevation']

                with config.set(**{'array.slicing.split_large_chunks': True}):
                    tm = zarrf.master[:,ChanIndx,:].sum(dim=Dim)
                    tc = zarrf.counter[:,ChanIndx,:].sum(dim=Dim)
                    np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
                    p = compute(tm.astype(float)/tc.astype(float))

                    ProbData.append(p)
                    print('Added file {} : {}'.format(i,ZarrList[i][32:]))
            except Exception as e:
                print(e)
                continue
    else:
        print("Frequency selection had some issues")
        
    return ProbData

def OpenZarrSelectTime(ZarrList, Time_Sel):
    ProbData = []
    for i in range(len(ZarrList)):
        try:
            zarrf = xr.open_zarr(ZarrList[i], group = 'arr')
            Dim = ['time','baseline','azimuth','elevation']

            with config.set(**{'array.slicing.split_large_chunks': True}):
                tm = zarrf.master.sel(time = Time_Sel).sum(dim=Dim)
                tc = zarrf.counter.sel(time = Time_Sel).sum(dim=Dim)
                np.seterr(divide='ignore', invalid='ignore') # to avoid warning on divide
                p = compute(tm.astype(float)/tc.astype(float))

                ProbData.append(p)
                print('Added file {} : {}'.format(i,ZarrList[i][32:]))
        except Exception as e:
            print(e)
            continue
            
    return(ProbData)
    

def MultiProb(FileList, DimenSions, MyDim):
    '''
    Takes the master, counter and dimension name one is
    interested in.
    Input:
        FileList - list holding names of zarr files
        DimenSions - list holding dims from zarr
        MyDim - dimension to work on
    Returns : Probability array for the chosen dimension.
    
    
    '''
    
    print(f'Going to return {MyDim} probability')
    print(f"Length MyDim {len(MyDim)},length {len(DimenSions)}")
    
    if len(MyDim) == 1:
        DimenSions.remove(MyDim[0])
    else:
        for indx in range(len(MyDim)):
            DimenSions.remove(MyDim[indx])
    
    print(DimenSions)
    
    ave_mon = []
    
    for i in tqdm(range(len(FileList))):

        try:
            p = ReturnProb(FileList[i],DimenSions) #.compute()
            ave_mon.append(p)
        except Exception as e:
            print(e)
            continue
            
    return(ave_mon)

def PlotWithStat(StartFreq,EndFreq,ProbArray,legend, freq, fig=None):
    '''
    This function will produce plot for Prob as function of time of the day 
    We can do frequemcy selction using StartFreq,EndFreq
    StartFreq,EndFreq - start and End frequency in GHz - obsolete can remove
    ProbArray - 2-D probability array 
    freq - is the full Frequency array 
    name - is the string to put in the title (disabled for now)
    '''

    if fig is None :
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time [UTC]')
        ax.set_ylabel('Probability')
        ax.set_xticks(np.arange(0,24,2))

    plt.step(np.arange(ProbArray.shape[0]),ProbArray,linewidth=5,label=legend)
#     plt.show()
    return(fig)

def plot_RFI_mask(pltobj,axis,alpha=0.1):
    
    '''
    This function will plot a massk around all know RFI soucres,
    '''
    
    if axis == 'v':
        objfunc = pltobj.axvspan
    else:
        objfunc = pltobj.axhspan
    objfunc(470,854, alpha=alpha, color='cyan')#UHF TV  (470 - 854 MHz)
    objfunc(925,960, alpha=alpha, color='yellow')#GSM 900 Mobile
    objfunc(960,1164, alpha=alpha, color='purple')#Aeronautical Nav
    objfunc(1674,1677, alpha=alpha, color='grey')#Meteosat
    objfunc(1667,1667, alpha=alpha, color='grey')#Fengun
    objfunc(1682,1682, alpha=alpha, color='grey')#Meteosat
    objfunc(1685,1687, alpha=alpha, color='grey')#Meteosat
    objfunc(1687,1687, alpha=alpha, color='grey')#Fengun
    objfunc(1690,1690, alpha=alpha, color='grey')#Meteosat
    objfunc(1699,1699, alpha=alpha, color='grey')#Meteosat
    objfunc(1702,1702, alpha=alpha, color='grey')#Fengyun
    objfunc(1705,1706, alpha=alpha, color='grey')#Meteosat
    objfunc(1709,1709, alpha=alpha, color='grey')#Fengun
    objfunc(1501,1570, alpha=alpha, color='blue')#Inmarsat
    objfunc(1496,1585, alpha=alpha, color='blue')#Inmarsat
    objfunc(1574,1576, alpha=alpha, color='blue')#Inmarsat
    objfunc(1509,1572, alpha=alpha, color='blue')#Inmarsat
    objfunc(1574,1575, alpha=alpha, color='blue')#Inmarsat
    objfunc(1512,1570, alpha=alpha, color='yellow')#Thuraya
    objfunc(1450,1498, alpha=alpha, color='red')#Afristar
    objfunc(1652,1694, alpha=alpha, color='red')#Afristar
    objfunc(1542,1543, alpha=alpha, color='cyan')#Express AM1
    objfunc(1554,1554, alpha=alpha, color='cyan')#Express AM 44
    objfunc(1190,1215, alpha=alpha, color='green')#Galileo
    objfunc(1260,1300, alpha=alpha, color='green')#Galileo
    objfunc(1559,1591, alpha=alpha, color='green')#Galileo
    objfunc(1544,1545, alpha=alpha, color='green')#Galileo
    objfunc(1190,1217, alpha=alpha, color='green')#Beidou
    objfunc(1258,1278, alpha=alpha, color='green')#Beidou
    objfunc(1559,1563, alpha=alpha, color='green')#Beidou
    objfunc(1555,1596, alpha=alpha, color='green')#GPS L1  1555 -> 1596
    objfunc(1207,1238, alpha=alpha, color='green')#GPS L2  1207 -> 1248
    objfunc(1378,1384, alpha=alpha, color='green')#GPS L3
    objfunc(1588,1615, alpha=alpha, color='green')#GLONASS  1588 -> 1615 L1
    objfunc(1232,1259, alpha=alpha, color='green')#GLONASS  1232 -> 1259 L2
    objfunc(1616,1630, alpha=alpha, color='grey')#IRIDIUM    
    #objfunc(1722,1739, alpha=alpha, color='grey')#IRIDIUM


def PlotWithStat(StartFreq,EndFreq,ProbArray,legend, freq, fig=None):
    '''
    This function will produce plot for Prob as function of time of the day 
    We can do frequemcy selction using StartFreq,EndFreq
    StartFreq,EndFreq - start and End frequency in GHz - obsolete can remove
    ProbArray - 2-D probability array 
    freq - is the full Frequency array 
    name - is the string to put in the title (disabled for now)
    '''

    if fig is None :
        fig = plt.figure(figsize=(12,6))
    
        ax = fig.add_subplot(111)

        #plt.fill_between(np.arange(p.shape[0]), er68[0], er68[1],color='g',alpha=0.3,label=str(con68)+'% Confidence interval')

        ax.set_xlabel('Time [UTC]')
        ax.set_ylabel('Probability')
        ax.set_xticks(np.arange(0,24,2))

    plt.step(np.arange(ProbArray.shape[0]),ProbArray,linewidth=5,label=legend)
    return(fig)