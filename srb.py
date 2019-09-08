#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
###############################################################################
#						Title 	:	Autonomous Detection of Solar Radio Bursts
#                       Author	:	Ian Finnegan
#						Year	:	2017-2018
###############################################################################
###############################################################################

from skimage.filters import (threshold_local, threshold_adaptive, threshold_otsu,
                             threshold_yen, threshold_isodata, threshold_li, 
                             threshold_minimum, threshold_mean,threshold_triangle, 
                             threshold_niblack, threshold_sauvola)
from skimage.transform import (hough_line, hough_line_peaks, 
                               probabilistic_hough_line, radon)
import matplotlib.pyplot as plt
from matplotlib import dates
from PIL import Image
import numpy as np
import datetime
import struct
import os

def xrange(x):
	return iter(range(x))

#Set path to file 
path = '/Users/Ianfinnegan/Downloads/'
filename = '20170907_104714_bst_00X.dat'

# observation frequencies - mode 3,5,7 -array freq axis
freq1 = np.linspace( 10, 90,200)  #   10-90 MHz
freq2 = np.linspace(110,190,200)  # 110-190 MHz
freq3 = np.linspace(210,240, 88)  # 210-240 MHz

freq = np.concatenate((freq1, freq2, freq3)) 

# Get the size of the file
filelen = os.path.getsize(path+filename)

# Get the number of values (each value is an 8-byte double)
rawdatalen = filelen/8

# Get the bit-mode -- mode 357 and full band observations are always 8 bit modes
bit_mode = 8
#print ("bit_mode =",bit_mode)

# Calculate intermediate variables
num_beamlets = int(244*16/bit_mode)        # Determine the numger of beamlets
num_records = int(rawdatalen/num_beamlets) # Determine the number of time samples we have
datalen = int((rawdatalen/num_beamlets)*num_beamlets)  # Use int division to clip off surplus
datastruct = str(datalen)+'d'         # Used by struct.unpack to work out how to load


# Open the data file
fp = open(path+filename, 'rb')
bin_data = fp.read() 

# Read the raw data from the file into a buffer
dataraw = struct.unpack(datastruct, bin_data)

# re-organise the raw data into an array that is easy to plot
data_all = np.reshape(dataraw[:datalen],(datalen//num_beamlets,num_beamlets))


#print (data_all.shape)

time_len  =data_all.shape[0]
data = data_all[:, :]

#print (data.shape)
#print (data)

#normalizing frequency channel responses using median of values
for sb in xrange(data.shape[1]):
    data[:,sb] = data[:,sb]/np.median(data[:,sb])
#transposing the data for frequency vs. time format
data = np.transpose(data)
#print(data)

# start and end time
start_time = datetime.datetime(int(filename[0:4]), int(filename[4:6]), 
	int(filename[6:8]), int(filename[9:11]), int(filename[11:13]), int(filename[13:15]))
end_second = data.shape[1] # the last second is the same as the length of the time dimension
end_time = start_time + datetime.timedelta( seconds = end_second ) # add end_second to start_time

#makes time array if you want to plot slices of freq and time
timey = []
for i in range(end_second):
	timey.append(start_time+datetime.timedelta(seconds = i))

timee = datetime.datetime.strptime('2017-09-07 18:40', '%Y-%m-%d %H:%M') #restricted start time
timee1 = datetime.datetime.strptime('2017-09-07 18:43', '%Y-%m-%d %H:%M') #restricted end time

ind1 = int((timee - start_time).total_seconds())
ind2 = int((timee1 - start_time).total_seconds())

data_short = data[:, ind1: ind2]


def plot_all():

###############################################################################
#                        Plot Full Spectrum from File
###############################################################################

	fig, ax = plt.subplots(figsize=(14,9))
	# vmin, vmax can be changes
	# frequency limits can be changed with more accurate limits using the sub-band to frequency calculator
	ax.imshow(np.log10(data), vmin = 1, vmax= 2.5,aspect='auto',extent=(dates.date2num(start_time), 
		dates.date2num(end_time), freq[-1], freq[0]))
	ax.xaxis_date()
	#ax.xaxis.set_major_locator(dates.HourLocator())
	# specify the time format for x-axis
	ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
	# specify number of axis labels
	fig.autofmt_xdate()

	# save the plot as png
	plt.xlabel('Start Time: '  + str(start_time) + ' (UT)')
	plt.ylabel('Frequency (MHz)')
	plt.title(str(start_time)[0:10])
	plt.savefig(str(start_time)[0:10]+'_dynamic_spectrum.png')
	#show the plot
	plt.show()


def threshcomp(Data):
    
###############################################################################
#                        Determining Binary Values
###############################################################################    
    
    binary_local = Data > threshold_local(Data, 3)
    binary_adaptive = Data > threshold_adaptive(Data, 3)
    binary_otsu = Data > threshold_otsu(Data)
    binary_yen = Data > threshold_yen(Data)
    binary_isodata = Data > threshold_isodata(Data)
    binary_li = Data > threshold_li(Data)
    binary_min = Data > threshold_minimum(Data)
    binary_mean = Data > threshold_mean(Data)
    binary_triangle = Data > threshold_triangle(Data)
    binary_niblack = Data > threshold_niblack(Data)
    binary_sauvola = Data > threshold_sauvola(Data)
    
###############################################################################
#                                 Plots
###############################################################################

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 8))
    ax = axes.ravel()
    
    ax[0] = plt.subplot(6, 2, 1)
    ax[1] = plt.subplot(6, 2, 2)
    ax[2] = plt.subplot(6, 2, 3)
    ax[3] = plt.subplot(6, 2, 4)
    ax[4] = plt.subplot(6, 2, 5)
    ax[5] = plt.subplot(6, 2, 6)
    ax[6] = plt.subplot(6, 2, 7)
    ax[7] = plt.subplot(6, 2, 8)
    ax[8] = plt.subplot(6, 2, 9)
    ax[9] = plt.subplot(6, 2, 10)

    ax[0].imshow(Data, cmap=plt.cm.gray, aspect='auto')
    ax[0].set_title("Original")
    ax[0].set_ylim([200,0])
    ax[0].set_xlim([100,400])
    ax[0].axis('off')
    
    ax[1].imshow(binary_local, cmap=plt.cm.gray, aspect='auto')
    ax[1].set_title("Local")
    ax[1].set_ylim([200,0])
    ax[1].set_xlim([100,400])
    ax[1].axis('off')
   
    ax[2].imshow(binary_otsu, cmap=plt.cm.gray, aspect='auto')
    ax[2].set_title("Otsu")
    ax[2].set_ylim([200,0])
    ax[2].set_xlim([100,400])
    ax[2].axis('off')
    
    ax[3].imshow(binary_yen, cmap=plt.cm.gray, aspect='auto')
    ax[3].set_title("Yen")
    ax[3].set_ylim([200,0])
    ax[3].set_xlim([100,400])
    ax[3].axis('off')

    ax[4].imshow(binary_isodata, cmap=plt.cm.gray, aspect='auto')
    ax[4].set_title("Isodata")
    ax[4].set_ylim([200,0])
    ax[4].set_xlim([100,400])
    ax[4].axis('off')

    ax[5].imshow(binary_li, cmap=plt.cm.gray, aspect='auto')
    ax[5].set_title("Li")
    ax[5].set_ylim([200,0])
    ax[5].set_xlim([100,400])
    ax[5].axis('off')

    ax[6].imshow(binary_min, cmap=plt.cm.gray, aspect='auto')
    ax[6].set_title("Minimum")
    ax[6].set_ylim([200,0])
    ax[6].set_xlim([100,400])
    ax[6].axis('off')

    ax[7].imshow(binary_mean, cmap=plt.cm.gray, aspect='auto')
    ax[7].set_title("Mean")
    ax[7].set_ylim([200,0])
    ax[7].set_xlim([100,400])
    ax[7].axis('off')

    ax[8].imshow(binary_triangle, cmap=plt.cm.gray, aspect='auto')
    ax[8].set_title("Triangle")
    ax[8].set_ylim([200,0])
    ax[8].set_xlim([100,400])
    ax[8].axis('off')

    ax[9].imshow(binary_niblack, cmap=plt.cm.gray, aspect='auto')
    ax[9].set_title("Niblack")
    ax[9].set_ylim([200,0])
    ax[9].set_xlim([100,400])
    ax[9].axis('off')

    plt.savefig('thresh_comp.png')
    plt.show()


def pixel_hist(Data):

    fig, axes4 = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    ax4 = axes4.ravel()
    
    ax4[0] = plt.subplot(1, 2, 1)
    ax4[1] = plt.subplot(1, 2, 2)

    ax4[0].imshow(Data, cmap=plt.cm.gray, aspect='auto')
    ax4[0].set_ylim([200,0])
    #ax4[0].axis('off')

    ax4[1].imshow((plt.hist(Data.ravel(), bins=256, range=(0, 2.0), fc='k')), aspect='auto')

    plt.savefig('pixelhist.png')
    plt.show


def burst_intensity(Data, n):
    
    intensity_matrix = Data[n]
    intensity_list = intensity_matrix.tolist()
    time = list(xrange(180))
    
    plt.plot(time, intensity_list, 'k')
    #plt.xlim([200,300])
    plt.xlim([50,150])
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Intensity')
    plt.title('Burst Intensity at 55 Mhz')
       
    #plt.hlines(threshold_otsu(Data), 100, 400, 'r', linestyles='dashed', label='Otsu')
    #plt.hlines(threshold_yen(Data), 100, 400, 'b', linestyles='dashed', label='Yen')
    #plt.hlines(threshold_isodata(Data), 100, 400, 'r', linestyles='dashed', label='Isodata')
    #plt.hlines(threshold_li(Data), 100, 400, 'g', linestyles='dashed', label='Li')
    #plt.hlines(threshold_minimum(Data), 100, 400, 'c', linestyles='dashed', label='Minimum')
    plt.hlines(threshold_mean(Data), 0, 400, 'y', linestyles='dashed', label='Mean')
    #plt.hlines(threshold_triangle(Data), 100, 400, 'k', linestyles='dashed', label='Triangle')
    
    
    mean_1std = np.std(Data) + threshold_mean(Data)
    mean_2std = np.std(Data) + np.std(Data) + threshold_mean(Data)
    
    plt.hlines(mean_1std, 0, 400, 'r', linestyles='dashed', label='1σ')
    plt.hlines(mean_2std, 0, 400, 'r', linestyles='solid', label='2σ')
    
    
    plt.legend(loc='upper right', title="Thresholds")
    plt.show


def intensitylocation(Data, n):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax = axes.ravel()
    fig.suptitle("2nd September 2017", fontsize=16)
    
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)

    ax[0].imshow(Data, cmap=plt.cm.gray, aspect='auto')
    ax[0].set_ylim([200,0])
    ax[0].set_xlim([100,400])
    ax[0].set_xlabel('Time (Second)')
    ax[0].set_ylabel('Frequency (Mhz)')
    ax[0].set_title('Radio Burst Dynamic Spectrum')
    ax[0].hlines(n, 100, 400, 'r', linestyles='dashed', label=("55 Mhz"))
    ax[0].legend(loc='upper right')

    img = burst_intensity(Data, n)
    ax[1].imshow(img, cmap=plt.cm.gray, aspect='auto')
    
    plt.show
    

def data2thresh(Data):

###############################################################################
#                            Otsu Thresholding
###############################################################################
    
    thresh = threshold_otsu(Data)
    print('Threshold Value =', thresh)
    
    binary = Data > thresh
    
    print('-------------------------------')
    print('-------------------------------')
    print('Threshold Image Data Array')
    print(binary)
    print('Threshold Data Type =', type(binary))
    print('Threshold Data dtype =', binary.dtype)
    print('Threshold Data Shape =', binary.shape)
    print('-------------------------------')
    print('-------------------------------')
    
###############################################################################
#                                 Plots
###############################################################################
    
    """
             Original Spectrum to Greyscale to Thresholded Plots
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    ax = axes.ravel()
    
    ax[0] = plt.subplot(2, 2, 1)
    ax[1] = plt.subplot(2, 2, 2)
    ax[2] = plt.subplot(2, 2, 3)
    ax[3] = plt.subplot(2, 2, 4)

    ax[0].imshow(Data, aspect='auto')
    ax[0].set_title('Original')
    ax[0].set_ylim([200,0])
    ax[0].axis('off')
    
    ax[1].imshow(Data, aspect='auto', cmap=plt.cm.gray)
    ax[1].set_title('Greyscale')
    ax[1].set_ylim([200,0])
    ax[1].axis('off')
    
    ax[2].hist(binary.ravel(), bins=256)
    ax[2].set_title('Histogram')
    ax[2].axvline(thresh, color='r')
    ax[2].set_ylim([200,0])
    
    ax[3].imshow(binary, cmap=plt.cm.gray, aspect='auto')
    ax[3].set_title('Thresholded')
    ax[3].set_ylim([200,0])
    ax[3].axis('off')

    plt.savefig('data2thresh.png')
    
    plt.show()


def data2hough(Data, thresh=None):

###############################################################################
#                            Thresholding
###############################################################################
    
    if thresh is None:
        thresh = threshold_mean(Data)
    print('Threshold Value =', thresh)
    
    binary = Data > thresh
    
    print('-------------------------------')
    print('-------------------------------')
    print('Threshold Image Data Array')
    print(binary)
    print('Threshold Data Type =', type(binary))
    print('Threshold Data dtype =', binary.dtype)
    print('Threshold Data Shape =', binary.shape)
    print('-------------------------------')
    print('-------------------------------')
 
###############################################################################
#                        Probabilistic Hough Lines
###############################################################################
    
    lines = probabilistic_hough_line(binary)

#    lines = probabilistic_hough_line(binary, threshold=10, line_length=100,
#                                     line_gap=10, theta=None)    
   
    print('Probabilistic Hough Line Data Type =', type(lines))
    print('-------------------------------')
    print('-------------------------------')

###############################################################################
#                            Straight Hough Line 
###############################################################################
    
    h, theta, d = hough_line(binary, 
                             theta = np.linspace(-np.pi*7 / 36, 
                                                 np.pi*7 / 36, 70))
    
    print('Hough Line h Data Type =', type(h))
    print('Hough Line h Data dtype =', h.dtype)
    print('Hough Line h Data Shape =', h.shape)    
    print('-------------------------------')
    print('-------------------------------')
        
    print('Hough Line theta Data Type =', type(theta))
    print('Hough Line theta Data dtype =', theta.dtype)
    print('Hough Line theta Data Shape =', theta.shape)
    print('-------------------------------')
    print('-------------------------------')
            
    print('Hough Line d Data Type =', type(d))
    print('Hough Line d Data dtype =', d.dtype)
    print('Hough Line d Data Shape =', d.shape)    
    
    print('-------------------------------')
    print('-------------------------------')
    
###############################################################################
#                                 Plots
###############################################################################

    """
         Thresholded to Hough Transform (Probabilistic and Straight Line)
    """
    
    fig, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax2 = axes2.ravel()
    
    ax2[0] = plt.subplot(2, 2, 1)
    ax2[1] = plt.subplot(2, 2, 2)
    ax2[2] = plt.subplot(2, 2, 3)
    ax2[3] = plt.subplot(2, 2, 4)
    
    ax2[0].imshow(binary, cmap=plt.cm.gray, aspect='auto')
    ax2[0].set_title('Thresholded')
    ax2[0].set_ylim([140,10])
    ax2[0].axis('off')
    
    ax2[1].imshow(binary * 0, aspect='auto')
    for line in lines:
        p0, p1 = line
        ax2[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax2[1].set_title('Probabilistic Hough')
    ax2[1].set_ylim([140,10])
    ax2[1].axis('off')
    
    ax2[2].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
               d[-1], d[0]], cmap=plt.cm.gray, aspect='auto')
    ax2[2].set_title('Hough transform')
    ax2[2].set_xlabel('Angles (degrees)')
    ax2[2].set_ylabel('Distance (pixels)')
    
    ax2[3].imshow(binary, cmap=plt.cm.gray, aspect='auto')
    row1, col1 = binary.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        ax2[3].plot((0, col1), (y0, y1), '-r')
        ax2[3].axis((0, col1, row1, 0))
        ax2[3].set_title('Straight Line Hough')
        ax2[3].set_ylim([140,10])
        #ax2[3].set_xlim([22,34])
        #ax2[3].axis('off')
    
    plt.savefig('thresh2hough.png')

    plt.show()

def houghimage(Data, thresh=None):
    
    if thresh is None:
        thresh = threshold_mean(Data) + 2*np.std(Data) 
    print('Threshold Value =', thresh)
    
    binary = Data > thresh
    
    h, theta, d = hough_line(binary, 
                             theta = np.linspace(0, 
                                                 np.pi/4, 90))
    """
    fig2, ax2 = plt.subplots(figsize=(12,8))
    ax2.imshow(binary, cmap=plt.cm.gray)
    ax2.axis('off')
    
    fig1, ax1 = plt.subplots(figsize=(12,8))
   
    ax1.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
               d[-1], d[0]], cmap=plt.cm.gray, aspect='auto')
    ax1.set_title('Hough transform')
    ax1.set_xlabel('Angles (degrees)')
    ax1.set_ylabel('Distance (pixels)')    
    """
    fig, ax = plt.subplots(figsize=(12,8))
    
    ax.imshow(binary, cmap=plt.cm.gray, aspect='auto')
    row1, col1 = binary.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        plt.plot((0, col1), (y0, y1), '-r')
        plt.axis((0, col1, row1, 0))
        plt.ylim([170,0])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Straight Line Hough')
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')    
    #plt.savefig('houghimage.png')   
    
    plt.show()
    

def radonhoughcomp(Data):
    
###############################################################################
#                         Hough and Radon Transforms
###############################################################################

    thresh = threshold_otsu(Data)    
    binary = Data > thresh
    h, theta, d = hough_line(binary)        
    sinogram = radon(binary, circle=False)
    
###############################################################################
#                                 Plots
###############################################################################
    
    fig, axes3 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax3 = axes3.ravel()
    
    ax3[0] = plt.subplot(2, 2, 1)
    ax3[1] = plt.subplot(2, 2, 2)
    ax3[2] = plt.subplot(2, 2, 3)
    ax3[3] = plt.subplot(2, 2, 4)

    ax3[0].imshow(Data, cmap=plt.cm.gray, aspect='auto')
    ax3[0].set_title("Original")
    #ax3[0].set_ylim([200,0])
    ax3[0].axis('off')
    
    ax3[1].imshow(binary, cmap=plt.cm.gray, aspect='auto')
    row1, col1 = binary.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        ax3[1].plot((0, col1), (y0, y1), '-r')
        ax3[1].axis((0, col1, row1, 0))
        ax3[1].set_title('After Transform')
        #ax3[1].set_ylim([200,0])
        ax3[1].axis('off')

    ax3[2].imshow(sinogram, cmap=plt.cm.gray,
               extent=(0,180,0, sinogram.shape[0]), aspect='auto')
    ax3[2].set_title("Radon transform")
    ax3[2].set_xlabel("Angles (degrees)")
    ax3[2].set_ylabel("Distance (pixels)")
    
    ax3[3].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
               0, d[0]*-1], cmap=plt.cm.gray, aspect='auto')
    ax3[3].set_title('Hough transform')
    ax3[3].set_xlabel('Angles (degrees)')
    ax3[3].set_ylabel('Distance (pixels)')

    plt.savefig('radonhoughcomp.png')
    plt.show()
    

def newkirk():
    
    r = np.arange(0,100,0.001 ) / 10. + 1 #r/r_s
    
    n_e_an = 8.3e4 * np.exp( 9.947 / r ) # Active Newkirk

    n_e_pn = 4.2e4 * np.exp( 9.947 / r )  # Passive Newkirk
    
    
    plt.plot(r, n_e_an, 'r', label='Active Region')
    #plt.plot(r, n_e_pn, 'k', label='Passive Region')
    
    plt.xlim(1,2.5)
    plt.ylim(-0.25e9,1.75e9)
    plt.title("Newkirk Models")
    plt.xlabel("Radii (R/R_sun)")
    plt.ylabel("Electron Density (n_e)")
    
    # Adding lines

    plt.hlines(0.1e9, 0, 1.4, 'g', linestyles='dashed',label='Initial Values')
    plt.vlines(1.4, -0.25e9, 0.1e9, 'g', linestyles='dashed')
    
    plt.hlines(0.23e9, 0, 1.25, 'g', linestyles='dashed')
    plt.vlines(1.25, -0.25e9, 0.23e9, 'g', linestyles='dashed')
    
    plt.hlines(0.42e9, 0, 1.16, 'g', linestyles='dashed')
    plt.vlines(1.16, -0.25e9, 0.42e9, 'g', linestyles='dashed')
            
    plt.hlines(0.07e9, 0, 1.45, 'g', linestyles='dashed')
    plt.vlines(1.45, -0.25e9, 0.07e9, 'g', linestyles='dashed')
            
    plt.hlines(0.06e9, 0, 1.49, 'g', linestyles='dashed')
    plt.vlines(1.49, -0.25e9, 0.06e9, 'g', linestyles='dashed')       
            
    plt.hlines(0.09e9, 0, 1.41, 'g', linestyles='dashed')
    plt.vlines(1.41, -0.25e9, 0.09e9, 'g', linestyles='dashed')    
    
    
    plt.hlines(0.01e9, 0, 2.03, 'b', linestyles='dashed', label='Final Values')
    plt.vlines(2.03, -0.25e9,  0.01e9, 'b', linestyles='dashed')

    plt.hlines(0.004e9, 0, 2.4, 'b', linestyles='dashed')
    plt.vlines(2.4, -0.25e9,  0.004e9, 'b', linestyles='dashed')

    plt.hlines(0.004e9, 0, 2.4, 'b', linestyles='dashed')
    plt.vlines(2.4, -0.25e9,  0.004e9, 'b', linestyles='dashed')

    plt.hlines(0.015e9, 0, 1.9, 'b', linestyles='dashed')
    plt.vlines(1.9, -0.25e9,  0.015e9, 'b', linestyles='dashed')

    plt.hlines(0.01e9, 0, 1.9, 'b', linestyles='dashed')
    plt.vlines(1.9, -0.25e9,  0.01e9, 'b', linestyles='dashed')

    plt.hlines(0.003e9, 0, 2.6, 'b', linestyles='dashed')
    plt.vlines(2.6, -0.25e9,  0.003e9, 'b', linestyles='dashed')
    
    # Plotting and saving
    plt.legend(loc='upper right')
    #plt.savefig('newkirkmodels.png')
    plt.show()


###############################################################################
#                              Running Data
###############################################################################

#plot_all() 
#threshcomp(data_short)
#pixel_hist(data_short)
#burst_intensity(data_short, 55)
#intensitylocation(data_short, 55)
#data2thresh(data_short)
#data2hough(data_short)
#houghimage(data_short)
#radonhoughcomp(data_short)
newkirk()
"""
img=Image.open('control_white_line.png').convert('L')
imgdata=np.asarray(img)
houghimage(imgdata)
"""
"""
thresh = threshold_mean(data_short)    
binary = data_short > thresh
plt.imshow(binary)
plt.show
"""

#(4.)*(np.pi**2)*(9.10938356e-31)*(8.854187817620e-12)*(np.e**-2)



