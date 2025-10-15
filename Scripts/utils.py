
import numpy as np
import warnings
warnings.filterwarnings("ignore") 
from math import cos
from math import pi
from csaps import csaps
from sklearn.preprocessing import scale
import pandas as pd 
import numpy as np
from scipy.linalg import sqrtm


# Gershunov Testfunction [sd99,sd95,sd90,sd5,psig]=gsbtest(seriesLength,iters,window,seriesCorrelation,method,obs)

# GSBTEST  'Gershunov' Test of Moving Correlation Significance Levels
#              
#       [sd99,sd95,sd90,sd5,psig]=gsbtest(seriesLength,iters,window,seriesCorrelation,method,obs)
#
# Implementation of a Monte Carlo type simulation of moving correlation functions 
# between two weakly correlated white noise series for the purposes of constructing
# significance test for moving correlation functions based on their standard deviation (Gershunov et al. 2001).
#
# Inputs:
#   seriesLength = length of the time series used in the running correlation analysis
#   iters = number of Monte Carlo iterations to perform (should be >= 1,000)
#   window = width (in units of the time step for the data) of the correlation window
#   seriesCorrelation = desired overall correlation of the simulated time series (simulations will be within 1# of this value)
#   method [optional] = Which method to use to create correlated random series [default = 1], see below
#   obs [optional] = find the significance level for a specific (observed) value of the std of the moving correlation function
#
# Correlation Methods
#   1 = Cholsky decomposition of the covariance matrix [default]
#   2 = Matrix square root
#
# Outputs:
#  sd99  = bootstrapped 99# confidence level
#  sd95  = bootstrapped 95# confidence level
#  sd90  = bootstrapped 90# confidence level
#  sd5   = bootstrapped 5# confidence level  
#  psig  = significance for specific (observed) standard deviation (obs)
#

# Example:
#  > [sd99,sd95]=gsbtest(100,1000,11,0.30,2)
#
# sd99 =
#
#    0.4002
#
# sd95 =
#
#    0.3624
#           
# NOTE: Although the solution should converge for sN >= 1000, small differences (due to the correlation between
# the two artificial time series generated in the script), may still result, especially for a small number (<<10000)
# of simulations; so it is recommended that you verify convergence with several runs and be cautious of values too
# close to your chosen test statistic (particularly at 90# or 95# CI).  You can compare the above result for 'sd95'
# to Table 1 from Gershunov et al. 2001 (for interseries r = 0.30, window length = 11, their 95# confidence level = 0.36).
#
# References
# ----------
# Gershunov, A., N. Schneider, and T. Barnett, 2001. Low-frequency modulation of the ENSO-Indian 
# monsoon rainfall relationship: Signal or noise?, Journal of Climate, 14: 2486-2492.
#

# Update History:
#
# 09 13 -- minor code improvements for clarity; added 'findnearest' as subfunction
# 03.11 -- added calculation of 'psig' using input 'obs' 
# 06.08 -- posted online
# 10.07 -- initial public version
# 09.03 -- initial version
# 
# Function gsbtest.m written by Kevin Anchukaitis
# Subfunction build_win is a modification of the function by David Meko (dmeko@ltrr.arizona.edu)
# Subfunction pairwise is a modification of corrpair.m by David Meko (dmeko@ltrr.arizona.edu)
# Included subfunction 'findnearest' by Tom Benson
# tranlsated to Python and vectorized

def gsbtest_fast(seriesLength, iters, window, seriesCorrelation, method=1, obs=None, seed=None):
    """
    Vectorized Gershunov Test of Moving Correlation Significance Levels.
    """

    rng = np.random.default_rng(seed)

    # correlation matrix
    Rmat = np.array([[1, seriesCorrelation],
                     [seriesCorrelation, 1]])

    if method == 1:
        # Cholesky
        C = np.linalg.cholesky(Rmat)
        Z = rng.standard_normal((iters, 2, seriesLength))
        X = np.einsum("ij,njk->nik", C.T, Z)  # shape (iters, 2, seriesLength)
    elif method == 2:
        # Matrix square root
        C = sqrtm(Rmat)
        Z = rng.standard_normal((iters, 2, seriesLength))
        X = np.einsum("ij,njk->nik", C, Z)
    else:
        raise ValueError("Invalid method, choose 1 (Cholesky) or 2 (matrix sqrt).")

    mcf_std = []
    for i in range(iters):
        x1 = build_win(X[i,0,:], window)
        x2 = build_win(X[i,1,:], window)
        s  = pairwise_corr(x1, x2)
        mcf_std.append(np.std(s))

    mcf_std = np.sort(mcf_std)

    sd99 = mcf_std[int(0.99 * iters) - 1]
    sd95 = mcf_std[int(0.95 * iters) - 1]
    sd90 = mcf_std[int(0.90 * iters) - 1]
    sd5  = mcf_std[int(0.05 * iters) - 1]

    psig = None
    if obs is not None:
    # percentile location of obs in the null distribution
        idx = (np.abs(mcf_std - obs)).argmin()
        psig = idx / len(mcf_std)

    return sd99, sd95, sd90, sd5, psig


def build_win(x, m):
    """Build windowed matrix from time series vector."""
    x = np.asarray(x).flatten()
    mx = len(x)
    if m >= mx:
        raise ValueError("Window width must be shorter than time series!")
    # each row = one window
    return np.array([x[i:i+m] for i in range(mx - m + 1)])

def pairwise_corr(X, Y):
    """Compute correlation for each row in windowed matrices X and Y."""
    out = []
    for i in range(X.shape[0]):
        r = np.corrcoef(X[i, :], Y[i, :])[0, 1]
        out.append(r)
    return np.array(out)
    



##### biweight mean

def tbrm(data, c=9):
    e = 1 * pow(10, -8)
    m = np.median(data)

    s = np.median(getabs(data - m)) # star?

    u = (data - m) / ((c * s) + e)

    w = np.zeros(len(data))

    for i in range(len(u)):
        if abs(u[i]) <= 1:
            w[i] = pow((1 - pow(u[i], 2)), 2)

    return np.sum(w*data)/np.sum(w)

def getabs(data):
    for i in range(len(data)):
        data[i] = abs(data[i])
    
    return data


# The function returns the reconstructed climate variable as a estimated
# from the proxy record as well as statistics that can be used to assess
# the reconstruction skill
# yhat       = reconstructed climate with length of the proxy m x 3
# s1, s2, s3 = structure with regression statistics for MLR, PCR, and CPS respectively


# climate   = n x 1 vector of the climate (observation, predictand)
# cyr       = n x 1 vector of the years corresponding to the climate observations
# proxy     = m x 1 vector of proxy measurements (predictor)
# pyr       = m x 1 vector of the years corresponding to the proxy measurements
# calibrate = either vector of years or a start and end year for the calibration period 
# validate  = either vector of years or a start and end year for the validation period

# make sure proxies are in columns
# Written 2014 by Kevin Anchukaitis (kja@whoi.edu) for MATLAB, Python - 2023 


def simple_cps(climate,cyr,proxy,pyr,calibrate,validate):
    #if type(climate)
    climate.index=cyr
    icalc = np.intersect1d(cyr,calibrate)
    icalp=np.intersect1d(pyr,calibrate)
    ivalc = np.intersect1d(cyr, validate)
    ivalp=np.intersect1d(pyr,validate)

# preallocate
    yhat= np.empty((len(proxy),1))
#CPS  specific
    if proxy.ndim >1:
        proxy = pd.Series(np.mean(scale(proxy),axis=1))
    else:
       proxy=pd.Series(scale(proxy))
    proxy.index=pyr
    mu = np.mean(climate[icalc])
    sig = np.std(climate[icalc],ddof=1)
# isolate from the proxy the mean of that proxy calculated over the calibration period. 
# Retain that calibration period mean and standard deviation to use on full period
    xm=np.mean(proxy[icalp])
    xs=np.std(proxy[icalp],ddof=1)
# remove mean of calibration period, divide by the standard deviation, then
# rescale and add calibration period mean.
    yhat= ((proxy-xm)/xs)*sig + mu
    # VALIDATION
# calibration and validation simple residuals
    s3ec = yhat[icalp] - climate[icalc]; # calibration period residuals
    s3ev = yhat[ivalp] - climate[ivalc]; # validation period residuals
    #  Reduction of Error
# Based on validation residuals vs difference between calibration period climate and calibration period mean
    cbar = np.mean(climate[icalc])

# Calculate s3RE
    s3RE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - cbar) ** 2)
    #s3RE = 1 - sum( (climate(ivalc)-yhat(ivalp,3)).^2) ./ nansum( (climate(icalc)-repmat(cbar,max(size(climate(icalc))),1)).^2);
    
# Coefficient of efficiency
# Based on validation residuals vs difference between validation period climate and validation period mean
    vbar = np.mean(climate[ivalc])
    s3CE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - vbar) ** 2)

    #s3CE = 1 - sum( (climate(ivalc)-yhat(ivalp,3)).^2) ./ nansum( (climate(ivalc)-repmat(vbar,max(size(climate(ivalc))),1)).^2);
# correlation coefficient
    rhov=np.corrcoef(climate[ivalc],yhat[ivalp])
    rhoc = np.corrcoef(climate[icalc],yhat[icalp])
    #s3rhov = rhov[0,1]
    s3R2v  = rhov[0,1] ** 2 # square the off-diagonal element
    s3R2c  = rhoc[0,1] ** 2 # square the off-diagonal element

    return yhat, s3R2v, s3R2c,s3RE, s3CE

def tbrm(data, c=9):
    e = 1 * pow(10, -8)
    m = np.median(data)

    s = np.median(getabs(data - m)) # star?

    u = (data - m) / ((c * s) + e)

    w = np.zeros(len(data))

    for i in range(len(u)):
        if abs(u[i]) <= 1:
            w[i] = pow((1 - pow(u[i], 2)), 2)

    return np.sum(w*data)/np.sum(w)

def getabs(data):
    for i in range(len(data)):
        data[i] = abs(data[i])
    
    return data

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: smoothingspline.py
# Description: This contains the spline method which fits a series to
#              a spline curve.
# example usage (in other file):
# from smoothingspline import spline
# yi = spline(series)



# Returns the spline parameter, given amplitude of the series and the period
def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

# Fits a curve to the series given as input and returns the y-values of the curve
def spline(x, y, period=None):
    if period == None:
        period = len(x) * 0.67
    
    p = get_param(0.5, period)
    yi = csaps(x, y, x, smooth=p)
    detrendy = y - yi
    return detrendy

def varimax_rotation(A, reltol=1e-9, itermax=1000, normalize=False):
    # following Lawley and Maxwell (1962)
  
    if normalize: # this is the 'Kaiser normalization', not the pre-normalization we were talking about above
        h = np.sqrt(np.sum(A**2, axis=1))
        A = A / h[:, np.newaxis]

    # get the size of the selected eigenvectors matrix
    nrows, ncol = A.shape
    
    # set the gamma value for varimax
    gamma = 1.0 
    
    # this is the initial rotation matrix, T, which starts simply as the identity matrix
    T = np.eye(ncol) 
    
    # first 'rotation' is just the dot product of the loadings and the identity matrix
    B = np.dot(A, T) 
    
    var_new = 0
    converged = False

    for k in range(1, itermax + 1): # iterations can be up to itermax
        
        # set the prior variance to the previous step
        var_prior = var_new
        
        # svd of the dot product of the unroated EOFs and the variance of the squared loadings of the current REOF 
        U, S, V = np.linalg.svd(np.dot(A.T, (nrows * B**3 - gamma * B @ np.diag(np.sum(B**2, axis=0)))))
        
        # recalculate the rotation matrix as the dot product of the eigenvectors 
        T = np.dot(U, V)
        
        # the new variance metric is the sum of the diagonals of S
        var_new = np.sum(S)
        
        # recalculate the REOFs as the dot product of the EOFs and new rotation matrix
        B = np.dot(A, T)
        
        # if we are now below the relative tolerance, exit and report the values
        if abs(var_new - var_prior) / var_new < reltol:
            break
    
    # Unnormalize the rotated loadings if necessary
    if normalize:
        B = B * h[:, np.newaxis]
    
    return B, T



def long_cps(climate,cyr,proxy,pyr,calibrate,validate):
    climate.index=cyr
    icalc = np.intersect1d(cyr,calibrate)
    icalp=np.intersect1d(pyr,calibrate)
    ivalc = np.intersect1d(cyr, validate)
    ivalp=np.intersect1d(pyr,validate)
    yhat= np.empty((len(proxy),1))
    if proxy.ndim >1:
        proxy = pd.Series(np.mean(scale(proxy),axis=1))
    else:
       proxy=pd.Series(scale(proxy))
    proxy.index=pyr
    mu = np.mean(climate[icalc])
    sig = np.std(climate[icalc],ddof=1)
    xm=np.mean(proxy[icalp])
    xs=np.std(proxy[icalp],ddof=1)
    yhat= ((proxy-xm)/xs)*sig + mu
    s3ec = yhat[icalp] - climate[icalc]; # calibration period residuals
    s3ev = yhat[ivalp] - climate[ivalc]; # validation period residuals
    cbar = np.mean(climate[icalc])
    s3RE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - cbar) ** 2)
    vbar = np.mean(climate[ivalc])
    s3CE= 1 - np.sum((climate[ivalc] - yhat[ivalp]) ** 2) / np.nansum((climate[ivalc] - vbar) ** 2)
    rhoc = np.corrcoef(climate[icalc],yhat[icalp])
    RSS = np.sum((climate[icalc] - yhat[icalc]) ** 2)
    TSS = np.sum((climate[icalc] - climate[icalc].mean()) ** 2)
    CE = 1 - (RSS / TSS)

    #s3rhov = rhov[0,1]
    #s3R2v  = rhov[0,1] ** 2 # square the off-diagonal element
    s3R2c  = rhoc[0,1] ** 2 # square the off-diagonal element
    return yhat, s3R2c,s3RE, s3CE