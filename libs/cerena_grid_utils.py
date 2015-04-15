# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:23:29 2013

@author: pedro.correia
"""

from __future__ import division
import numpy as np

import scipy.ndimage.filters as scifilt
import scipy.ndimage.fourier as scifourier
import scipy.ndimage.interpolation as sciest
from scipy import interpolate
import scipy.stats as st

def exist_coordinate(x,y,z,coord):
    for i in xrange(x.shape[0]):
        if x[i]==coord[0] and y[i] == coord[1] and z[i] == coord[2]:
            return True
    return False
    
def give_me_value(x,y,z,v,coord):
    dist = np.sqrt((x-coord[0])**2+(y-coord[1])**2+(z-coord[2])**2)
    #print (np.sum(v/(dist**2)))/np.sum(1/dist**2)
    return (np.sum(v/(dist**2)))/np.sum(1/dist**2)

def grid_interpolation(x,y,z,v,blocks,method):
    if method == 'linear':
        flag1 = exist_coordinate(x,y,z,(0,0,0))
        flag2 = exist_coordinate(x,y,z,(blocks[0]-1,0,0))
        flag3 = exist_coordinate(x,y,z,(blocks[0]-1,blocks[1]-1,0))
        flag4 = exist_coordinate(x,y,z,(0,blocks[1]-1,0))
        flag5 = exist_coordinate(x,y,z,(0,0,blocks[2]-1))
        flag6 = exist_coordinate(x,y,z,(blocks[0]-1,0,blocks[2]-1))
        flag7 = exist_coordinate(x,y,z,(blocks[0]-1,blocks[1]-1,blocks[2]-1))
        flag8 = exist_coordinate(x,y,z,(0,blocks[1]-1,blocks[2]-1))
        if not flag1:
            v = np.hstack((v,give_me_value(x,y,z,v,(0,0,0))))
            x = np.hstack((x,np.array([0])))
            y = np.hstack((y,np.array([0])))
            z = np.hstack((z,np.array([0])))
        if not flag2:
            v = np.hstack((v,give_me_value(x,y,z,v,(blocks[0]-1,0,0))))
            x = np.hstack((x,np.array([blocks[0]-1])))
            y = np.hstack((y,np.array([0])))
            z = np.hstack((z,np.array([0])))
        if not flag3:
            v = np.hstack((v,give_me_value(x,y,z,v,(blocks[0]-1,blocks[1]-1,0))))
            x = np.hstack((x,np.array([blocks[0]-1])))
            y = np.hstack((y,np.array([blocks[1]-1])))
            z = np.hstack((z,np.array([0])))
        if not flag4:
            v = np.hstack((v,give_me_value(x,y,z,v,(0,blocks[1]-1,0))))
            x = np.hstack((x,np.array([0])))
            y = np.hstack((y,np.array([blocks[1]-1])))
            z = np.hstack((z,np.array([0])))
        if not flag5:
            v = np.hstack((v,give_me_value(x,y,z,v,(0,0,blocks[2]-1))))
            x = np.hstack((x,np.array([0])))
            y = np.hstack((y,np.array([0])))
            z = np.hstack((z,np.array([blocks[2]-1])))
        if not flag6:
            v = np.hstack((v,give_me_value(x,y,z,v,(blocks[0]-1,0,blocks[2]-1))))
            x = np.hstack((x,np.array([blocks[0]-1])))
            y = np.hstack((y,np.array([0])))
            z = np.hstack((z,np.array([blocks[2]-1])))
        if not flag7:
            v = np.hstack((v,give_me_value(x,y,z,v,(blocks[0]-1,blocks[1]-1,blocks[2]-1))))
            x = np.hstack((x,np.array([blocks[0]-1])))
            y = np.hstack((y,np.array([blocks[1]-1])))
            z = np.hstack((z,np.array([blocks[2]-1])))
        if not flag8:
            v = np.hstack((v,give_me_value(x,y,z,v,(0,blocks[1]-1,blocks[2]-1))))
            x = np.hstack((x,np.array([0])))
            y = np.hstack((y,np.array([blocks[1]-1])))
            z = np.hstack((z,np.array([blocks[2]-1])))
    grid_x,grid_y,grid_z = np.mgrid[0:blocks[0],0:blocks[1],0:blocks[2]]
    points  = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1)),z.reshape((z.shape[0],1))))
    grid = interpolate.griddata(points,v,(grid_x,grid_y,grid_z),method=method,fill_value=-999)
    return grid

def transform_into_point(grid,size,first,dtype='float32'):
    if len(grid.shape)==4:
        columns = grid.shape[3]
        point = np.zeros((np.prod(grid),3+columns),dtype=dtype)
        counter = 0
        for z in xrange(grid.shape[2]):
            for y in xrange(grid.shape[1]):
                for x in xrange(grid.shape[0]):
                    point[counter,0] = x*size[0]+first[0]
                    point[counter,1] = y*size[1]+first[1]
                    point[counter,2] = z*size[2]+first[2]
                    for j in xrange(grid.shape[3]):
                        point[counter,3+j] = grid[x,y,z,j]
    else:
        columns = 1
        point = np.zeros((np.prod(grid),3+columns),dtype=dtype)
        counter = 0
        for z in xrange(grid.shape[2]):
            for y in xrange(grid.shape[1]):
                for x in xrange(grid.shape[0]):
                    point[counter,0] = x*size[0]+first[0]
                    point[counter,1] = y*size[1]+first[1]
                    point[counter,2] = z*size[2]+first[2]
                    point[counter,3] = grid[x,y,z]
    return point
    
def select_by_index(grid,xlimits,ylimits,zlimits):
    return grid[xlimits[0]:xlimits[1],ylimits[0]:ylimits[1],zlimits[0]:zlimits[1]]

def select_by_locations(grid,size,first,xlimits,ylimits,zlimits):
    new_xlimits = (np.int((xlimits[0]-first[0])/size[0]),np.int((xlimits[1]-first[0])/size[0]))
    new_ylimits = (np.int((ylimits[0]-first[1])/size[1]),np.int((ylimits[1]-first[1])/size[1]))
    new_zlimits = (np.int((zlimits[0]-first[2])/size[2]),np.int((zlimits[1]-first[2])/size[2]))
    return grid[new_xlimits[0]:new_xlimits[1],new_ylimits[0]:new_ylimits[1],new_zlimits[0]:new_zlimits[1]]
    
def grid_binarize(grid,signal='above',number=0.5,output_bool=False):
    if output_bool:
        bool_grid = np.zeros((grid.shape[0],grid.shape[1],grid.shape[2]),dtype='bool')
        if signal == 'equal_above':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] >= number: bool_grid[x,y,z] = True
        elif signal == 'above':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] > number: bool_grid[x,y,z] = True
        elif signal == 'equal':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] == number: bool_grid[x,y,z] = True
        elif signal == 'below':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] < number: bool_grid[x,y,z] = True
        elif signal == 'equal_below':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] <= number: bool_grid[x,y,z] = True
    else:
        bool_grid = np.zeros((grid.shape[0],grid.shape[1],grid.shape[2]),dtype='uint8')
        if signal == 'equal_above':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] >= number: bool_grid[x,y,z] = 1
        elif signal == 'above':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] > number: bool_grid[x,y,z] = 1
        elif signal == 'equal':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] == number: bool_grid[x,y,z] = 1
        elif signal == 'below':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] < number: bool_grid[x,y,z] = 1
        elif signal == 'equal_below':
            for x in xrange(bool_grid.shape[0]):
                for y in xrange(bool_grid.shape[1]):
                    for z in xrange(bool_grid.shape[2]):
                        if grid[x,y,z] <= number: bool_grid[x,y,z] = 1
    return bool_grid
    
##########FILTERS AND SMOOTHING#############

def local_isoVariance(grid,window,Vtype):
    filtered = grid.copy()
    if Vtype == 'maximum':
        for x in xrange(grid.shape[0]):
            for y in xrange(grid.shape[1]):
                for z in xrange(grid.shape[2]):
                    window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                    window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                    np.clip(window_inf,0,100000,window_inf)
                    lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                    filtered[x,y,z] = ((lgrid-grid[x,y,z])**2).max()
    elif Vtype == 'minimum':
        for x in xrange(grid.shape[0]):
            for y in xrange(grid.shape[1]):
                for z in xrange(grid.shape[2]):
                    window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                    window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                    np.clip(window_inf,0,100000,window_inf)
                    lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                    appex = (lgrid-grid[x,y,z])**2
                    appex = appex[np.where(appex!=0)]
                    if appex.shape[0] == 0: filtered[x,y,z] = 0
                    else: filtered[x,y,z] = appex.min()
    elif Vtype == 'mean':
        for x in xrange(grid.shape[0]):
            for y in xrange(grid.shape[1]):
                for z in xrange(grid.shape[2]):
                    window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                    window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                    np.clip(window_inf,0,100000,window_inf)
                    lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                    filtered[x,y,z] = ((lgrid-grid[x,y,0])**2).mean()
    return filtered

def local_Gstatistics(grid,window,Wtype):
    filtered = grid.copy()
    m = grid.mean()
    v = grid.var()
    tsum = np.sum((grid-m)/v)
    if Wtype == 'distance':
        W = np.zeros((window[0]*2+1,window[1]*2+1,window[2]*2+1))
        for x in xrange(W.shape[0]):
            for y in xrange(W.shape[1]):
                for z in xrange(W.shape[2]):
                    W[x,y,z] = np.sqrt((x-window[0])**2+(y-window[1])**2+(z-window[2])**2)
        W = 1-W/W.max()
    elif Wtype == 'constant':
        W = np.ones((window[0]*2+1,window[1]*2+1,window[2]*2+1))
    for x in xrange(grid.shape[0]):
        for y in xrange(grid.shape[1]):
            for z in xrange(grid.shape[2]):
                window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                np.clip(window_inf,0,100000,window_inf)
                np.clip(window_sup,[0,0,0],[grid.shape[0],grid.shape[1],grid.shape[2]],window_sup)
                lW = W[window_inf[0]-x+window[0]:window_sup[0]-x+window[0],window_inf[1]-y+window[1]:window_sup[1]-y+window[1],window_inf[2]-z+window[2]:window_sup[2]-z+window[2]]
                lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                lgrid = (lgrid-m)/v
                #filtered[x,y,0] = np.sum(lW*lgrid)/np.sum(lgrid)
                filtered[x,y,z] = np.sum(lW*lgrid)/tsum
    return filtered

def local_gearyC(grid,window,Wtype):
    filtered = grid.copy()
    m = grid.mean()
    v = grid.var()
    if Wtype == 'distance':
        W = np.zeros((window[0]*2+1,window[1]*2+1,window[2]*2+1))
        for x in xrange(W.shape[0]):
            for y in xrange(W.shape[1]):
                for z in xrange(W.shape[2]):
                    W[x,y,z] = np.sqrt((x-window[0])**2+(y-window[1])**2+(z-window[2])**2)
        W = 1-W/W.max()
    elif Wtype == 'constant':
        W = np.ones((window[0]*2+1,window[1]*2+1,window[2]*2+1))
    for x in xrange(grid.shape[0]):
        for y in xrange(grid.shape[1]):
            for z in xrange(grid.shape[2]):
                window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                np.clip(window_inf,0,100000,window_inf)
                np.clip(window_sup,[0,0,0],[grid.shape[0],grid.shape[1],grid.shape[2]],window_sup)
                lW = W[window_inf[0]-x+window[0]:window_sup[0]-x+window[0],window_inf[1]-y+window[1]:window_sup[1]-y+window[1],window_inf[2]-z+window[2]:window_sup[2]-z+window[2]]
                lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                filtered[x,y,z] = np.sum(lW*(grid[x,y,z]-lgrid)**2)/v
    return filtered

def local_moranI(grid,window,Wtype='distance'):
    filtered = grid.copy()
    m = grid.mean()
    v = grid.var()
    if Wtype == 'distance':
        W = np.zeros((window[0]*2+1,window[1]*2+1,window[2]*2+1))
        for x in xrange(W.shape[0]):
            for y in xrange(W.shape[1]):
                for z in xrange(W.shape[2]):
                    W[x,y,z] = np.sqrt((x-window[0])**2+(y-window[1])**2+(z-window[2])**2)
        W = 1-W/W.max()
    elif Wtype == 'constant':
        W = np.ones((window[0]*2+1,window[1]*2+1,window[2]*2+1))
    for x in xrange(grid.shape[0]):
        for y in xrange(grid.shape[1]):
             for z in xrange(grid.shape[2]):
                window_inf = np.array([x-window[0],y-window[1],z-window[2]])
                window_sup = np.array([x+window[0],y+window[1],z+window[2]])
                np.clip(window_inf,0,100000,window_inf)
                np.clip(window_sup,[0,0,0],[grid.shape[0],grid.shape[1],grid.shape[2]],window_sup)
                lW = W[window_inf[0]-x+window[0]:window_sup[0]-x+window[0],window_inf[1]-y+window[1]:window_sup[1]-y+window[1],window_inf[2]-z+window[2]:window_sup[2]-z+window[2]]
                lgrid = grid[window_inf[0]:window_sup[0],window_inf[1]:window_sup[1],window_inf[2]:window_sup[2]]
                filtered[x,y,z] = ((grid[x,y,z]-m)/v)*np.sum((grid[x,y,z]-lgrid)*lW)/np.prod(grid.shape)
    return filtered    

def equalization_procedure(grid):
    filtered = (grid[:,:,:]-grid.min())*(255/(grid.max()-grid.min()))
    imhist,bins = np.histogram(filtered,256,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    filtered = np.int_(np.interp(filtered.flatten(),bins[:-1],cdf)).reshape(grid.shape)
    return filtered
    
def sigmoid_procedure(grid):
    filtered = (grid[:,:,:]-grid.min())*(255/(grid.max()-grid.min()))
    alpha = filtered.max()-filtered.min()
    beta  = 256
    filtered = 255*(1/(1+np.e**((filtered-beta)/alpha)))
    return filtered

def uniform_filter(grid,window_size = (3,3,3)):
    filtered = grid.copy()
    scifilt.uniform_filter(grid,window_size,filtered, mode='nearest')
    return filtered
    
def gaussian_filter(grid,sigma = (1,1,1),order=0):
    filtered = grid.copy()
    scifilt.gaussian_filter(grid,sigma,order,filtered, mode='nearest')
    return filtered
    
def order_statistics_filter(grid,window_size=(3,3,3),statistics_type='median',rank=1):
    filtered = grid.copy()
    if statistics_type=='minimum':
        scifilt.minimum_filter(grid,window_size,None,filtered, mode='nearest')
    elif statistics_type=='maximum':
        scifilt.maximum_filter(grid,window_size,None,filtered, mode='nearest')
    elif statistics_type=='median':
        scifilt.median_filter(grid,window_size,None,filtered, mode='nearest')
    elif statistics_type[:-2]=='percentile' or statistics_type[:-2]=='per':
        per = np.int(statistics_type[-2:])
        scifilt.percentile_filter(grid,per,window_size,None,filtered, mode='nearest')
    elif statistics_type=='rank':
        scifilt.rank_filter(grid,rank,window_size,None,filtered, mode='nearest')
    return filtered
        
    return filtered
    
def prewitt_filter(grid,axis):
    filtered = grid.copy()
    scifilt.prewitt(grid,axis,filtered, mode='nearest')
    return filtered
    
def sobel_filter(grid,axis):
    filtered = grid.copy()
    scifilt.sobel(grid,axis,filtered, mode='nearest')
    return filtered
    
def laplace_filter(grid):
    filtered = grid.copy()
    scifilt.laplace(grid,filtered, mode='nearest')
    return filtered
    
def gaussian_laplace_filter(grid,sigma=(1,1,1)):
    filtered = grid.copy()
    scifilt.gaussian_laplace(grid,sigma,filtered, mode='nearest')
    return filtered
    
def fourier_uniform_filter(grid,window_size=(3,3,3),axis=-1,n=-1):
    filtered = grid.copy()
    scifourier.fourier_uniform(grid,window_size,n,axis,filtered)
    return filtered
    
def fourier_gaussian_filter(grid,sigma_size=(3,3,3),axis=-1,n=-1):
    filtered = grid.copy()
    scifourier.fourier_uniform(grid,sigma_size,n,axis,filtered)
    return filtered
    
def fourier_shift_filter(grid,shift_size=(3,3,3),axis=-1,n=-1):
    filtered = grid.copy()
    scifourier.fourier_shift(grid,shift_size,n,axis,filtered)
    return filtered
    
def fourier_ellipsoid_filter(grid,window_size=(3,3,3),axis=-1,n=-1):
    filtered = grid.copy()
    scifourier.fourier_ellipsoid(grid,window_size,n,axis,filtered)
    return filtered
    
def spline_filter(grid,order=3):
    filtered = grid.copy()
    sciest.spline_filter(grid,order,filtered)
    return filtered
    
def moving_window_atribute(data,atribute='mean',window=(3,3,3),percentile=50,rank=1,clip_limits=(0,1),fisher=True,border_mode='nearest'):
    #reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’
    # minimum,maximum,median,percentile,rank,mean,variance,std,clip,sum,product,peak2peak,signal2noise,skewness
    # kurtosis
    if atribute == 'minimum':
        atrib = data.copy()
        scifilt.minimum_filter(data,window,None,atrib, mode=border_mode)
        return atrib
    elif atribute == 'maximum':
        atrib = data.copy()
        scifilt.maximum_filter(data,window,None,atrib, mode=border_mode)
        return atrib
    elif atribute == 'median':
        atrib = data.copy()
        scifilt.median_filter(data,window,None,atrib, mode=border_mode)
        return atrib
    elif atribute == 'percentile':
        atrib = data.copy()
        scifilt.percentile_filter(data,percentile,window,None,atrib, mode=border_mode)
        return atrib
    elif atribute == 'rank':
        atrib = data.copy()
        scifilt.rank_filter(data,rank,window,None,atrib, mode=border_mode)
        return atrib
    elif atribute == 'mean':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].mean()
        return atrib
    elif atribute == 'variance':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].var()
        return atrib
    elif atribute == 'std':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].std()
        return atrib
    elif atribute == 'clip':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    m = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].flatten()
                    l0 = np.percentile(m,clip_limits[0])
                    l1 = np.percentile(m,clip_limits[1])
                    m.clip(l0,l1)
                    atrib[i,j,k] = np.percentile(m,50)
        return atrib
    elif atribute == 'sum':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].sum()
        return atrib
    elif atribute == 'product':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].prod()
        return atrib
    elif atribute == 'peak2peak':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    atrib[i,j,k] = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].ptp()
        return atrib
    elif atribute == 'signal2noise':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    m = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].mean()
                    v = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].std()
                    atrib[i,j,k] = m/v
        return atrib
    elif atribute == 'skewness':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    m = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].flatten()
                    v = st.skew(m)
                    atrib[i,j,k] = v
        return atrib
    elif atribute == 'kurtosis':
        atrib = data.copy()
        blocks = atrib.shape
        for i in xrange(blocks[0]):
            for j in xrange(blocks[1]):
                for k in xrange(blocks[2]):
                    m = data[np.clip(i-window[0],0,blocks[0]):i+window[0]+1,np.clip(j-window[1],0,blocks[1]):j+window[1]+1,np.clip(k-window[2],0,blocks[2]):k+window[2]+1].flatten()
                    v = st.kurtosis(m,fisher=fisher)
                    atrib[i,j,k] = v
        return atrib
    else:
        return False

def grid_upscale(data,choice,steps):
    blocks = data.shape
    xsize = blocks[0]/steps[0]
    if xsize-int(xsize)!=0: xsize = int(xsize)+1
    ysize = blocks[1]/steps[1]
    if ysize-int(ysize)!=0: ysize = int(ysize)+1
    zsize = blocks[2]/steps[2]
    if zsize-int(zsize)!=0: zsize = int(zsize)+1
    atrib = np.zeros((xsize,ysize,zsize))
    if choice == 'mean':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = data[x:x+steps[0],y:y+steps[1],z:z+steps[2]].mean()
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    elif choice == 'minimum':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = data[x:x+steps[0],y:y+steps[1],z:z+steps[2]].min()
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    elif choice == 'maximum':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = data[x:x+steps[0],y:y+steps[1],z:z+steps[2]].max()
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    elif choice == 'p25':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = np.percentile(data[x:x+steps[0],y:y+steps[1],z:z+steps[2]],25)
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    elif choice == 'p50':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = np.percentile(data[x:x+steps[0],y:y+steps[1],z:z+steps[2]],50)
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    elif choice == 'p75':
        xf = 0
        for x in xrange(0,blocks[0],steps[0]):
            yf = 0
            for y in xrange(0,blocks[1],steps[1]):
                zf = 0
                for z in xrange(0,blocks[2],steps[2]):
                    atrib[xf,yf,zf] = np.percentile(data[x:x+steps[0],y:y+steps[1],z:z+steps[2]],75)
                    zf = zf + 1
                yf = yf + 1
            xf = xf + 1
    return atrib