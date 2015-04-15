# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:03:58 2013

@author: pedro.correia
"""

from __future__ import division
import numpy as np

from scipy import linalg
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as Scaler
import pylab as pl
from itertools import cycle
import matplotlib.pyplot as plt

def mean_shift_cluster_analysis(x,y,quantile=0.2,n_samples=1000):
    # ADAPTED FROM:
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#example-cluster-plot-mean-shift-py
    # The following bandwidth can be automatically detected using
    X = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    #print("number of estimated clusters : %d" % n_clusters_)
    colors = 'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk' #cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for i in xrange(len(np.unique(labels))):
        my_members = labels == i
        cluster_center = cluster_centers[i]
        plt.scatter(X[my_members, 0], X[my_members, 1],s=90,c=colors[i],alpha=0.7)
        plt.scatter(cluster_center[0], cluster_center[1],marker='+',s=280,c=colors[i])
    tolx = (X[:,0].max()-X[:,0].min())*0.03
    toly = (X[:,1].max()-X[:,1].min())*0.03
    plt.xlim(X[:,0].min()-tolx,X[:,0].max()+tolx)
    plt.ylim(X[:,1].min()-toly,X[:,1].max()+toly)
    plt.show()
    return labels
    
def affinity_propagation_cluster_analysis(x,y,preference):
    # NOT WORKING BECAUSE I DONT REALLY UNDERSTAND WHAT IT DOES...
    # ADAPTED FROM:
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#example-cluster-plot-affinity-propagation-py
    X = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
    af = AffinityPropagation()
    af = af.fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    #print("number of estimated clusters : %d" % n_clusters_)
    colors = 'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk' #cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for i in xrange(len(np.unique(labels))):
        my_members = labels == i
        cluster_center = X[cluster_centers_indices[i]]
        plt.scatter(X[my_members, 0], X[my_members, 1],s=90,c=colors[i],alpha=0.7)
        plt.scatter(cluster_center[0], cluster_center[1],marker='+',s=280,c=colors[i])
        for j in X[my_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]],c=colors[i],linestyle='--')
    tolx = (X[:,0].max()-X[:,0].min())*0.03
    toly = (X[:,1].max()-X[:,1].min())*0.03
    plt.xlim(X[:,0].min()-tolx,X[:,0].max()+tolx)
    plt.ylim(X[:,1].min()-toly,X[:,1].max()+toly)
    plt.show()
    return labels
    
def DBSCAN_cluster_analysis(x,y,eps,min_samples=10):
    # ADAPTED FROM:
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
    X = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
    X = Scaler().fit_transform(X)

    ##############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples = db.core_sample_indices_
    labels = db.labels_
    
    labels_unique = set(labels) #np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    #print("number of estimated clusters : %d" % n_clusters_)
    colors = 'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk' #cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    #colors = pl.cm.Spectral(np.linspace(0, 1, len(labels_unique)))
    plt.scatter(X[:, 0], X[:, 1],s=30,c='black',alpha=1)
    for i in xrange(len(labels_unique)):
        my_members = labels == i
        #cluster_center = cluster_centers[i]
        plt.scatter(X[my_members, 0], X[my_members, 1],s=90,c=colors[i],alpha=0.7)
        #plt.scatter(cluster_center[0], cluster_center[1],marker='+',s=280,c=colors[i])
    tolx = (X[:,0].max()-X[:,0].min())*0.03
    toly = (X[:,1].max()-X[:,1].min())*0.03
    plt.xlim(X[:,0].min()-tolx,X[:,0].max()+tolx)
    plt.ylim(X[:,1].min()-toly,X[:,1].max()+toly)
    plt.show()
    return labels
    
def Kmeans_cluster_analysis(x,y,n_clusters):
    X = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
    X = Scaler().fit_transform(X)
    km = KMeans(n_clusters)
    km.fit(X)
    labels = km.labels_
    cluster_centers = km.cluster_centers_
    
    labels_unique = set(labels) #np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    #print("number of estimated clusters : %d" % n_clusters_)
    colors = 'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk' #cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    #colors = pl.cm.Spectral(np.linspace(0, 1, len(labels_unique)))
    for i in xrange(len(labels_unique)):
        my_members = labels == i
        cluster_center = cluster_centers[i]
        plt.scatter(X[my_members, 0], X[my_members, 1],s=90,c=colors[i],alpha=0.7)
        plt.scatter(cluster_center[0], cluster_center[1],marker='+',s=280,c=colors[i])
    tolx = (X[:,0].max()-X[:,0].min())*0.03
    toly = (X[:,1].max()-X[:,1].min())*0.03
    plt.xlim(X[:,0].min()-tolx,X[:,0].max()+tolx)
    plt.ylim(X[:,1].min()-toly,X[:,1].max()+toly)
    plt.show()
    return labels

#data = np.loadtxt('EXAMPLE\\GALP\\bidis.prn')
#result = mean_shift_cluster_analysis(data[:,1],data[:,0],0.2,1000)
#result = affinity_propagation_cluster_analysis(data[:,1],data[:,0],-50)
#result = DBSCAN_cluster_analysis(data[:,1],data[:,0],0.3,10)
#result = Kmeans_cluster_analysis(data[:,1],data[:,0],4)
#print result.shape,result

# QUANTILE/EPS
# NUMBER OF SAMPLES/MINIMUM SAMPLES
# NUMBER OF CLUSTERS
