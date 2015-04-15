# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:19:11 2014

@author: pedro.correia
"""

from __future__ import division
#import sys
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')
import numpy as np
import random
import matplotlib.pyplot as plt
import subprocess
import os
import wx

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import BivariateSpline
from scipy import signal
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

from scipy.linalg import solve_triangular,solve
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.spatial import cKDTree as KDTree
import scipy.stats as ST
import time

def per_values(d):
    #res = np.zeros(100,dtype='float32')
    #res[0] = d.min()
    #res[-1] = d.max()
    #for i in xrange(1,100):
    return np.percentile(d,range(101))

def get_percentile_dist(mean,d,number,var,mi,ma):
    #'g - gaussian     e - exponential     t - triangular     r - rayleigh
    #v - von mises     u - uniform     l - lognormal     w - weibull
    #p - power     pa - pareto     b - beta     c - chi square
    #f - fisher      gm - gamma     gb - gumbel'
    def linear_transform(this,minimum,maximum):
        return ((this-this.min())*(maximum-minimum))/(this.max()-this.min())+minimum
    if d=='g':
        return per_values(np.random.normal(mean,var,number))
    elif d=='e':
        return per_values(linear_transform(np.random.exponential(var,number),mean-var,mean+var))
    elif d=='t':
        return per_values(np.random.triangular(mean-var,mean,mean+var,number))
    elif d=='r':
        return per_values(linear_transform(np.random.rayleigh(var,number),mean-var,mean+var))
    elif d=='v':
        return per_values(linear_transform(np.random.vonmises(mean,var,number),mean-var,mean+var))
    elif d=='u':
        return per_values(np.random.uniform(mean-var,mean+var,number))
    elif d=='l':
        return per_values(np.random.lognormal(mean,var,number))
    elif d=='w':
        return per_values(linear_transform(np.random.weibull(var,number),mean-var,mean+var))
    elif d=='p':
        return per_values(linear_transform(np.random.power(var,number),mean-var,mean+var))
    elif d=='pa':
        return per_values(linear_transform(np.random.pareto(var,number),mean-var,mean+var))
    elif d=='b':
        return per_values(linear_transform(np.random.beta(mean,var,number),mean-var,mean+var))
    elif d=='c':
        return per_values(linear_transform(np.random.chisquare(number),mean-var,mean+var))
    #elif d=='f':
    #    return per_values(np.random.f(mean,var,number))
    #elif d=='gm':
    #    return per_values(np.random.gamma((mean,var,number))
    #elif d=='gb':
    #    return per_values(np.random.normal(mean,var,number))

class trace():
    def __init__(self,coord):
        self.coord = coord
        self.seismic = None
        self.ip      = None
        self.changed_ip = None
        self.reflectivity = None
        self.synthetic = None
        self.changed_synthetic = None
        self.current_cc = None
        self.seismic_residues = None
        self.variogram_residues = None
        self.switch_probability = None
        self.vres = None
        self.variogram = None
        self.global_distribution = None
        self.wavelet = None
        self.myccs = []
        self.myvres = []
        self.mysres = []
        self.myaccepted = []
        self.residue = None
        self.O1 = None
        
    def save2file(self,path,odir):
        res = np.zeros((self.ip.shape[0],6))
        res[:,0] = self.coord[0]
        res[:,1] = self.coord[1]
        res[:,2] = np.arange(self.ip.shape[0])
        res[:,3] = self.ip[:]
        res[:,4] = self.synthetic[:]
        res[:,5] = self.seismic[:]
        np.savetxt(odir+'\\IP'+path,res,fmt='%10.3f')
        
    def perform_switch(self,ite,total):    
        p = np.random.uniform()
        if p <= self.switch_probability:
            ids = random.sample(xrange(0,self.ip.shape[0]), 2)
            self.changed_ip = self.ip.copy()
            a = self.changed_ip[ids[0]]
            b = self.changed_ip[ids[1]]
            self.changed_ip[ids[0]] = b
            self.changed_ip[ids[1]] = a
        else:
            i = np.random.randint(0,self.global_distribution.shape[0])
            self.changed_ip = self.ip.copy()
            wi = np.random.randint(0,self.changed_ip.shape[0])
            self.changed_ip[wi] = self.global_distribution[i]
        self.changed_synthetic = self.convolve_changed_trace(self.wavelet)
        r1 = np.sum(self.seismic_residues)
        r2 = np.sum(np.abs(self.seismic-self.changed_synthetic))
        v1 = self.residue
        #O1 = self.O1
        v2,O2 = self.calculate_changed_variogram_values()
        self.mysres.append(r2)
        self.myvres.append(v2)
        c=((ite+1)/total) # LINEAR
        #c= np.clip(1-5*(np.e**ite/(np.e**total)),0,1)
        prob = 1-np.e**(-1*np.abs(self.O1-O2)/c)
        #prob = 1-(ite+1)/total
        p2   = np.random.uniform()
        #print self.O1,O2,p2,prob
        #print self.O1,O2,prob,p2
        if p2 < prob:
            self.ip = self.changed_ip.copy()
            self.synthetic = self.changed_synthetic.copy()
            self.calculate_seismic_residues()
            self.residue, self.O1 = self.calculate_variogram_values()
            self.myaccepted.append(2)
        else:
            #print r1,r2,v1,v2
            #raw_input()
            if (r1-r2)>=0 and (v1-v2)>=0:
                self.ip = self.changed_ip.copy()
                self.synthetic = self.changed_synthetic.copy()
                self.calculate_seismic_residues()
                self.residue, self.O1 = self.calculate_variogram_values()
                self.myaccepted.append(1)
            else:
                self.myaccepted.append(0)
        self.myccs.append(self.do_pearson_correlation(self.synthetic,self.seismic))
        
    def calculate_variogram(self,model,rang):
        var = np.arange(self.seismic.shape[0])
        if model == 'Exponential':
            var = (1-np.e**(-3*var/rang))
        else:
            var = (1-np.e**(-3*var/rang))
        return var
       
    def calculate_seismic_residues(self):
        self.seismic_residues = np.abs(self.seismic-self.synthetic)
        
    def calculate_changed_variogram_values(self):
        var = np.zeros(self.changed_ip.shape[0])
        counter = np.zeros(self.changed_ip.shape[0])
        for i in xrange(var.shape[0]):
            c=0
            for j in xrange(i,var.shape[0]):
                var[c] = (self.changed_ip[i]-self.changed_ip[j])**2
                counter[c] = counter[c] + 1
                c=c+1
        var = var/(2*counter)
        var = var/self.changed_ip.var()
        var2 = np.sum(((var[1:self.vres.shape[0]]-self.variogram[1:self.vres.shape[0]])**2)/self.variogram[1:self.vres.shape[0]])
        var = np.abs(var-self.variogram)
        res = np.sum(var[:self.vres.shape[0]]*self.vres[:])
        return res,var2
        
    def calculate_variogram_values(self):
        var = np.zeros(self.ip.shape[0])
        counter = np.zeros(self.ip.shape[0])
        for i in xrange(var.shape[0]):
            c=0
            for j in xrange(i,var.shape[0]):
                var[c] = (self.ip[i]-self.ip[j])**2
                counter[c] = counter[c] + 1
                c=c+1
        var = var/(2*counter)
        var = var/self.ip.var()
        var2 = np.sum(((var[1:self.vres.shape[0]]-self.variogram[1:self.vres.shape[0]])**2)/self.variogram[1:self.vres.shape[0]])
        var = np.abs(var-self.variogram)
        res = np.sum(var[:self.vres.shape[0]]*self.vres[:])
        return res,var2
        
    def do_pearson_correlation(self,a,b):
        return np.sum((a-a.mean())*(b-b.mean()))/np.sqrt(np.sum((a-a.mean())**2)*np.sum((b-b.mean())**2))
        
    def get_seismic_trace(self,seismic):
        self.seismic = seismic[self.coord[0],self.coord[1],:]
        
    def get_ip_trace(self,ip):
        self.ip = ip[self.coord[0],self.coord[1],:]
        self.synthetic = self.convolve_trace(self.wavelet)
        self.calculate_seismic_residues()
        self.residue,self.O1 = self.calculate_variogram_values()
        
    def convolve_changed_trace(self,wavelet):
        self.reflectivity = self.changed_ip.copy()
        for i in xrange(self.changed_ip.shape[0]-1):
            self.reflectivity[i]=(self.reflectivity[i+1]-self.reflectivity[i])/(self.reflectivity[i+1]+self.reflectivity[i])
        self.reflectivity[-1]=0 #grid[:,:,-2]
        synthetic = self.changed_ip.copy()
        synthetic[:] = 0
        h_size=(wavelet.shape[0]-1)/2
        for i in xrange(self.changed_ip.shape[0]):
            if i-h_size<0:
                wa=h_size-i
                a=0
            else:
                wa=0
                a=i-h_size
            if i+h_size>self.changed_ip.shape[0]:
                wb=h_size+i-self.changed_ip.shape[0]
                b=self.ip.shape[0]
            else:
                wb=2*h_size+1
                b=i+h_size
            synthetic[a:b]=synthetic[a:b]+self.reflectivity[i]*wavelet[wa:(2*h_size-wb)]
        return synthetic  
        
    def convolve_trace(self,wavelet):
        self.reflectivity = self.ip.copy()
        for i in xrange(self.ip.shape[0]-1):
            self.reflectivity[i]=(self.reflectivity[i+1]-self.reflectivity[i])/(self.reflectivity[i+1]+self.reflectivity[i])
        self.reflectivity[-1]=0 #grid[:,:,-2]
        synthetic = self.ip.copy()
        synthetic[:] = 0
        h_size=(wavelet.shape[0]-1)/2
        for i in xrange(self.ip.shape[0]):
            if i-h_size<0:
                wa=h_size-i
                a=0
            else:
                wa=0
                a=i-h_size
            if i+h_size>self.ip.shape[0]:
                wb=h_size+i-self.ip.shape[0]
                b=self.ip.shape[0]
            else:
                wb=2*h_size+1
                b=i+h_size
            synthetic[a:b]=synthetic[a:b]+self.reflectivity[i]*wavelet[wa:(2*h_size-wb)]
        return synthetic
        
class trace_zones():
    def __init__(self,coord):
        self.coord = coord
        self.seismic = None
        self.ip      = None
        self.changed_ip = None
        self.reflectivity = None
        self.synthetic = None
        self.changed_synthetic = None
        self.current_cc = None
        self.seismic_residues = None
        self.variogram_residues = None
        self.switch_probability = None
        self.vres = None
        self.variogram = None
        self.global_distribution = None
        self.local_distributions = None
        self.wavelet = None
        self.myccs = []
        self.myvres = []
        self.mysres = []
        self.myaccepted = []
        self.residue = None
        self.O1 = None
        self.zones = None
        self.uniques = None
        
    def calculate_uniques(self):
        self.uniques = np.unique(self.zones)
        
    def save2file(self,path,odir):
        res = np.zeros((self.ip.shape[0],6))
        res[:,0] = self.coord[0]
        res[:,1] = self.coord[1]
        res[:,2] = np.arange(self.ip.shape[0])
        res[:,3] = self.ip[:]
        res[:,4] = self.synthetic[:]
        res[:,5] = self.seismic[:]
        np.savetxt(odir+'\\IP'+path,res,fmt='%10.3f')
        
    def perform_switch(self,ite,total):    
        p = np.random.uniform()
        uni = self.uniques[np.random.randint(0,len(self.uniques))]
        if p <= self.switch_probability:
            #ids = random.sample(xrange(0,self.ip.shape[0]), 2)
            ind = np.where(self.ip==uni)
            if len(ind[0])<2:
                i = np.random.randint(0,len(self.local_distributions[uni]))
                self.changed_ip = self.ip.copy()
                wi = np.random.randint(0,self.changed_ip.shape[0])
                #self.changed_ip[wi] = self.global_distribution[i]
                self.changed_ip[wi] = self.local_distributions[uni][i]
            else:    
                ids = (ind[0][np.random.randint(0,len(ind[0]),1)],ind[0][np.random.randint(0,len(ind[0]),1)])
                self.changed_ip = self.ip.copy()
                a = self.changed_ip[ids[0]]
                b = self.changed_ip[ids[1]]
                self.changed_ip[ids[0]] = b
                self.changed_ip[ids[1]] = a
        else:
            #i = np.random.randint(0,self.global_distribution.shape[0])
            i = np.random.randint(0,len(self.local_distributions[uni]))
            self.changed_ip = self.ip.copy()
            wi = np.random.randint(0,self.changed_ip.shape[0])
            #self.changed_ip[wi] = self.global_distribution[i]
            self.changed_ip[wi] = self.local_distributions[uni][i]
        self.changed_synthetic = self.convolve_changed_trace(self.wavelet)
        r1 = np.sum(self.seismic_residues)
        r2 = np.sum(np.abs(self.seismic-self.changed_synthetic))
        v1 = self.residue
        #O1 = self.O1
        v2,O2 = self.calculate_changed_variogram_values()
        self.mysres.append(r2)
        self.myvres.append(v2)
        c=((ite+1)/total) # LINEAR
        #c= np.clip(1-5*(np.e**ite/(np.e**total)),0,1)
        prob = 1-np.e**(-1*np.abs(self.O1-O2)/c)
        #prob = 1-(ite+1)/total
        p2   = np.random.uniform()
        #print self.O1,O2,p2,prob
        #print self.O1,O2,prob,p2
        if p2 < prob:
            self.ip = self.changed_ip.copy()
            self.synthetic = self.changed_synthetic.copy()
            self.calculate_seismic_residues()
            self.residue, self.O1 = self.calculate_variogram_values()
            self.myaccepted.append(2)
        else:
            #print r1,r2,v1,v2
            #raw_input()
            if (r1-r2)>=0 and (v1-v2)>=0:
                self.ip = self.changed_ip.copy()
                self.synthetic = self.changed_synthetic.copy()
                self.calculate_seismic_residues()
                self.residue, self.O1 = self.calculate_variogram_values()
                self.myaccepted.append(1)
            else:
                self.myaccepted.append(0)
        self.myccs.append(self.do_pearson_correlation(self.synthetic,self.seismic))
        
    def calculate_variogram(self,model,rang):
        var = np.arange(self.seismic.shape[0])
        if model == 'Exponential':
            var = (1-np.e**(-3*var/rang))
        else:
            var = (1-np.e**(-3*var/rang))
        return var
       
    def calculate_seismic_residues(self):
        self.seismic_residues = np.abs(self.seismic-self.synthetic)
        
    def calculate_changed_variogram_values(self):
        var = np.zeros(self.changed_ip.shape[0])
        counter = np.zeros(self.changed_ip.shape[0])
        for i in xrange(var.shape[0]):
            c=0
            for j in xrange(i,var.shape[0]):
                var[c] = (self.changed_ip[i]-self.changed_ip[j])**2
                counter[c] = counter[c] + 1
                c=c+1
        var = var/(2*counter)
        var = var/self.changed_ip.var()
        var2 = np.sum(((var[1:self.vres.shape[0]]-self.variogram[1:self.vres.shape[0]])**2)/self.variogram[1:self.vres.shape[0]])
        var = np.abs(var-self.variogram)
        res = np.sum(var[:self.vres.shape[0]]*self.vres[:])
        return res,var2
        
    def calculate_variogram_values(self):
        var = np.zeros(self.ip.shape[0])
        counter = np.zeros(self.ip.shape[0])
        for i in xrange(var.shape[0]):
            c=0
            for j in xrange(i,var.shape[0]):
                var[c] = (self.ip[i]-self.ip[j])**2
                counter[c] = counter[c] + 1
                c=c+1
        var = var/(2*counter)
        var = var/self.ip.var()
        var2 = np.sum(((var[1:self.vres.shape[0]]-self.variogram[1:self.vres.shape[0]])**2)/self.variogram[1:self.vres.shape[0]])
        var = np.abs(var-self.variogram)
        res = np.sum(var[:self.vres.shape[0]]*self.vres[:])
        return res,var2
        
    def do_pearson_correlation(self,a,b):
        return np.sum((a-a.mean())*(b-b.mean()))/np.sqrt(np.sum((a-a.mean())**2)*np.sum((b-b.mean())**2))
        
    def get_seismic_trace(self,seismic):
        self.seismic = seismic[self.coord[0],self.coord[1],:]
        
    def get_ip_trace(self,ip,zones):
        self.ip = ip[self.coord[0],self.coord[1],:]
        self.zones = zones[self.coord[0],self.coord[1],:]
        self.synthetic = self.convolve_trace(self.wavelet)
        self.calculate_seismic_residues()
        self.residue,self.O1 = self.calculate_variogram_values()
        
    def convolve_changed_trace(self,wavelet):
        self.reflectivity = self.changed_ip.copy()
        for i in xrange(self.changed_ip.shape[0]-1):
            self.reflectivity[i]=(self.reflectivity[i+1]-self.reflectivity[i])/(self.reflectivity[i+1]+self.reflectivity[i])
        self.reflectivity[-1]=0 #grid[:,:,-2]
        synthetic = self.changed_ip.copy()
        synthetic[:] = 0
        h_size=(wavelet.shape[0]-1)/2
        for i in xrange(self.changed_ip.shape[0]):
            if i-h_size<0:
                wa=h_size-i
                a=0
            else:
                wa=0
                a=i-h_size
            if i+h_size>self.changed_ip.shape[0]:
                wb=h_size+i-self.changed_ip.shape[0]
                b=self.ip.shape[0]
            else:
                wb=2*h_size+1
                b=i+h_size
            synthetic[a:b]=synthetic[a:b]+self.reflectivity[i]*wavelet[wa:(2*h_size-wb)]
        return synthetic  
        
    def convolve_trace(self,wavelet):
        self.reflectivity = self.ip.copy()
        for i in xrange(self.ip.shape[0]-1):
            self.reflectivity[i]=(self.reflectivity[i+1]-self.reflectivity[i])/(self.reflectivity[i+1]+self.reflectivity[i])
        self.reflectivity[-1]=0 #grid[:,:,-2]
        synthetic = self.ip.copy()
        synthetic[:] = 0
        h_size=(wavelet.shape[0]-1)/2
        for i in xrange(self.ip.shape[0]):
            if i-h_size<0:
                wa=h_size-i
                a=0
            else:
                wa=0
                a=i-h_size
            if i+h_size>self.ip.shape[0]:
                wb=h_size+i-self.ip.shape[0]
                b=self.ip.shape[0]
            else:
                wb=2*h_size+1
                b=i+h_size
            synthetic[a:b]=synthetic[a:b]+self.reflectivity[i]*wavelet[wa:(2*h_size-wb)]
        return synthetic
        
class read_par():
    def __init__(self,seismic,ai,model,rang,sycoord,wavelet,ites,distribution,sprobability,vres):
        self.seismic = seismic
        self.ai = ai
        self.nodes   = self.seismic.shape                       # Este é interno
        self.model   = model                   # 4 modelo # Struture type ;1=spherical,2=exponential,3=gaussian (IT(i))
        self.range   = rang     # 6 ranges variograma
        self.sycoord = sycoord  # 7 Caminho para o ficheiro de coordenada dos sintéticos.
        self.wavelet = wavelet  # 8 Caminho da wavelet.
        self.ites    = ites                   # 10 número de iteraçoes
        self.distribution = distribution # 11 distributiçao para simular
        self.sprobability = sprobability # 12 probabilidade da troca ser interna.
        self.vres = vres # 13 pesos dos residuos de variograma
        self.sy_min = self.seismic.min()
        self.sy_max = self.seismic.max()
        self.sy_traces = self.load_sy_traces()
        
    def load_sy_traces(self):
        dic = {}
        for i in xrange(self.sycoord.shape[0]):
            dic[i] = trace(self.sycoord[i,:])
            dic[i].get_seismic_trace(self.seismic)
            dic[i].switch_probability = self.sprobability
            dic[i].vres = self.vres
            dic[i].global_distribution = self.distribution.copy()
            dic[i].wavelet = self.wavelet.copy()
            dic[i].variogram = dic[i].calculate_variogram(self.model,self.range)
        return dic
        
class read_par_zones():
    def __init__(self,seismic,ai,model,rang,sycoord,wavelet,ites,distribution,sprobability,vres,zones):
        self.seismic = seismic
        self.ai = ai
        self.zones = zones
        self.nodes   = self.seismic.shape                       # Este é interno
        self.model   = model                   # 4 modelo # Struture type ;1=spherical,2=exponential,3=gaussian (IT(i))
        self.range   = rang     # 6 ranges variograma
        self.sycoord = sycoord  # 7 Caminho para o ficheiro de coordenada dos sintéticos.
        self.wavelet = wavelet  # 8 Caminho da wavelet.
        self.ites    = ites                   # 10 número de iteraçoes
        self.distribution = distribution # 11 distributiçao para simular
        self.sprobability = sprobability # 12 probabilidade da troca ser interna.
        self.vres = vres # 13 pesos dos residuos de variograma
        self.sy_min = self.seismic.min()
        self.sy_max = self.seismic.max()
        self.local_distributions = self.local_distributions()
        self.sy_traces = self.load_sy_traces()
        
    def local_distributions(self):
        unique = np.unique(self.zones)
        ldists = {}
        for i in unique:
            ind = np.where(self.zones==i)
            ldists[i] = np.percentile(self.ai[ind].flatten(),range(0,101))
        return ldists
        
    def load_sy_traces(self):
        dic = {}
        for i in xrange(self.sycoord.shape[0]):
            dic[i] = trace_zones(self.sycoord[i,:])
            dic[i].get_seismic_trace(self.seismic)
            dic[i].switch_probability = self.sprobability
            dic[i].vres = self.vres
            dic[i].global_distribution = self.distribution.copy()
            dic[i].local_distributions = self.local_distributions
            dic[i].wavelet = self.wavelet.copy()
            dic[i].variogram = dic[i].calculate_variogram(self.model,self.range)
        return dic
        
class run_procedure():
    def __init__(self,par,odir):
        for i in par.sy_traces.keys():
            par.sy_traces[i].get_ip_trace(par.ai)
        synth_models = {}
        ip_models = {}
        for i in par.sy_traces.keys():
            synth_models[i] = np.zeros((par.nodes[2],par.ites))
            ip_models[i] = np.zeros((par.nodes[2],par.ites))
            synth_models[i][:,0] = par.sy_traces[i].synthetic[:]
            ip_models[i][:,0] = par.sy_traces[i].ip[:]
        dialog = wx.ProgressDialog ( 'Progress', 'Doing procedure.', maximum = par.ites-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
        for i in xrange(par.ites):
            for j in par.sy_traces.keys():
                par.sy_traces[j].perform_switch(i,par.ites)
                synth_models[j][:,i] = par.sy_traces[j].synthetic[:]
                ip_models[j][:,i] = par.sy_traces[j].ip[:]
            dialog.Update ( i, 'Step...'+'  '+repr(i)+'   of   '+repr(par.ites-1) )
        for i in synth_models.keys():
            np.save(odir+'\\evolve'+str(i)+'.npy',synth_models[i])
            np.save(odir+'\\ipevolve'+str(i)+'.npy',ip_models[i])
        for j in par.sy_traces.keys():
            par.sy_traces[j].save2file(str(j)+'.prn',odir)
        ccresult = np.zeros((par.ites+1,par.sycoord.shape[0]))
        ccresult[0,:] = np.arange(1,par.sycoord.shape[0]+1)[:]
        vvresult=ccresult.copy()
        ssresult=ccresult.copy()
        aaresult=ccresult.copy()
        for i in par.sy_traces.keys():
            ccresult[1::,i] = par.sy_traces[i].myccs[:]
            vvresult[1::,i] = par.sy_traces[i].myvres[:]
            ssresult[1::,i] = par.sy_traces[i].mysres[:]
            aaresult[1::,i] = par.sy_traces[i].myaccepted[:]
        self.ccresult = ccresult
        self.vvresult = vvresult
        self.ssresult = ssresult
        self.aaresult = aaresult
        np.savetxt(odir+'\\cc_table.prn',ccresult,fmt='%10.3f')
        np.savetxt(odir+'\\vr_table.prn',vvresult,fmt='%10.3f')
        np.savetxt(odir+'\\sr_table.prn',ssresult,fmt='%10.3f')
        np.savetxt(odir+'\\aa_table.prn',aaresult,fmt='%10.3f')
        full = len(ip_models.keys())*ip_models[0].shape[0]
        isize = ip_models[0].shape[0]
        self.ips = np.zeros((full,4),dtype='float32')
        c=0
        for i in ip_models.keys():
            self.ips[c:c+isize,-1] = ip_models[i][:,-1]
            c = c + isize
        self.ips[:,0] = 0
        self.ips[:,1] = 0
        self.ips[:,2] = np.arange(self.ips.shape[0])
        dialog.Destroy()
        
class run_procedure_zones():
    def __init__(self,par,odir):
        for i in par.sy_traces.keys():
            par.sy_traces[i].get_ip_trace(par.ai,par.zones)
            par.sy_traces[i].calculate_uniques()
        synth_models = {}
        ip_models = {}
        for i in par.sy_traces.keys():
            synth_models[i] = np.zeros((par.nodes[2],par.ites))
            ip_models[i] = np.zeros((par.nodes[2],par.ites))
            synth_models[i][:,0] = par.sy_traces[i].synthetic[:]
            ip_models[i][:,0] = par.sy_traces[i].ip[:]
        dialog = wx.ProgressDialog ( 'Progress', 'Doing procedure.', maximum = par.ites-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
        for i in xrange(par.ites):
            for j in par.sy_traces.keys():
                par.sy_traces[j].perform_switch(i,par.ites)
                synth_models[j][:,i] = par.sy_traces[j].synthetic[:]
                ip_models[j][:,i] = par.sy_traces[j].ip[:]
            dialog.Update ( i, 'Step...'+'  '+repr(i)+'   of   '+repr(par.ites-1) )
        for i in synth_models.keys():
            np.save(odir+'\\evolve'+str(i)+'.npy',synth_models[i])
            np.save(odir+'\\ipevolve'+str(i)+'.npy',ip_models[i])
        for j in par.sy_traces.keys():
            par.sy_traces[j].save2file(str(j)+'.prn',odir)
        ccresult = np.zeros((par.ites+1,par.sycoord.shape[0]))
        ccresult[0,:] = np.arange(1,par.sycoord.shape[0]+1)[:]
        vvresult=ccresult.copy()
        ssresult=ccresult.copy()
        aaresult=ccresult.copy()
        for i in par.sy_traces.keys():
            ccresult[1::,i] = par.sy_traces[i].myccs[:]
            vvresult[1::,i] = par.sy_traces[i].myvres[:]
            ssresult[1::,i] = par.sy_traces[i].mysres[:]
            aaresult[1::,i] = par.sy_traces[i].myaccepted[:]
        self.ccresult = ccresult
        self.vvresult = vvresult
        self.ssresult = ssresult
        self.aaresult = aaresult
        np.savetxt(odir+'\\cc_table.prn',ccresult,fmt='%10.3f')
        np.savetxt(odir+'\\vr_table.prn',vvresult,fmt='%10.3f')
        np.savetxt(odir+'\\sr_table.prn',ssresult,fmt='%10.3f')
        np.savetxt(odir+'\\aa_table.prn',aaresult,fmt='%10.3f')
        full = len(ip_models.keys())*ip_models[0].shape[0]
        isize = ip_models[0].shape[0]
        self.ips = np.zeros((full,4),dtype='float32')
        c=0
        for i in ip_models.keys():
            self.ips[c:c+isize,-1] = ip_models[i][:,-1]
            c = c + isize
        self.ips[:,0] = 0
        self.ips[:,1] = 0
        self.ips[:,2] = np.arange(self.ips.shape[0])
        dialog.Destroy()
        
def create_discrete_synthetic_channels(blocks=(100,100,30),number=1,thickness=(5,15),startin='x',step=10,delta=7,deltaz=3,seed=12345):
    if seed!=0: np.random.seed(seed)
    result = np.zeros(blocks,dtype='uint8')
    for c in xrange(1,number+1):
        if startin=='x':
            xo = np.random.randint(0,blocks[0])
            yl = np.int_(np.linspace(0,blocks[1],step))
            xl = [xo]
            zo = np.random.randint(0,blocks[2])
            zl = [zo]
            th = np.random.randint(thickness[0],thickness[1])
            tv = np.random.randint(thickness[0],thickness[1])
            thl = [th]
            tvl = [tv]
            for x in xrange(1,yl.shape[0]):
                xl.append(np.random.randint(xl[x-1]-delta,xl[x-1]+delta))
                zl.append(np.random.randint(zl[x-1]-deltaz,zl[x-1]+deltaz))
                thl.append(np.random.randint(thickness[0],thickness[1]))
                tvl.append(np.random.randint(thickness[0],thickness[1]))
            fx = interp1d(yl,xl,kind='cubic')
            fz = interp1d(yl,zl,kind='cubic')
            fh = interp1d(yl,thl,kind='cubic')
            fv = interp1d(yl,tvl,kind='cubic')
            yf = np.arange(blocks[1])
            xf = np.int_(fx(yf))
            zf = np.int_(fz(yf))
            thf = np.int_(fh(yf))
            tvf = np.int_(fv(yf))
            ind = np.where((zf>0) & (zf<blocks[2]) & (xf>0) & (xf<blocks[0]))
            result[xf[ind],yf[ind],zf[ind]]=c
            for i in xrange(xf[ind].shape[0]):
                indices = np.indices((np.clip(xf[ind][i]+thf[ind][i],0,blocks[0])-np.clip(xf[ind][i]-thf[ind][i],0,blocks[0]),np.clip(zf[ind][i]+tvf[ind][i],0,blocks[2])-np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2])))
                mask = (((indices[0]+np.clip(xf[ind][i]-thf[ind][i],0,blocks[0])-xf[ind][i])**2)/thf[ind][i]**2+((indices[1]+np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2])-zf[ind][i])**2)/tvf[ind][i]**2<=1)
                result[np.clip(xf[ind][i]-thf[ind][i],0,blocks[0]):np.clip(xf[ind][i]+thf[ind][i],0,blocks[0]),i,np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2]):np.clip(zf[ind][i]+tvf[ind][i],0,blocks[2])][mask]=c
        elif startin=='y':
            yo = np.random.randint(0,blocks[1])
            xl = np.int_(np.linspace(0,blocks[0],step))
            yl = [yo]
            zo = np.random.randint(0,blocks[2])
            zl = [zo]
            th = np.random.randint(thickness[0],thickness[1])
            tv = np.random.randint(thickness[0],thickness[1])
            thl = [th]
            tvl = [tv]
            for y in xrange(1,xl.shape[0]):
                yl.append(np.random.randint(yl[y-1]-delta,yl[y-1]+delta))
                zl.append(np.random.randint(zl[y-1]-deltaz,zl[y-1]+deltaz))
                thl.append(np.random.randint(thickness[0],thickness[1]))
                tvl.append(np.random.randint(thickness[0],thickness[1]))
            fx = interp1d(xl,yl,kind='cubic')
            fz = interp1d(xl,zl,kind='cubic')
            fh = interp1d(xl,thl,kind='cubic')
            fv = interp1d(xl,tvl,kind='cubic')
            xf = np.arange(blocks[0])
            yf = np.int_(fx(xf))
            zf = np.int_(fz(xf))
            thf = np.int_(fh(xf))
            tvf = np.int_(fv(xf))
            ind = np.where((zf>0) & (zf<blocks[2]) & (yf>0) & (yf<blocks[1]))
            result[xf[ind],yf[ind],zf[ind]]=c           
            for i in xrange(xf[ind].shape[0]):
                indices = np.indices((np.clip(yf[ind][i]+thf[ind][i],0,blocks[1])-np.clip(yf[ind][i]-thf[ind][i],0,blocks[1]),np.clip(zf[ind][i]+tvf[ind][i],0,blocks[2])-np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2])))
                mask = (((indices[0]+np.clip(yf[ind][i]-thf[ind][i],0,blocks[1])-yf[ind][i])**2)/thf[ind][i]**2+((indices[1]+np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2])-zf[ind][i])**2)/tvf[ind][i]**2<=1)
                result[i,np.clip(yf[ind][i]-thf[ind][i],0,blocks[1]):np.clip(yf[ind][i]+thf[ind][i],0,blocks[1]),np.clip(zf[ind][i]-tvf[ind][i],0,blocks[2]):np.clip(zf[ind][i]+tvf[ind][i],0,blocks[2])][mask]=c
    return result
    
def create_discrete_synthetic_layers(blocks=(100,100,30),number=1,thickness=(3,7),step=10,seed=12345):
    if seed!=0: np.random.seed(seed)
    result = np.zeros(blocks,dtype='uint8')
    x = np.random.randint(0,blocks[1],step)
    y = np.random.randint(0,blocks[1],step)
    x = np.hstack((x,np.array([-1,-1,blocks[0],blocks[0]])))
    y = np.hstack((y,np.array([-1,blocks[1],-1,blocks[1]])))
    z = np.random.randint(thickness[0],thickness[1],step+4)
    indices = np.indices(blocks)
    surf0 = np.int_(griddata(x,y,z,np.arange(blocks[0]),np.arange(blocks[1])))
    for zc in xrange(result.shape[2]):
        mask = indices[2][:,:,zc]<=surf0
        result[:,:,zc][mask] = 1
    for i in xrange(2,number+1):
        z = z + np.random.randint(thickness[0],thickness[1],step+4)
        surf1 = np.int_(griddata(x,y,z,np.arange(blocks[0]),np.arange(blocks[1])))
        for zc in xrange(result.shape[2]):
            mask = ((indices[2][:,:,zc]>surf0) & (indices[2][:,:,zc]<=surf1))
            result[:,:,zc][mask] = i
        surf0 = surf1.copy()
    return result
            
def create_synthetic_structure(omesh,blocks,stype='sin_fold',startin='x',fraction=0.5,radius=10,center=15,reflective=False):
    mesh = np.mgrid[0:blocks[0],0:blocks[1]]
    if startin=='x':
        if stype=='sin_fold':
            z = np.clip(np.int_(np.sin((mesh[0]/fraction)*np.pi/180)*radius+center),0,blocks[2])
        elif stype=='cos_fold':
            z = np.clip(np.int_(np.cos((mesh[0]/fraction)*np.pi/180)*radius+center),0,blocks[2])
    elif startin=='y':
        if stype=='sin_fold':
            z = np.clip(np.int_(np.sin((mesh[1]/fraction)*np.pi/180)*radius+center),0,blocks[2])
        elif stype=='cos_fold':
            z = np.clip(np.int_(np.cos((mesh[1]/fraction)*np.pi/180)*radius+center),0,blocks[2])
    result = omesh.copy()
    if reflective:
        for x in xrange(result.shape[0]):
            for y in xrange(result.shape[0]):
                top = np.clip(z[x,y],0,blocks[2])
                result[x,y,top:] = omesh[x,y,:blocks[2]-top]
                result[x,y,:top] = omesh[x,y,blocks[2]-top:blocks[2]][::-1]
    else:
        for x in xrange(result.shape[0]):
            for y in xrange(result.shape[0]):
                top = np.clip(z[x,y],0,blocks[2])
                result[x,y,top:] = omesh[x,y,:blocks[2]-top]
                result[x,y,:top] = omesh[x,y,blocks[2]-top:blocks[2]]
    return result
    
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()
    
def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = im.copy()
    for i in xrange(im.shape[2]):
        improc[:,:,i] = signal.convolve(im[:,:,i],g, mode='same')
    #for i in xrange(im.shape[0]):
    #    improc[i,:,:] = signal.convolve(im[i,:,:],g, mode='same')
    return improc
    
def getnextpos(pos,maxwalk,maxsize,seed):
    np.random.seed(seed)
    x=-1
    y=-1
    npos=[-1,-1]
    flag=True
    #print maxsize
    while flag:
        x=np.random.randint(1,maxwalk+1)
        y=np.random.randint(-maxwalk,maxwalk+1)
        npos=[pos[0]+x,pos[1]+y]
        
        if npos[0]<0 or npos[1]<0 or npos[0]>maxsize[0]-1 or npos[1]>maxsize[1]-1:
            #print npos
            flag=True
        else:
            flag=False
    return npos
    
def do_random_sneeze(blocks,start,noise,seed,times=20,maxwalk=2):
    np.random.seed(seed)
    res=np.zeros(blocks,dtype='float32')
    starty=[]
    startz=[]
    for i in xrange(start):
        starty.append(np.random.randint(1,blocks[1]-1))
        startz.append(np.random.randint(0,blocks[2]))
    #print startz
    c = 0
    for j in starty:
        res[0,j,startz[c]]=1
        for i in xrange(times):
            pos=[0,j]
            for i in xrange(blocks[0]+50):
                pos=getnextpos(pos,maxwalk,res.shape,seed)
                seed=seed+3
                res[pos[0],pos[1],startz[c]]=1
                if pos[0]==res.shape[0]-1:
                    break
        c=c+1
    res=blur_image(res,noise)
    return res
    
def blur_image2(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='same')
    return(improc)
    
def do_random_islands(size,delta,thick,step,number,noise,seed):
    res=np.zeros((size[0],size[1]))
    np.random.seed(seed)
    for i in xrange(number):
        x=np.random.randint(0,size[0]-1)
        y=np.random.randint(0,size[1]-1)
        a=np.random.normal(x,delta[0],100).astype('int')
        b=np.random.normal(y,delta[1],100).astype('int')
        res[a[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))],b[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))]]=1
    blur = blur_image2(res,noise)
    blur = ((blur/blur.max())*size[2]+3).astype('int32')
    mesh = create_discrete_synthetic_layers(size,number,thick,step,seed)
    unit = mesh.max()+1
    for x in xrange(size[0]):
        for y in xrange(size[1]):
            zin = np.clip(blur[x,y],0,size[2]-1)
            mesh[x,y,:zin] = unit
    return mesh
    
def do_random_islands2(size,delta,thick,step,number,noise,seed):
    res=np.zeros((size[0],size[1]))
    np.random.seed(seed)
    for i in xrange(number):
        x=np.random.randint(0,size[0]-1)
        y=np.random.randint(0,size[1]-1)
        a=np.random.normal(x,delta[0],100).astype('int')
        b=np.random.normal(y,delta[1],100).astype('int')
        res[a[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))],b[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))]]=1
    blur = blur_image2(res,noise)
    blur = ((blur/blur.max())*size[2]+3).astype('int32')
    mesh = create_discrete_synthetic_layers(size,number,thick,step,seed)
    unit = mesh.max()+1
    for x in xrange(size[0]):
        for y in xrange(size[1]):
            zin = np.clip(blur[x,y],0,size[2]-1)
            f = size[2]-zin
            mesh[x,y,zin:] = mesh[x,y,:f]
            mesh[x,y,:zin] = unit
    return mesh
            

def build_synthetic_mesh(blocks,size,first,stype,number,step,thick,delta,startin,seed,fraction,reflective,allequal):
    # ['channels','layers','sin_fold','cos_fold'] # PLUS intrusion, faults
    if stype=='layers':
        mesh = create_discrete_synthetic_layers(blocks,number,thick,step,seed)
    elif stype =='channels':
        mesh = create_discrete_synthetic_channels(blocks,number,thick,startin,step,delta[0],delta[1],seed)
        if allequal: mesh[np.where(mesh[:,:,:]>=1)]=1
    elif stype =='delta':
        mesh = do_random_sneeze(blocks,number,step,seed,thick[1],thick[0])*100
        mesh[np.where(mesh<=fraction)]=0
        mesh[np.where(mesh>fraction)] =1
    elif stype =='intrusion':
        mesh = do_random_islands2(blocks,delta,thick,step,number,step,seed)
    elif stype =='stromatolites':
        mesh = do_random_islands(blocks,delta,thick,step,number,step,seed)
    elif stype in ['sin_fold','cos_fold']:
        mesh = create_discrete_synthetic_layers(blocks,number,thick,step,seed)
        mesh = create_synthetic_structure(mesh,blocks,stype,startin,fraction/100,step,int(blocks[2]/2),reflective)
    else:
        return False
    return mesh
    
def build_continuous_variable(mesh,distributions,wind=2,filter_flag=True,seed=1234567):
    np.random.seed(seed)
    c=0
    #print distributions
    for i in np.unique(mesh):
        ind = np.where(mesh==i)
        mesh[ind] = np.random.normal(distributions[c][0],distributions[c][1],ind[0].shape[0])[:]
        c=c+1
    if filter_flag:
        cmesh = mesh.copy().astype('float32')
        for z in xrange(mesh.shape[2]):
            for y in xrange(mesh.shape[1]):
                for x in xrange(mesh.shape[0]):
                    ax = np.clip(x-wind,0,mesh.shape[0])
                    ay = np.clip(y-wind,0,mesh.shape[1])
                    az = np.clip(z-wind,0,mesh.shape[2])
                    cmesh[x,y,z] = mesh[ax:x+wind,ay:y+wind,az:z+wind].mean()
        return cmesh
    else:
        return mesh
        
def build_multiple_continuous_variable(mesh,distributions,cc1,cc2,tflag,wind=2,filter_flag=True,seed=1234567):
    np.random.seed(seed)
    c=0
    if tflag: imesh = np.zeros((mesh.shape[0],mesh.shape[1],mesh.shape[2],3),dtype='float32')
    else: imesh = np.zeros((mesh.shape[0],mesh.shape[1],mesh.shape[2],2),dtype='float32')
    #print distributions
    for i in np.unique(mesh):
        ind = np.where(mesh==i)
        imesh[ind[0],ind[1],ind[2],0] = np.random.normal(distributions[c][0],distributions[c][1],ind[0].shape[0])[:]
        var = imesh[ind[0],ind[1],ind[2],0].max()-imesh[ind[0],ind[1],ind[2],0].min()
        mi  = imesh[ind[0],ind[1],ind[2],0].min()
        ma  = imesh[ind[0],ind[1],ind[2],0].max()
        #print var,mi,ma
        # (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
        middle = (ma-mi)/2+mi
        #print (1-abs(cc1[c]))*var,cc1[c]
        for j in xrange(ind[0].shape[0]):
            imesh[ind[0][j],ind[1][j],ind[2][j],1] = np.random.normal(imesh[ind[0][j],ind[1][j],ind[2][j],0],(1-abs(cc1[c]/100))*var,1)
            # 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            if cc1[i] < 0:  imesh[ind[0][j],ind[1][j],ind[2][j],1] = 2*middle-imesh[ind[0][j],ind[1][j],ind[2][j],1]
            if tflag:
                imesh[ind[0][j],ind[1][j],ind[2][j],2] = np.random.normal(imesh[ind[0][j],ind[1][j],ind[2][j],0],(1-abs(cc2[c]/100))*var,1)
                if cc2[i] < 0:  imesh[ind[0][j],ind[1][j],ind[2][j],2] = 2*middle-imesh[ind[0][j],ind[1][j],ind[2][j],2]              
        c=c+1
    if filter_flag:
        cmesh = imesh.copy().astype('float32')
        for z in xrange(mesh.shape[2]):
            for y in xrange(mesh.shape[1]):
                for x in xrange(mesh.shape[0]):
                    ax = np.clip(x-wind,0,mesh.shape[0])
                    ay = np.clip(y-wind,0,mesh.shape[1])
                    az = np.clip(z-wind,0,mesh.shape[2])
                    cmesh[x,y,z,0] = imesh[ax:x+wind,ay:y+wind,az:z+wind,0].mean()
                    cmesh[x,y,z,1] = imesh[ax:x+wind,ay:y+wind,az:z+wind,1].mean()
                    if tflag: cmesh[x,y,z,2] = imesh[ax:x+wind,ay:y+wind,az:z+wind,2].mean()
        return cmesh
    else:
        return mesh
        
def wavelet_convolve(mesh,wave):
    """
    if wave.shape[0]%2==0:
        wave_mesh = np.zeros((mesh.shape[0],mesh.shape[1],wave.shape[0]-1),dtype='float32')
        for i in xrange(mesh.shape[0]):
            for j in xrange(mesh.shape[1]):
                wave_mesh[i,j,:]=wave[:-1]
        h_size=int((wave.shape[0])/2)
    else:
    """
    #if wave.shape[0]%2==0: wopi = 1
    #else: wopi = 0
    wave_mesh = np.zeros((mesh.shape[0],mesh.shape[1],wave.shape[0]),dtype='float32')
    for i in xrange(mesh.shape[0]):
        for j in xrange(mesh.shape[1]):
            wave_mesh[i,j,:]=wave[:]
    h_size=int((wave.shape[0])/2)
    for i in xrange(mesh.shape[2]-1):
        mesh[:,:,i]=(mesh[:,:,i+1]-mesh[:,:,i])/(mesh[:,:,i+1]+mesh[:,:,i])
    mesh[:,:,-1]=0 #grid[:,:,-2]
    synth=np.zeros(mesh.shape).astype('float32')
    for i in xrange(mesh.shape[2]):
        if i-h_size<0:
            wa=h_size-i
            a=0
        else:
            wa=0
            a=i-h_size
        if i+h_size>mesh.shape[2]:
            wb=h_size+i-mesh.shape[2]
            b=mesh.shape[2]
        else:
            wb=2*h_size+1
            b=i+h_size-1
        #print synth[:,:,a:b].shape,mesh[:,:,i][:,:,np.newaxis].shape,wave_mesh[:,:,wa:(2*h_size-wb)].shape
        synth[:,:,a:b]=synth[:,:,a:b]+mesh[:,:,i][:,:,np.newaxis]*wave_mesh[:,:,wa:wa+synth[:,:,a:b].shape[2]] #(2*h_size-wb-wopi)]
    return synth
    
def ricker_wavelet(f, size, dt=1):
    t = np.int_(np.linspace(-size, size, (2*size+1)/dt))
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    data = np.hstack((t[:,np.newaxis],y[:,np.newaxis]))
    return data
    
def draw_ricker_wavelet(f,size,dt=1):
    t = np.linspace(-size, size, 1000)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    plt.plot(t,y,color='red')
    plt.xlim(t.min(),t.max())
    toly = (y.max()-y.min())*0.05
    plt.ylim(y.min()-toly,y.max()+toly)
    plt.plot([0,0],[y.min(),y.max()],color='black',linestyle='dashed')
    plt.show()
    
def __do_covariance_lookup__(model,amp,angle,nugget,sill,ratio1,ratio2):
    angle = angle - 90
    if ratio1 == 1:
        ratio1 = 0.999
    if model == 'Exponential':
        cov_table = np.zeros((np.int(amp)*2+1,np.int(amp),np.int(amp*ratio2)),dtype='float32')
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                dist = np.sqrt(cx**2+cy**2)
                ang = np.arctan2(cx,cy)
                #angZ = np.arctan2(0,cx)
                ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                #ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                cov_table[x+amp,y,0] = sill-(nugget+(sill*(1-np.e**(-3*dist/ran))))/sill
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                for z in xrange(1,np.int(amp*ratio2)):
                    cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                    cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                    dist = np.sqrt(cx**2+cy**2+z**2)
                    dist2 = np.sqrt(cx**2+cy**2)
                    ang = np.arctan2(cx,cy)
                    ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                    angZ = np.arctan2(z,1)
                    ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                    cov_table[x+amp,y,z] = sill-(nugget+(sill*(1-np.e**(-3*dist/ran2))))/sill
    elif model == 'Gaussian':
        cov_table = np.zeros((np.int(amp)*2+1,np.int(amp),np.int(amp*ratio2)),dtype='float32')
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                dist = np.sqrt(cx**2+cy**2)
                ang = np.arctan2(cx,cy)
                #angZ = np.arctan2(0,cx)
                ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                #ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                cov_table[x+amp,y,0] = sill-(nugget+(sill * (1 - np.e**(-3 * (dist/ran)**2))))/sill
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                for z in xrange(1,np.int(amp*ratio2)):
                    cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                    cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                    dist = np.sqrt(cx**2+cy**2+z**2)
                    ang = np.arctan2(cx,cy)
                    angZ = np.arctan2(z,1)
                    ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                    ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                    cov_table[x+amp,y,z] = sill-(nugget+(sill * (1 - np.e**(-3 * (dist/ran2)**2))))/sill
    elif model == 'Spheric':
        cov_table = np.zeros((np.int(amp)*2+1,np.int(amp),np.int(amp*ratio2)),dtype='float32')
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                dist = np.sqrt(cx**2+cy**2)
                ang = np.arctan2(cx,cy)
                #angZ = np.arctan2(0,cx)
                ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                #ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                if dist < ran:
                    cov_table[x+amp,y,0] = sill-(nugget+(sill * (1.5*dist/ran-0.5*(dist/ran)**3)))/sill
                else: cov_table[x+amp,y,0] = 0
        for x in xrange(-amp,amp+1):
            for y in xrange(0,amp):
                for z in xrange(1,np.int(amp*ratio2)):
                    cx = (x)*np.cos(angle*np.pi/180)-y*np.sin(angle*np.pi/180)
                    cy = (x)*np.sin(angle*np.pi/180)+y*np.cos(angle*np.pi/180)
                    dist = np.sqrt(cx**2+cy**2+z**2)
                    ang = np.arctan2(cx,cy)
                    angZ = np.arctan2(z,1)
                    ran = np.sqrt((amp*ratio1*np.cos(ang))**2+(amp*np.sin(ang))**2)
                    ran2 = np.sqrt((ran*ratio2*np.sin(angZ))**2+(ran*np.cos(angZ))**2)
                    if dist < ran2:
                        cov_table[x+amp,y,z] = sill-(nugget+(sill * (1.5*dist/ran2-0.5*(dist/ran2)**3)))/sill
                    else: cov_table[x+amp,y,z] = 0
    return cov_table

def __normal_score_transform__(data):
    n = data.shape[0] #len(data)
    pk = (np.arange(1, n + 1) - 0.5) / n
    normscore = ST.norm.ppf(pk)   
    pk_s = np.zeros_like(pk)
    pk_s[np.argsort(data)] = pk
    normal_score = np.zeros_like(data)
    normal_score[np.argsort(data)] = normscore.astype('float32')
    normal_score.sort()    
    return pk,normal_score
    
def __make_random_walk__(indices,mask,seed=123456789):
    np.random.seed(seed)
    s = np.random.permutation(np.prod(indices[0][~mask].shape))
    return indices[0][~mask][s],indices[1][~mask][s],indices[2][~mask][s]
    
def __search__(loc,indices,m,window,number):   
    i0 = indices[0][np.clip(loc[0]-window[0],0,100000000):loc[0]+window[0],np.clip(loc[1]-window[1],0,100000000):loc[1]+window[1],np.clip(loc[2]-window[2],0,100000000):loc[2]+window[2]][m].flatten()
    i1 = indices[1][np.clip(loc[0]-window[0],0,100000000):loc[0]+window[0],np.clip(loc[1]-window[1],0,100000000):loc[1]+window[1],np.clip(loc[2]-window[2],0,100000000):loc[2]+window[2]][m].flatten()
    i2 = indices[2][np.clip(loc[0]-window[0],0,100000000):loc[0]+window[0],np.clip(loc[1]-window[1],0,100000000):loc[1]+window[1],np.clip(loc[2]-window[2],0,100000000):loc[2]+window[2]][m].flatten()
    #return i0-loc[0]-np.clip(loc[0]-window[0],0,100000000),i1-loc[1]-np.clip(loc[1]-window[1],0,100000000),i2-loc[2]-np.clip(loc[2]-window[2],0,100000000)
    return i0,i1,i2
    
def __mok__(K,M,values,number):
    W = solve(K,M)
    res = np.zeros((values.shape[0],number),dtype='uint8')
    for r in xrange(number):
        res[np.where((values==r+1))[0],r] = 1
    result = np.zeros(res.shape[1],dtype='float32')
    for i in xrange(res.shape[1]): result[i] = np.dot(res[:,i], W[:-1])
    return result
    
def __ok__(K,M,cvalues):
    W = solve(K,M)
    return np.dot(cvalues,W[:-1])[0],np.sum(np.multiply(W[:-1],(M[:-1])))
    
def direct_simulation(minimum,maximum,value_data, nscores, z_krig, var_krig, tries_number):
    if z_krig < minimum:
        y_krig = nscores.min()
    elif z_krig > maximum:
        y_krig = nscores.max()
    else:
        y_krig = np.interp(z_krig, value_data, nscores)
    tries = np.zeros(tries_number)
    for t in xrange(tries_number):
        p = np.random.random()
        y_sim = ST.norm.ppf(q=p, loc=0, scale=1)
        z_sim = np.interp(y_sim * var_krig + y_krig, nscores, value_data)     
        tries[t] = z_sim
    simulated = z_sim + z_krig - tries.mean()
    return simulated    

def cbis(opath,ipoint,blpoint,blocks,error_flag=False,error=None,nodes=4,bnodes=4,model='Exponential',amp=30,nugget=0,ratio1=1,ratio2=1,angle=0,simulations=1,seed=12345,tries=3,size=(1,1,1),first=(0,0,0)):
    #blocks = cobject.blocks
    #size   = (1,1,1) #cobject.size
    #first  = (0,0,0) #cobject.first
    igrid = np.zeros(blocks,dtype='uint8')
    #cgrid = np.zeros(blocks,dtype='float32')
    ix=ipoint[:,0]
    iy=ipoint[:,1]
    iz=ipoint[:,2]
    iv=ipoint[:,-1]
    igrid[np.int_((ix-first[0])/size[0]),np.int_((iy-first[1])/size[1]),np.int_((iz-first[2])/size[2])] = iv
    #mgrid = cgrid.copy()
    idistribution = iv
    unique = np.unique(idistribution)
    #di_all = {}
    bdistribution = blpoint[:,-1]
    bx = blpoint[:,0]
    by = blpoint[:,1]
    bz = blpoint[:,2]
    #nsrs = []
    #for i in unique:
    #    a,b = __normal_score_transform__(di_all[i])
    #    nsrs.append(b)
    mask = np.bool_(igrid)
    xb,yb,zb = blocks[0],blocks[1],blocks[2]
    x,y,z = np.indices(igrid.shape)
    window = (int(xb/5)+1,int(yb/5)+1,int(zb/5)+1)
    sill = 1
    ctable = __do_covariance_lookup__(model,amp,angle,nugget,sill,ratio1,ratio2)
    middle = np.int(ctable.shape[0]/2)
    blocks_tree = KDTree(np.hstack((np.int_((bx[:,np.newaxis]-first[0])/size[0]),np.int_((by[:,np.newaxis]-first[0])/size[0]),np.int_((bz[:,np.newaxis]-first[0])/size[0]))),13)
    np.random.seed(seed)
    seeds = np.random.randint(1000000,9999999,simulations)
    total = np.prod(blocks)
    for s in xrange(simulations):
        np.random.seed(seeds[s])
        random_walk = __make_random_walk__((x,y,z),mask)
        for i in xrange(random_walk[0].shape[0]):
            window = (np.clip(int(window[0]/2),1,xb),np.clip(int(window[1]/2),1,yb),np.clip(int(window[2]/2),1,zb))
            m = mask[np.clip(random_walk[0][i]-window[0],0,100000000):random_walk[0][i]+window[0],np.clip(random_walk[1][i]-window[1],0,100000000):random_walk[1][i]+window[1],np.clip(random_walk[2][i]-window[2],0,100000000):random_walk[2][i]+window[2]] == True
            while np.count_nonzero(m)<nodes:
                window = (window[0]*2,window[1]*2,window[2]*2)
                m = mask[np.clip(random_walk[0][i]-window[0],0,100000000):random_walk[0][i]+window[0],np.clip(random_walk[1][i]-window[1],0,100000000):random_walk[1][i]+window[1],np.clip(random_walk[2][i]-window[2],0,100000000):random_walk[2][i]+window[2]] == True
            #########
            # K   Cb
            # Cb  Kb
            #########
            indexes = __search__((random_walk[0][i],random_walk[1][i],random_walk[2][i]),(x,y,z),m,window,nodes)           
            ir1 = pdist(indexes[0][:,np.newaxis]).astype('int32')+middle
            ir2 = pdist(indexes[1][:,np.newaxis]).astype('int32')
            ir3 = pdist(indexes[2][:,np.newaxis]).astype('int32')
            cmask = (ir1<ctable.shape[0]) & (ir2<ctable.shape[1]) & (ir3<ctable.shape[2])
            ir1[~cmask] = -1
            ir2[~cmask] = -1
            ir3[~cmask] = -1
            C = ctable[ir1,ir2,ir3]

            K = np.zeros((indexes[0].shape[0],indexes[0].shape[0]),dtype='float16')

            K[np.tri(indexes[0].shape[0],k=-1,dtype='bool').T] = C
            K = K + K.T
            #Kc = K.copy()
            blocations = blocks_tree.query((random_walk[0][i],random_walk[1][i],random_walk[2][i]),bnodes)
            br1 = pdist(blpoint[blocations[1],0][:,np.newaxis]).astype('int32')+middle
            br2 = pdist(blpoint[blocations[1],1][:,np.newaxis]).astype('int32')
            br3 = pdist(blpoint[blocations[1],2][:,np.newaxis]).astype('int32')
            bmask = (br1<ctable.shape[0]) & (br2<ctable.shape[1]) & (br3<ctable.shape[2])
            br1[~bmask] = -1
            br2[~bmask] = -1
            br3[~bmask] = -1
            B = ctable[br1,br2,br3]
            Kb = np.zeros((blocations[0].shape[0],blocations[0].shape[0]),dtype='float16')
            Kb[np.tri(blocations[0].shape[0],k=-1,dtype='bool').T] = B
            Kb = Kb + Kb.T

            cr1 = np.int_(cdist(indexes[0][:,np.newaxis],blpoint[blocations[1],0][:,np.newaxis]))
            cr2 = np.int_(cdist(indexes[1][:,np.newaxis],blpoint[blocations[1],1][:,np.newaxis]))
            cr3 = np.int_(cdist(indexes[2][:,np.newaxis],blpoint[blocations[1],2][:,np.newaxis]))
            jmask = (cr1<ctable.shape[0]) & (cr2<ctable.shape[1]) & (cr3<ctable.shape[2])
            cr1[~jmask] = -1
            cr2[~jmask] = -1
            cr3[~jmask] = -1
            J = ctable[cr1,cr2,cr3].T

            Cb = np.zeros((blocations[0].shape[0],indexes[0].shape[0]),dtype='float16')

            Cb[:,:] = J[:,:]
            Cb[np.where((Cb==1))] = 0 # CORRECAO PARA CENTROIDES NO MESMO SITIO QUE PONTOS.     
            K = np.vstack((K,Cb))
            B = np.vstack((Cb.T,Kb))

            K = np.hstack((K,B,np.ones((indexes[0].shape[0]+blocations[0].shape[0],1))))
            K = np.vstack((K,np.ones(K.shape[1])))

            K[np.arange(indexes[0].shape[0]+blocations[0].shape[0]),np.arange(indexes[0].shape[0]+blocations[0].shape[0])]=1        
            K[-1,-1] = 0
            M = ctable[np.clip(np.abs(indexes[0]-random_walk[0][i]+middle),0,ctable.shape[0]-1),
                       np.clip(np.abs(indexes[1]-random_walk[1][i]),0,ctable.shape[1]-1)
                       ,np.clip(np.abs(indexes[2]-random_walk[2][i]),0,ctable.shape[2]-1)]
            #Mc = M.copy()
            M2 = ctable[np.int_(np.clip(np.abs(blpoint[blocations[1],0]-random_walk[0][i]+middle),0,ctable.shape[0]-1)),
                       np.int_(np.clip(np.abs(blpoint[blocations[1],1]-random_walk[1][i]),0,ctable.shape[1]-1))
                       ,np.int_(np.clip(np.abs(blpoint[blocations[1],2]-random_walk[2][i]),0,ctable.shape[2]-1))]
            M = np.hstack((M,M2,np.ones(1)))[:,np.newaxis]
         
            values = np.hstack((igrid[indexes[0],indexes[1],indexes[2]],bdistribution[blocations[1]]))
            mean = __mok__(K,M,values,unique.shape[0])

            mean = mean/np.sum(mean)
            cmean = np.cumsum(mean)
            done = np.zeros((unique.shape[0],2))
            done[0,1] = cmean[0]
            for k in xrange(1,unique.shape[0]): 
                done[k,0] = done[k-1,1]
                done[k,1] = cmean[k]

            p = (np.random.rand()+0.0001)
            for idx in xrange(unique.shape[0]):
                if p>done[idx,0] and p<=done[idx,1]: break

            igrid[random_walk[0][i],random_walk[1][i],random_walk[2][i]] = idx+1
            mask[random_walk[0][i],random_walk[1][i],random_walk[2][i]] = True
            #############################################################
            #print i,' in ',total
        np.save('simulation'+str(s)+'.npy',igrid)
        igrid = np.zeros(blocks,dtype='uint8')
        igrid[np.int_((ix-first[0])/size[0]),np.int_((iy-first[1])/size[1]),np.int_((iz-first[2])/size[2])] = iv
        mask = np.bool_(igrid)
        
def create_synthetic_intrusion(mesh,delta,thick,step,number,noise,seed):
    size = mesh.shape
    res=np.zeros((size[0],size[1]))
    np.random.seed(seed)
    for i in xrange(number):
        x=np.random.randint(0,size[0]-1)
        y=np.random.randint(0,size[1]-1)
        a=np.random.normal(x,delta[0],100).astype('int')
        b=np.random.normal(y,delta[1],100).astype('int')
        res[a[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))],b[np.where((a>0) & (a<size[0]-1) & (b>0) & (b<size[1]-1))]]=1
    blur = blur_image2(res,noise)
    blur = ((blur/blur.max())*size[2]+3).astype('int32')
    #mesh = create_discrete_synthetic_layers(size,number,thick,step,seed)
    omesh = mesh.copy()
    #unit = mesh.max()+1
    for x in xrange(size[0]):
        for y in xrange(size[1]):
            zin = np.clip(blur[x,y],0,size[2]-1)
            f = size[2]-zin
            omesh[x,y,zin:] = mesh[x,y,:f]
            #omesh[x,y,:zin] = unit
    return omesh
        
def geometric_transform(mesh,gtype,number,step,thick,delta,startin,seed,fraction,reflective):
    blocks = mesh.shape
    if gtype in ['sin_fold','cos_fold']:
        return create_synthetic_structure(mesh,blocks,gtype,startin,fraction/100,step,int(blocks[2]/2),reflective)
    elif gtype=='intrusion':
        return create_synthetic_intrusion(mesh,delta,thick,step,number,step,seed)
        
def fracture_sim(number,dis_size,dis_azi,dis_dip,sims,seed,tries,blocks,size,first):
    x = np.random.randint(0,blocks[0],number)
    y = np.random.randint(0,blocks[1],number)
    z = np.random.randint(0,blocks[2],number)
    size = np.random.triangular(dis_size[0],dis_size[2],dis_size[1],number)
    azi  = np.random.triangular(dis_azi[0],dis_azi[2],dis_azi[1],number)
    dip  = np.random.triangular(dis_dip[0],dis_dip[2],dis_dip[1],number)
    data = np.hstack([x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis],size[:,np.newaxis],azi[:,np.newaxis],dip[:,np.newaxis]])
    return data

def fracture_sim_wpmesh(number,dis_size,dis_azi,dis_dip,sims,seed,tries,blocks,size,first,pmesh):
    x = np.random.randint(0,pmesh.shape[0],number)
    y = np.random.randint(0,pmesh.shape[1],number)
    z = np.random.randint(0,pmesh.shape[2],number)
    p = np.random.rand(number)
    pr = p < pmesh[x,y,z]
    x = x[pr]
    y = y[pr]
    z = z[pr]
    number = x.shape[0]
    size = np.random.triangular(dis_size[0],dis_size[2],dis_size[1],number)
    azi  = np.random.triangular(dis_azi[0],dis_azi[2],dis_azi[1],number)
    dip  = np.random.triangular(dis_dip[0],dis_dip[2],dis_dip[1],number)
    data = np.hstack([x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis],size[:,np.newaxis],azi[:,np.newaxis],dip[:,np.newaxis]])
    return data

#"""
from shapely.geometry import Polygon

def return_plane(tri):
    A = tri[0]
    B = tri[1]
    C = tri[2]
    AB = [B[0]-A[0],B[1]-A[1],B[2]-A[2]]
    AC = [C[0]-A[0],C[1]-A[1],C[2]-A[2]]
    cro = np.cross(AB,AC)
    d=np.sum(cro*C)
    return cro,d
    
def return_hessian_plane(tri):
    A = tri[0]
    B = tri[1]
    C = tri[2]
    AB = [B[0]-A[0],B[1]-A[1],B[2]-A[2]]
    AC = [C[0]-A[0],C[1]-A[1],C[2]-A[2]]
    cro = np.cross(AB,AC)
    hcro = np.array([cro[0]/np.sqrt(cro[0]**2+cro[1]**2+cro[2]**2),cro[1]/np.sqrt(cro[0]**2+cro[1]**2+cro[2]**2),cro[2]/np.sqrt(cro[0]**2+cro[1]**2+cro[2]**2)])
    d=np.sum(cro*C)/np.sqrt(cro[0]**2+cro[1]**2+cro[2]**2)
    return hcro,d
    
def tri_tri_intercept(tri1 = [[0,0,0],[1,0,0],[0,1,0]],tri2 = [[0.5,0.5,-0.5],[0.5,0.5,0.5],[0,0,0]]):
    p1,d1 = return_hessian_plane(tri1)
    p2,d2 = return_hessian_plane(tri2)
    print p1,d1
    print p2,d2
    # d = NV +D
    dv1_0 = np.dot(p1,tri1[0])+d1
    dv1_1 = np.dot(p1,tri1[1])+d1
    dv1_2 = np.dot(p1,tri1[2])+d1
    dv2_0 = np.dot(p2,tri2[0])+d2
    dv2_1 = np.dot(p2,tri2[1])+d2
    dv2_2 = np.dot(p2,tri2[2])+d2
    if dv1_0 != 0 and dv1_1 !=0 and dv1_2 !=0 and dv2_0 != 0 and dv2_1 != 0 and dv2_2 != 0:
        return False
    else:
        D = np.cross(p1,p2)
        pv1_0 = np.dot(D,tri1[0])
        pv1_1 = np.dot(D,tri1[1])
        pv1_2 = np.dot(D,tri1[2])
        pv2_0 = np.dot(D,tri2[0])
        pv2_1 = np.dot(D,tri2[1])
        pv2_2 = np.dot(D,tri2[2])
        print pv1_0,dv1_1
        t1 = pv1_0+(pv1_1-pv1_0)*(dv1_0/(dv1_0-dv1_1))
        t2 = pv2_0+(pv2_1-pv2_0)*(dv2_0/(dv2_0-dv2_1))
        print t1,t2,D
        
def tri_tri_intercept2(tri1 = [[0,0,0],[1,0,0],[0,1,0]],tri2 = [[0.5,0.5,-0.5],[0.5,0.5,0.5],[0,0,0]]):
    p1,d1 = return_hessian_plane(tri1)
    p2,d2 = return_hessian_plane(tri2)
    print p1,np.dot(p1,[-30,-30,-30])

def test_shapely_tri_tri_intercept(tri1 = [[0.3,0,0.1],[1.3,0,0],[0.3,1,0]],tri2 = [[0.5,0.2,-0.5],[0.5,0.5,0.5],[0,0,0]]):
    # XOY
    p1_xoy=Polygon([(tri1[0][0],tri1[0][1]),(tri1[1][0],tri1[1][1]),(tri1[2][0],tri1[2][1])])
    p2_xoy=Polygon([(tri2[0][0],tri2[0][1]),(tri2[1][0],tri2[1][1]),(tri2[2][0],tri2[2][1])])
    #print p1_xoy.is_valid
    # XOZ
    p1_xoz=Polygon([(tri1[0][0],tri1[0][2]),(tri1[1][0],tri1[1][2]),(tri1[2][0],tri1[2][2])])
    p2_xoz=Polygon([(tri2[0][0],tri2[0][2]),(tri2[1][0],tri2[1][2]),(tri2[2][0],tri2[2][2])])
    p1_zox=Polygon([(tri1[0][2],tri1[0][0]),(tri1[1][2],tri1[1][0]),(tri1[2][2],tri1[2][0])])
    p2_zox=Polygon([(tri2[0][2],tri2[0][0]),(tri2[1][2],tri2[1][0]),(tri2[2][2],tri2[2][0])])
    # YOZ
    p1_yoz=Polygon([(tri1[0][1],tri1[0][2]),(tri1[1][1],tri1[1][2]),(tri1[2][1],tri1[2][2])])
    p2_yoz=Polygon([(tri2[0][1],tri2[0][2]),(tri2[1][1],tri2[1][2]),(tri2[2][1],tri2[2][2])])
    #print p1_xoy.intersects(p2_xoy),p1_xoz.intersects(p2_xoz),p1_yoz.intersects(p2_yoz)
    if p1_xoy.intersects(p2_xoy) and p1_xoz.intersects(p2_xoz) and p1_yoz.intersects(p2_yoz):
        print p1_xoy.intersection(p2_xoy)
        print p1_xoz.intersection(p2_xoz)
        print p1_yoz.intersection(p2_yoz)
        
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([tri1[0][0],tri1[1][0],tri1[2][0],tri1[0][0]],[tri1[0][1],tri1[1][1],tri1[2][1],tri1[0][1]],[tri1[0][2],tri1[1][2],tri1[2][2],tri1[0][2]],color='black')
        ax.plot([tri2[0][0],tri2[1][0],tri2[2][0],tri2[0][0]],[tri2[0][1],tri2[1][1],tri2[2][1],tri2[0][1]],[tri2[0][2],tri2[1][2],tri2[2][2],tri2[0][2]],color='gray')
        x, y = p2_xoy.intersection(p1_xoy).exterior.coords.xy
        ax.plot(x,y,np.zeros(len(x)),color='red')
        x, y = p1_xoz.intersection(p2_xoz).exterior.coords.xy
        ax.plot(x,np.zeros(len(x)),y,color='pink')
        x, y = p1_yoz.intersection(p2_yoz).exterior.coords.xy
        ax.plot(np.zeros(len(x)),x,y,color='purple')
        x,y = p2_xoy.intersection(p1_xoy).centroid.xy
        x1,z = p1_xoz.intersection(p2_xoz).centroid.xy
        y1,z1 = p1_yoz.intersection(p2_yoz).centroid.xy
        #z,xo = p2_zox.intersection(p2_zox).centroid.xy
        ax.scatter(np.mean([x,x1]),np.mean([y,y1]),np.mean([z,z1]),s=60,color='blue')
        plt.show()
    #print p1_yoz.bounds
    
def shapely_tri_tri_intercept(tri1 = [[0.3,0,0.1],[1.3,0,0],[0.3,1,0]],tri2 = [[0.5,0.2,-0.5],[0.5,0.5,0.5],[0,0,0]]):
    # XOY
    p1_xoy=Polygon([(tri1[0][0],tri1[0][1]),(tri1[1][0],tri1[1][1]),(tri1[2][0],tri1[2][1])])
    p2_xoy=Polygon([(tri2[0][0],tri2[0][1]),(tri2[1][0],tri2[1][1]),(tri2[2][0],tri2[2][1])])
    #print p1_xoy.is_valid
    # XOZ
    p1_xoz=Polygon([(tri1[0][0],tri1[0][2]),(tri1[1][0],tri1[1][2]),(tri1[2][0],tri1[2][2])])
    p2_xoz=Polygon([(tri2[0][0],tri2[0][2]),(tri2[1][0],tri2[1][2]),(tri2[2][0],tri2[2][2])])
    # YOZ
    p1_yoz=Polygon([(tri1[0][1],tri1[0][2]),(tri1[1][1],tri1[1][2]),(tri1[2][1],tri1[2][2])])
    p2_yoz=Polygon([(tri2[0][1],tri2[0][2]),(tri2[1][1],tri2[1][2]),(tri2[2][1],tri2[2][2])])
    #print p1_xoy.intersects(p2_xoy),p1_xoz.intersects(p2_xoz),p1_yoz.intersects(p2_yoz)
    if p1_xoy.intersects(p2_xoy) and p1_xoz.intersects(p2_xoz) and p1_yoz.intersects(p2_yoz):
        x,y = p2_xoy.intersection(p1_xoy).centroid.xy
        x1,z = p1_xoz.intersection(p2_xoz).centroid.xy
        y1,z1 = p1_yoz.intersection(p2_yoz).centroid.xy
        #z,xo = p2_zox.intersection(p2_zox).centroid.xy
        return np.mean([x,x1]),np.mean([y,y1]),np.mean([z,z1])
        
def shapely_tri_tri_intercept2(tri1 = ([0.3,1.3,0.3],[1.3,0,0],[0.3,0.1,0]),tri2 = ([0.5,0.5,0],[0.2,0.5,0],[-0.5,0.5,0])):
    # XOY
    p1_xoy=Polygon([(tri1[0][0],tri1[1][0]),(tri1[0][1],tri1[1][1]),(tri1[0][2],tri1[1][2])])
    p2_xoy=Polygon([(tri2[0][0],tri2[1][0]),(tri2[0][1],tri2[1][1]),(tri2[0][2],tri2[2][1])])
    #print p1_xoy.is_valid
    # XOZ
    p1_xoz=Polygon([(tri1[0][0],tri1[2][0]),(tri1[0][1],tri1[2][1]),(tri1[0][2],tri1[2][2])])
    p2_xoz=Polygon([(tri2[0][0],tri2[2][0]),(tri2[0][1],tri2[2][1]),(tri2[0][2],tri2[2][2])])
    # YOZ
    p1_yoz=Polygon([(tri1[1][0],tri1[2][0]),(tri1[1][1],tri1[2][1]),(tri1[1][2],tri1[2][2])])
    p2_yoz=Polygon([(tri2[1][0],tri2[2][0]),(tri2[1][1],tri2[2][1]),(tri2[1][2],tri2[2][2])])
    #print p1_xoy.intersects(p2_xoy),p1_xoz.intersects(p2_xoz),p1_yoz.intersects(p2_yoz)
    if p1_xoy.intersects(p2_xoy) and p1_xoz.intersects(p2_xoz) and p1_yoz.intersects(p2_yoz):
        x,y = p2_xoy.intersection(p1_xoy).centroid.xy
        x1,z = p1_xoz.intersection(p2_xoz).centroid.xy
        y1,z1 = p1_yoz.intersection(p2_yoz).centroid.xy
        #z,xo = p2_zox.intersection(p2_zox).centroid.xy
        return True,np.mean([x,x1]),np.mean([y,y1]),np.mean([z,z1])
    else:
        return False,0,0,0
        
def get_fracture_connections_points(xxs,yys,zzs,x,y,z):
    s = xxs.shape[0]/3
    nxs = xxs.reshape((s,3))
    nys = yys.reshape((s,3))
    nzs = zzs.reshape((s,3))
    connect = np.zeros((x.shape[0],x.shape[0]),dtype='bool')
    centroid = np.zeros((x.shape[0],x.shape[0],3),dtype='float32')
    for i in xrange(nxs.shape[0]-1):
        for j in xrange(i+1,nxs.shape[0]):
            res = shapely_tri_tri_intercept2((nxs[i,:],nys[i,:],nzs[i,:]),(nxs[j,:],nys[j,:],nzs[j,:]))
            connect[i,j] = res[0]
            centroid[i,j,:] = np.array([res[1],res[2],res[3]])[:]
    return connect,centroid,np.unique(np.where(connect==True)[0])
    
def run_fracture_connections(streams,ind,i,j,c,connect,centroid):
    ind2 = np.where(connect[ind[1][c],:]==True)
    if ind2[0].shape[0]>0:
        print 1
    
def get_fracture_connection_streams(xxs,yys,zzs,x,y,z):
    connect,centroid = get_fracture_connections_points(xxs,yys,zzs,x,y,z)
    ind = np.where(connect==True)
    streams = {}
    c = 0
    for i in xrange(connect.shape[0]):
        for j in xrange(connect.shape[1]):
            if connect[i,j]:
                streams[c] = []
                streams[c].append(centroid[ind[0][c],ind[1][c],:])
                run_fracture_connections(streams[c],ind,c,i,j,c,connect,centroid)
                c=c+1
            
    # É preciso contar os não zeros e depois criar um set de pontos.
    # Mais tarde é necessário separar em vários sets de pontos com os streamlines
            

#test_shapely_tri_tri_intercept()
#tri_tri_intercept2()
#"""
    
    
