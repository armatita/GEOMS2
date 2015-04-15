# -*- coding: utf-8 -*-
"""
Created on Fri Aug 09 19:39:56 2013

@author: Gaming
"""

from __future__ import division
#import sys
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')
import numpy as np
import os
import shutil
import wx

import cerena_file_utils as cfile
import cerena_grid_utils as cgrid
import cerena_multivariate_utils as cmlt
import pympllibrary as pympl

# OBJECTS
# POINT       | point   | pts
# MESH        | mesh    | grid
# BLOCK       | block   | block
# POLYGON     | polygon | polygon
# STATISTICS  | stats   | non-spatial

class surf():
    def __init__(self,data,null_flag=False,null=-9):
        '''
        surf(...)
            surf(data,null_flag=False,null=-9)
            
            Creates a surf variable to be used in a surf collection.
            
        Parameters
        -----------
        data : array_like (numpy array)
            Already with the desired shape and always with 3 dimensions.
            Since its numpy array it follows the characteristics of the
            input numpy array.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations.
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh class
            class with several variables including input ones.
        
        See Also
        --------
        point,block,polygon,stats       
        '''
        # EXPRESSING NULL VALUES FOR CLASS
        self.null = null
        self.null_flag = null_flag
        
        self.dtype = data.dtype
        self.vtype = 'continuous'

        # LOADING DATA TO OBJECT CONSIDERING (OR NOT) NULL VALUE
        # self.data
        if self.null_flag: self.data = np.ma.masked_array(data,data==null)
        else: self.data = np.ma.masked_array(data,data!=data)

        # BUILDING THE BASIC STATISTICS
        # self.basic_statistcs [mean,variance,stdev,min,max,p25,p50,p75]
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        # SETTING UP SOME DEFINITIONS
        
        self.map_flag = False
        self.map_variable = None        
        
        # COLOR DEFINITIONS
        self.myRGBAcolor=(0,255,0,255) # DEFAULT FOR MESH IS GREEN
        self.myRGBcolor=(0,255,0)      # DEFAULT FOR MESH IS GREEN
        self.mycolormap='jet'          # DEFAULT FOR MESH IS JET
        self.my_mpl_color = (0,1,0,1)  # DEFAULT FOR MESH IS GREEN
        self.myopacity = 1             # DEFAULT IS 1 (no transparency).
        
        self.colorbar_preferences = [False,False,'None','horizontal',8,20,False,'%.3f']
        self.colorbar_flag = False
        self.colorbar_title_flag = False
        self.colorbar_title = 'None'
        self.colorbar_orientation = 'horizontal'
        self.colorbar_labels = 8
        self.colorbar_colors = 20
        self.colorbar_fmt_flag = False
        self.colorbar_fmt = '%.3f'
        
        self.uselabel = False
        self.mylabel = 'Unknown'
        self.labelspot = 'Xo,Yo,Zo'
        self.mylabelsize = 1
        self.mylabelcolor = (1,1,1)
        self.mylabelopacity = 1
        self.mylabelorientation = (0,0,0)
        self.mylabelusescamera = True
        
        # PLOT CALLING INSTRUCTIONS
        # HISTOGRAM MUST HAVE COMPRESSED COMMAND
        # SCATTER CAN GO DIRECTLY
        
        self.ibins = None
        self.idata  = None
        self.ivariogram_flag = False
        self.ivariogram_list = None
        
    def calculate_itype(self):
        self.ibins = np.unique(self.data)
        if self.ibins.shape[0]<255:
            if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
            self.vtype='discrete'
            self.ibins = self.ibins.astype('uint8')
            self.idata = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],self.ibins.shape[0]),dtype='uint8')
            for i in xrange(self.ibins.shape[0]):
                self.idata[np.where(self.data==self.ibins[i]),i] = 1
            return True
        else:
            return False
        
    def change_type(self):
        if self.vtype=='continuous':
            self.ibins = np.unique(self.data)
            if self.ibins.shape[0]<255:
                if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
                self.vtype='discrete'
                self.ibins = self.ibins.astype('uint8')
                self.idata = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],self.ibins.shape[0]),dtype='uint8')
                for i in xrange(self.ibins.shape[0]):
                    self.idata[np.where(self.data==self.ibins[i]),i] = 1
                return True
            else:
                return False
        elif self.vtype=='discrete':
            self.vtype='continuous'
            return True
        
    def change_name(self,key,newkey):
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
        
    def update_colorbar_preferences(self,flag,tflag,title,orientation,labels,colors,fmt_flag,fmt):
        self.colorbar_flag = flag
        self.colorbar_title_flag = tflag
        self.colorbar_title = title
        self.colorbar_orientation = orientation
        self.colorbar_labels = labels
        self.colorbar_colors = colors
        self.colorbar_fmt_flag = fmt_flag
        self.colorbar_fmt = fmt
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def build_colorbar_preferences(self):
        # [ 0: SE EXISTE, 1: SE EXISTE TITULO, 2: TITULO, 3: orientacao, 4: nb_labels, 5: nb_colors, 6: string formatter, 7: format]
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]

    def update_to_masked_array(self,data):
        if self.null_flag: 
            self.data = np.ma.masked_array(data,data==self.null)
        else: 
            self.data = np.ma.masked_array(data,data!=data)
        
    def update_basic_statistics(self):
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]    
    
    def new_RGB(self,color):
        '''
        new_RGB(...)
            new_RGB(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of three integers (0 to 255)
            Tuple with the values of RED, GREEN and BLUE all 
            within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal managemente only.
        
        See also
        --------
        new_RGBA
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],255)
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,1)    
        
    def new_RGBA(self,color):
        '''
        new_RGBA(...)
            new_RGBA(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of four integers (0 to 255)
            Tuple with the values of RED, GREEN, BLUE and ALPHA
            (transparency) all within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        new_RGB
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],color[3])
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        
    def get_mpl_color(self):
        '''
        get_mpl_color()
            get_mpl_color(no arguments)
            
            Gives a tuple with the object color in matplotlib format (values
            from 0 to 1).
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: tuple
            tuple with four values 0 to 1 (reg,green,blue,alpha)
        
        See also
        --------
        new_RGB, new_RGBA
        '''
        return self.my_mpl_color
        
class triangle_mesh():
    def __init__(self,data,null_flag=False,null=-9):
        '''
        mesh(...)
            mesh(data,null_flag=False,null=-9)
            
            Creates a mesh variable to be used in a mesh collection.
            
        Parameters
        -----------
        data : array_like (numpy array)
            Already with the desired shape and always with 3 dimensions.
            Since its numpy array it follows the characteristics of the
            input numpy array.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations.
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh class
            class with several variables including input ones.
        
        See Also
        --------
        point,block,polygon,stats       
        '''
        # EXPRESSING NULL VALUES FOR CLASS
        self.null = null
        self.null_flag = null_flag
        
        self.dtype = data.dtype
        self.vtype = 'continuous'
        self.world_view = False

        # LOADING DATA TO OBJECT CONSIDERING (OR NOT) NULL VALUE
        self.data = None
        if self.null_flag: self.data = np.ma.masked_array(data,data==null)
        else: self.data = np.ma.masked_array(data,data!=data)

        # BUILDING THE BASIC STATISTICS
        # self.basic_statistcs [mean,variance,stdev,min,max,p25,p50,p75]
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        # SETTING UP SOME DEFINITIONS
        # COLOR DEFINITIONS
        self.myRGBAcolor=(0,255,0,255) # DEFAULT FOR MESH IS GREEN
        self.myRGBcolor=(0,255,0)      # DEFAULT FOR MESH IS GREEN
        self.mycolormap='jet'          # DEFAULT FOR MESH IS JET
        self.my_mpl_color = (0,1,0,1)  # DEFAULT FOR MESH IS GREEN
        self.myopacity = 1
        self.color_flag = True
        self.mesh_color = (1,0,0)
        
        self.colorbar_preferences = [False,False,'None','vertical',8,20,False,'%.3f']
        self.colorbar_flag = False
        self.colorbar_title_flag = False
        self.colorbar_title = 'None'
        self.colorbar_orientation = 'vertical'
        self.colorbar_labels = 8
        self.colorbar_colors = 20
        self.colorbar_fmt_flag = False
        self.colorbar_fmt = '%.3f'
        
        self.useoutline = False
        self.uselabel = False
        self.mylabel = 'Unknown'
        self.labelspot = 'Xo,Yo,Zo'
        self.mylabelsize = 1
        self.mylabelcolor = (1,1,1)
        self.mylabelopacity = 1
        self.mylabelorientation = (0,0,0)
        self.mylabelusescamera = True
        
        # PLOT CALLING INSTRUCTIONS
        # HISTOGRAM MUST HAVE COMPRESSED COMMAND
        # SCATTER CAN GO DIRECTLY
        self.variogram_adjusted_flag = False
        self.ivariogram_adjusted_flag = False
        
        self.ibins = None
        self.idata  = None
        self.number_of_imodels = 0
        self.imodel1_ranges = []
        self.imodel2_ranges = []
        self.imodel3_ranges = []
        self.imodel_sills = []
        self.imodel_types = []
        self.imodel_angles = [0,0,0]
        self.imodel_nugget = 0
        self.imodel_full_sills = []
        self.imodel_checks = []
        self.ivariogram_adjusted_flag = False

        self.ivariogram_flag = False
        self.ivariogram_list = None
        
        
    def change_name(self,key,newkey):
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
        
    def update_colorbar_preferences(self,flag,tflag,title,orientation,labels,colors,fmt_flag,fmt):
        self.colorbar_flag = flag
        self.colorbar_title_flag = tflag
        self.colorbar_title = title
        self.colorbar_orientation = orientation
        self.colorbar_labels = labels
        self.colorbar_colors = colors
        self.colorbar_fmt_flag = fmt_flag
        self.colorbar_fmt = fmt
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def build_colorbar_preferences(self):
        # [ 0: SE EXISTE, 1: SE EXISTE TITULO, 2: TITULO, 3: orientacao, 4: nb_labels, 5: nb_colors, 6: string formatter, 7: format]
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def update_to_masked_array(self,data):
        if self.null_flag: 
            self.data = np.ma.masked_array(data,data==self.null)
        else: 
            self.data = np.ma.masked_array(data,data!=data)
        
    def update_basic_statistics(self):
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        
    def new_RGB(self,color):
        '''
        new_RGB(...)
            new_RGB(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of three integers (0 to 255)
            Tuple with the values of RED, GREEN and BLUE all 
            within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal managemente only.
        
        See also
        --------
        new_RGBA
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],255)
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,1)    
        
    def new_RGBA(self,color):
        '''
        new_RGBA(...)
            new_RGBA(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of four integers (0 to 255)
            Tuple with the values of RED, GREEN, BLUE and ALPHA
            (transparency) all within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        new_RGB
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],color[3])
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        
    def get_mpl_color(self):
        '''
        get_mpl_color()
            get_mpl_color(no arguments)
            
            Gives a tuple with the object color in matplotlib format (values
            from 0 to 1).
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: tuple
            tuple with four values 0 to 1 (reg,green,blue,alpha)
        
        See also
        --------
        new_RGB, new_RGBA
        '''
        return self.my_mpl_color

class mesh():
    def __init__(self,data,null_flag=False,null=-9):
        '''
        mesh(...)
            mesh(data,null_flag=False,null=-9)
            
            Creates a mesh variable to be used in a mesh collection.
            
        Parameters
        -----------
        data : array_like (numpy array)
            Already with the desired shape and always with 3 dimensions.
            Since its numpy array it follows the characteristics of the
            input numpy array.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations.
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh class
            class with several variables including input ones.
        
        See Also
        --------
        point,block,polygon,stats       
        '''
        # EXPRESSING NULL VALUES FOR CLASS
        self.null = null
        self.null_flag = null_flag
        
        self.dtype = data.dtype
        self.vtype = 'continuous'
        self.world_view = False

        # LOADING DATA TO OBJECT CONSIDERING (OR NOT) NULL VALUE
        self.data = None
        if self.null_flag: self.data = np.ma.masked_array(data,data==null)
        else: self.data = np.ma.masked_array(data,data!=data)

        # BUILDING THE BASIC STATISTICS
        # self.basic_statistcs [mean,variance,stdev,min,max,p25,p50,p75]
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        # SETTING UP SOME DEFINITIONS
        # COLOR DEFINITIONS
        self.myRGBAcolor=(0,255,0,255) # DEFAULT FOR MESH IS GREEN
        self.myRGBcolor=(0,255,0)      # DEFAULT FOR MESH IS GREEN
        self.mycolormap='jet'          # DEFAULT FOR MESH IS JET
        self.my_mpl_color = (0,1,0,1)  # DEFAULT FOR MESH IS GREEN
        self.myopacity = 0
        
        self.colorbar_preferences = [False,False,'None','vertical',8,20,False,'%.3f']
        self.colorbar_flag = False
        self.colorbar_title_flag = False
        self.colorbar_title = 'None'
        self.colorbar_orientation = 'vertical'
        self.colorbar_labels = 8
        self.colorbar_colors = 20
        self.colorbar_fmt_flag = False
        self.colorbar_fmt = '%.3f'
        
        self.useoutline = False
        self.uselabel = False
        self.mylabel = 'Unknown'
        self.labelspot = 'Xo,Yo,Zo'
        self.mylabelsize = 1
        self.mylabelcolor = (1,1,1)
        self.mylabelopacity = 1
        self.mylabelorientation = (0,0,0)
        self.mylabelusescamera = True
        
        # PLOT CALLING INSTRUCTIONS
        # HISTOGRAM MUST HAVE COMPRESSED COMMAND
        # SCATTER CAN GO DIRECTLY
        self.variogram_adjusted_flag = False
        self.ivariogram_adjusted_flag = False
        
        self.ibins = None
        self.idata  = None
        self.number_of_imodels = 0
        self.imodel1_ranges = []
        self.imodel2_ranges = []
        self.imodel3_ranges = []
        self.imodel_sills = []
        self.imodel_types = []
        self.imodel_angles = [0,0,0]
        self.imodel_nugget = 0
        self.imodel_full_sills = []
        self.imodel_checks = []
        self.ivariogram_adjusted_flag = False

        self.ivariogram_flag = False
        self.ivariogram_list = None
        
    def calculate_itype(self):
        self.ibins = np.unique(self.data)
        if self.ibins.shape[0]<255:
            if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
            self.vtype='discrete'
            self.ibins = self.ibins.astype('uint8')
            self.idata = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],self.ibins.shape[0]),dtype='uint8')
            for i in xrange(self.ibins.shape[0]):
                ind = np.where(self.data==self.ibins[i])
                self.idata[ind[0],ind[1],ind[2],i] = 1
                self.imodel_checks.append(True)
                self.imodel1_ranges.append([1,1,1])
                self.imodel2_ranges.append([2,2,2])
                self.imodel3_ranges.append([3,3,3])
                self.imodel_sills.append([100,0,0])
                self.imodel_types.append(['Exponential','Exponential','Exponential'])
                self.imodel_full_sills.append(self.idata[:,:,:,i].var())
            self.imodel_checks.append(True)
            self.imodel1_ranges.append([1,1,1])
            self.imodel2_ranges.append([2,2,2])
            self.imodel3_ranges.append([3,3,3])
            self.imodel_sills.append([100,0,0])
            self.imodel_types.append(['Exponential','Exponential','Exponential'])
            self.imodel_full_sills.append(sum(self.imodel_full_sills)) #self.idata[:,:,:,i].var())
            return True
        else:
            return False
        
    def change_type(self):
        if self.vtype=='continuous':
            self.ibins = np.unique(self.data)
            if self.ibins.shape[0]<255:
                if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
                self.vtype='discrete'
                #print self.ibins,self.ibins.shape[0]
                self.ibins = self.ibins.astype('uint8')
                self.idata = np.zeros((self.data.shape[0],self.data.shape[1],self.data.shape[2],self.ibins.shape[0]),dtype='uint8')
                for i in xrange(self.ibins.shape[0]):
                    ind = np.where(self.data==self.ibins[i])
                    self.idata[ind[0],ind[1],ind[2],i] = 1
                    self.imodel_checks.append(True)
                    self.imodel1_ranges.append([1,1,1])
                    self.imodel2_ranges.append([2,2,2])
                    self.imodel3_ranges.append([3,3,3])
                    self.imodel_sills.append([100,0,0])
                    self.imodel_types.append(['Exponential','Exponential','Exponential'])
                    self.imodel_full_sills.append(self.idata[:,:,:,i].var())
                self.imodel_checks.append(True)
                self.imodel1_ranges.append([1,1,1])
                self.imodel2_ranges.append([2,2,2])
                self.imodel3_ranges.append([3,3,3])
                self.imodel_sills.append([100,0,0])
                self.imodel_types.append(['Exponential','Exponential','Exponential'])
                self.imodel_full_sills.append(sum(self.imodel_full_sills))
                return True
            else:
                return False
        elif self.vtype=='discrete':
            self.vtype='continuous'
            return True
        
    def change_name(self,key,newkey):
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
        
    def update_colorbar_preferences(self,flag,tflag,title,orientation,labels,colors,fmt_flag,fmt):
        self.colorbar_flag = flag
        self.colorbar_title_flag = tflag
        self.colorbar_title = title
        self.colorbar_orientation = orientation
        self.colorbar_labels = labels
        self.colorbar_colors = colors
        self.colorbar_fmt_flag = fmt_flag
        self.colorbar_fmt = fmt
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def build_colorbar_preferences(self):
        # [ 0: SE EXISTE, 1: SE EXISTE TITULO, 2: TITULO, 3: orientacao, 4: nb_labels, 5: nb_colors, 6: string formatter, 7: format]
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]

    def do_variogram3D(self,data,vector,size):
        if vector[0]!=0 and vector[1]!=0 and vector[2]!=0:
            counter = min(np.arange(0,data.shape[0],vector[0]).shape[0],np.arange(0,data.shape[1],vector[1]).shape[0],np.arange(0,data.shape[2],vector[2]).shape[0])-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            yo = np.arange(vector[1],data.shape[1],vector[1])
            zo = np.arange(vector[2],data.shape[2],vector[2])
            m = min(xo.shape[0],yo.shape[0],zo.shape[0])
            for x in xrange(0,data.shape[0],vector[0]):
                for y in xrange(0,data.shape[1],vector[1]):
                    for z in xrange(0,data.shape[2],vector[2]):
                        m2 = min(np.count_nonzero((x+xo<blocks[0])),np.count_nonzero((y+yo<blocks[1])),np.count_nonzero((z+zo<blocks[2])))
                        m3 = min(m,m2)
                        appex = (data[x,y,z]-data[x+xo[:m3],y+yo[:m3],z+zo[:m3]])**2
                        result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                        number[:appex.shape[0]] = number[:appex.shape[0]]+1
            return result/(number*2),dist
        elif vector[0]!= 0 and vector[1]!=0 and vector[2]==0:
            counter = min(np.arange(0,data.shape[0],vector[0]).shape[0],np.arange(0,data.shape[1],vector[1]).shape[0])-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            yo = np.arange(vector[1],data.shape[1],vector[1])
            m = min(xo.shape[0],yo.shape[0])
            for x in xrange(0,data.shape[0]):#,vector[0]):
                for y in xrange(0,data.shape[1]):#,vector[1]):
                    m2 = min(np.count_nonzero((x+xo<blocks[0])),np.count_nonzero((y+yo<blocks[1])))
                    m3 = min(m,m2)
                    appex = np.sum((data[x,y,:]-data[x+xo[:m3],y+yo[:m3],:])**2,axis=1)
                    result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                    number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[2]
            return result/(number*2),dist
        elif vector[0]!= 0 and vector[1]==0 and vector[2]!=0:
            counter = min(np.arange(0,data.shape[0],vector[0]).shape[0],np.arange(0,data.shape[2],vector[2]).shape[0])-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            zo = np.arange(vector[2],data.shape[2],vector[2])
            m = min(xo.shape[0],zo.shape[0])
            for x in xrange(0,data.shape[0]):#,vector[0]):
                for z in xrange(0,data.shape[2]):#,vector[2]):
                    m2 = min(np.count_nonzero((x+xo<blocks[0])),np.count_nonzero((z+zo<blocks[1])))
                    m3 = min(m,m2)
                    appex = np.sum((data[x,:,z]-data[x+xo[:m3],:,z+zo[:m3]])**2,axis=1)
                    result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                    number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[1]
            return result/(number*2),dist
        elif vector[0]== 0 and vector[1]!=0 and vector[2]!=0:
            counter = min(np.arange(0,data.shape[1],vector[1]).shape[0],np.arange(0,data.shape[2],vector[2]).shape[0])-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            yo = np.arange(vector[1],data.shape[1],vector[1])
            zo = np.arange(vector[2],data.shape[2],vector[2])
            m = min(yo.shape[0],zo.shape[0])
            for y in xrange(0,data.shape[1]):#,vector[1]):
                for z in xrange(0,data.shape[2]):#,vector[2]):
                    m2 = min(np.count_nonzero((z+zo<blocks[0])),np.count_nonzero((y+yo<blocks[1])))
                    m3 = min(m,m2)
                    appex = np.sum((data[:,y,z]-data[:,y+yo[:m3],z+zo[:m3]].T)**2,axis=1)
                    result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                    number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[0]
            return result/(number*2),dist
        elif vector[0]!= 0 and vector[1]==0 and vector[2]==0:
            counter = np.arange(0,data.shape[0],vector[0]).shape[0]-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            m = xo.shape[0]
            for x in xrange(0,data.shape[0]):#,vector[0]):            
                m2 = np.count_nonzero((x+xo<blocks[0]))
                m3 = min(m,m2)
                appex = np.sum(np.sum((data[x,:,:]-data[x+xo[:m3],:,:])**2,axis=2),axis=1)
                result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[2]*blocks[1]
            return result/(number*2),dist
        elif vector[0]== 0 and vector[1]!=0 and vector[2]==0:
            counter = np.arange(0,data.shape[1],vector[1]).shape[0]-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            yo = np.arange(vector[1],data.shape[1],vector[1])
            m = yo.shape[0]
            for y in xrange(0,data.shape[1]):#,vector[0]):            
                m2 = np.count_nonzero((y+yo<blocks[1]))
                m3 = min(m,m2)
                appex = np.sum(np.sum((data[:,y,:]-data[:,y+yo[:m3],:].swapaxes(0,1))**2,axis=2),axis=1)
                result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[2]*blocks[1]
            return result/(number*2),dist
        elif vector[0]== 0 and vector[1]==0 and vector[2]!=0:
            counter = np.arange(0,data.shape[2],vector[2]).shape[0]-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            zo = np.arange(vector[2],data.shape[2],vector[2])
            m = zo.shape[0]
            for z in xrange(0,data.shape[2]):#,vector[0]):            
                m2 = np.count_nonzero((z+zo<blocks[2]))
                m3 = min(m,m2)
                appex = np.sum(np.sum((data[:,:,z+zo[:m3]]-data[:,:,z][:,:,np.newaxis])**2,axis=0),axis=0)
                #appex = np.sum(np.sum((data[:,:,z]-data[:,:,z+zo[:m3]].swapaxes(0,2))**2,axis=2),axis=1)
                #appex = np.sum(np.sum((data[:,:,z][:,:,np.newaxis]-data[:,:,z+zo[:m3]])**2,axis=2),axis=1)
                result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[0]*blocks[1]
            return result/(number*2),dist
        else:
            return False
        
    def do_variogram2D(self,data,vector,size):
        """
        dialog = wx.ProgressDialog ( 'Progress', 'unFlattening with surface.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for x in xrange(self.call_top(selection2).blocks[0]):
                dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
        """
        if vector[0]!= 0 and vector[1]!=0:
            counter = min(np.arange(0,data.shape[0],vector[0]).shape[0],np.arange(0,data.shape[1],vector[1]).shape[0])-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            yo = np.arange(vector[1],data.shape[1],vector[1])
            m = min(xo.shape[0],yo.shape[0])
            dialog = wx.ProgressDialog ( 'Progress', 'Calculating variogram.', maximum = data.shape[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for x in xrange(0,data.shape[0]):#,vector[0]):
                for y in xrange(0,data.shape[1]):#,vector[1]):
                    m2 = min(np.count_nonzero((x+xo<blocks[0])),np.count_nonzero((y+yo<blocks[1])))
                    m3 = min(m,m2)
                    appex = (data[x,y,0]-data[x+xo[:m3],y+yo[:m3],0])**2
                    result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                    number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[2]
                dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(data.shape[0]-1) )
            return result/(number*2),dist
        elif vector[0]!= 0 and vector[1]==0:
            counter = np.arange(0,data.shape[0],vector[0]).shape[0]-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            xo = np.arange(vector[0],data.shape[0],vector[0])
            m = xo.shape[0]
            dialog = wx.ProgressDialog ( 'Progress', 'Calculating variogram.', maximum = data.shape[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for x in xrange(0,data.shape[0]):#,vector[0]):
                m2 = np.count_nonzero((x+xo<blocks[0]))
                m3 = min(m,m2)
                appex = np.sum((data[x,:,0]-data[x+xo[:m3],:,0])**2,axis=1)
                result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[1]
                dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(data.shape[0]-1) )
            return result/(number*2),dist
        elif vector[0]== 0 and vector[1]!=0:
            counter = np.arange(0,data.shape[1],vector[1]).shape[0]-1
            blocks = data.shape
            dist_unit = np.sqrt((vector[0]*size[0])**2+(vector[1]*size[1])**2+(vector[2]*size[2])**2)
            dist = np.linspace(dist_unit,dist_unit*counter,counter)
            result = np.zeros(counter)
            number = np.zeros(counter)
            yo = np.arange(vector[1],data.shape[1],vector[1])
            m = yo.shape[0]
            dialog = wx.ProgressDialog ( 'Progress', 'Calculating variogram.', maximum = data.shape[1]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for y in xrange(0,data.shape[1]):#,vector[0]):
                m2 = np.count_nonzero((y+yo<blocks[1]))
                m3 = min(m,m2)
                appex = np.sum((data[:,y,0]-data[y+yo[:m3],:,0])**2,axis=1)
                result[:appex.shape[0]] = result[:appex.shape[0]]+appex
                number[:appex.shape[0]] = number[:appex.shape[0]]+blocks[1]
                dialog.Update ( y, 'Step...'+'  '+repr(y)+'   of   '+repr(data.shape[1]-1) )
            return result/(number*2),dist
        else:
            return False

    def directional_variogram(self,steps,size):
        if self.data.shape[2]>1:
            return self.do_variogram3D(self.data,steps,size)
        else:
            return self.do_variogram2D(self.data,steps,size)
    
    def directional_multiphasic_variogram(self,steps,size):
        if self.data.shape[2]>1:
            l = [self.do_variogram3D(self.idata[:,:,:,0],steps,size)]
            total = l[0][0]
            for i in xrange(1,len(self.imodel_checks)-1):
                l.append(self.do_variogram3D(self.idata[:,:,:,i],steps,size))
                if self.imodel_checks[i]:
                    total = total+l[i][0]
            l.append((total,l[-1][1]))
            return l
        else:
            l = [self.do_variogram2D(self.idata[:,:,:,0],steps,size)]
            total = l[0][0]
            for i in xrange(1,len(self.imodel_checks)-1):
                l.append(self.do_variogram2D(self.idata[:,:,:,i],steps,size))
                if self.imodel_checks[i]:
                    total = total+l[i][0]
            l.append((total,l[-1][1]))
            return l
        
    def update_to_masked_array(self,data):
        if self.null_flag: 
            self.data = np.ma.masked_array(data,data==self.null)
        else: 
            self.data = np.ma.masked_array(data,data!=data)
        
    def update_basic_statistics(self):
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        
    def new_RGB(self,color):
        '''
        new_RGB(...)
            new_RGB(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of three integers (0 to 255)
            Tuple with the values of RED, GREEN and BLUE all 
            within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal managemente only.
        
        See also
        --------
        new_RGBA
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],255)
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,1)    
        
    def new_RGBA(self,color):
        '''
        new_RGBA(...)
            new_RGBA(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of four integers (0 to 255)
            Tuple with the values of RED, GREEN, BLUE and ALPHA
            (transparency) all within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        new_RGB
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],color[3])
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        
    def get_mpl_color(self):
        '''
        get_mpl_color()
            get_mpl_color(no arguments)
            
            Gives a tuple with the object color in matplotlib format (values
            from 0 to 1).
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: tuple
            tuple with four values 0 to 1 (reg,green,blue,alpha)
        
        See also
        --------
        new_RGB, new_RGBA
        '''
        return self.my_mpl_color
        

class point():
    def __init__(self,data,null_flag=False,null=-9):
        '''
        point(...)
            point(data,null_flag=False,null=-9)
            
            Creates a point variable to be used in a point collection.
            
        Parameters
        ----------
        data : array_like (numpy array)
            One dimensional array with only data (no spatial coordinates).
            Since its numpy array it follows the characteristics of the
            input numpy array.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations.
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : point class
            class with several variables including input ones.
        
        See Also
        --------
        mesh,block,polygon,stats        
        '''
        # EXPRESSING NULL VALUES FOR CLASS
        self.null = null
        self.null_flag = null_flag
        self.dtype = data.dtype
        self.vtype = 'continuous'

        # LOADING DATA TO OBJECT CONSIDERING (OR NOT) NULL VALUE
        # self.data
        if self.null_flag: self.data = np.ma.masked_array(data,data==null)
        else: self.data = np.ma.masked_array(data,data!=data)

        # BUILDING THE BASIC STATISTICS
        # self.basic_statistcs [mean,variance,stdev,min,max,p25,p50,p75]
        if self.null_flag:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                     ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
        else:
            self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                     self.data.min(),self.data.max(),np.percentile(self.data,25)
                                     ,np.percentile(self.data,50),np.percentile(self.data,75)]
                                     
        self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        
        # SETTING UP SOME DEFINITIONS
        # COLOR DEFINITIONS
        self.myRGBAcolor=(0,0,255,255)   # DEFAULT FOR POINT IS BLUE
        self.myRGBcolor=(0,0,255)        # DEFAULT FOR POINT IS BLUE
        self.mycolormap='jet'            # DEFAULT FOR POINT IS JET
        self.my_mpl_color = (0,0,1,1)    # DEFAULT FOR POINT IS BLUE
        
        # GLYPH DEFINITION
        self.glyph = 'sphere'
        self.glyph_size = 1
        
        self.quiver_flag = False
        self.u = None
        self.v = None
        self.w = None
        self.quiver_opacity=1
        
        self.stream_flag = False
        
        self.fracture_flag = False
        self.xxs = None
        self.yys = None
        self.zzs = None
        self.triangles = None
        self.fracture_color = (1,0,0)
        
        self.graph_flag = False
        self.graph_data = None
        self.gdata_flag = False
        self.graph_color = (1,0,0)
        self.graph_line = 1
        
        self.uselabel = False
        self.mylabel = np.zeros(self.data.shape[0],dtype='|S16')
        self.mylabelsize = 1
        self.mylabelcolor = (1,1,1)
        self.mylabelopacity = 1
        self.mylabelorientation = (0,0,0)
        self.mylabelusescamera = True
        
        # PLOT CALLING INSTRUCTIONS
        # HISTOGRAM MUST HAVE COMPRESSED COMMAND
        # SCATTER CAN GO DIRECTLY
        
        self.variogram_flag = False
        self.variogram_list = None
        
        self.variogram_adjusted_flag = False
        self.number_of_models = 0
        self.model_angles = [0,0,0]
        self.model_nugget = 0
        self.full_sill = self.basic_statistics[1]
        self.model_sills = [100,0,0]
        self.model_types = ['Exponential','Exponential','Exponential']
        self.model1_ranges = [1,1,1]
        self.model2_ranges = [2,2,2]
        self.model3_ranges = [3,3,3]
        
        self.colorbar_preferences = [False,False,'None','horizontal',8,20,False,'%.3f']
        self.colorbar_flag = False
        self.colorbar_title_flag = False
        self.colorbar_title = 'None'
        self.colorbar_orientation = 'horizontal'
        self.colorbar_labels = 8
        self.colorbar_colors = 20
        self.colorbar_fmt_flag = False
        self.colorbar_fmt = '%.3f'
        
        self.ibins = None
        self.idata  = None
        self.number_of_imodels = 0
        self.imodel1_ranges = []
        self.imodel2_ranges = []
        self.imodel3_ranges = []
        self.imodel_sills = []
        self.imodel_types = []
        self.imodel_angles = [0,0,0]
        self.imodel_nugget = 0
        self.imodel_full_sills = []
        self.imodel_checks = []
        self.ivariogram_adjusted_flag = False
        self.ivariogram_flag = False
        self.ivariogram_list = None
        
    def include_graph_data(self,gdata):
        if gdata.shape[0]!=self.data.shape[0]: return False
        if gdata.max() > self.data.shape[0]-1: return False
        connections = []
        for i in xrange(self.data.shape[0]):
            m = (gdata[i,:]>=0)
            if np.count_nonzero(m)>0:
                for j in xrange(gdata[i,:][m].shape[0]):
                    connections.append(np.array([i,gdata[i,:][m][j]]))
        connections = np.vstack(connections)
        self.graph_data = connections
        self.gdata_flag = True
        return True        
        
    def calculate_fracture_variables(self,x,y,z,azimuth,dip,sizeh): 
        xxs = np.zeros((x.shape[0],3),dtype='float16',order='F')
        yys = np.zeros((x.shape[0],3),dtype='float16',order='F')
        zzs = np.zeros((x.shape[0],3),dtype='float16',order='F')
        xxs[:,0]  = sizeh
        xxs[:,1] = -sizeh
        xxs[:,2] = -sizeh
        yys[:,1]  = sizeh
        yys[:,2]  = -sizeh
        triangles = np.zeros((x.shape[0],3),dtype='uint16')
        cos = np.cos
        sin = np.sin
        for i in xrange(x.shape[0]):
            R=0
            A=azimuth[i]*np.pi/180
            D=dip[i]*np.pi/180
            r = np.array([[cos(D)*cos(A),cos(A)*sin(R)*sin(D)-cos(R)*sin(A),cos(R)*cos(A)*sin(D)+sin(R)*sin(A)],
                           [cos(D)*sin(A),cos(R)*cos(A)+sin(R)*sin(D)*sin(A),-cos(A)*sin(R)+cos(R)*sin(D)*sin(A)]
                           ,[-sin(D),cos(D)*sin(R),cos(R)*cos(D)]])
            res = r.dot(np.array([xxs[i,0],yys[i,0],zzs[i,0]]))
            xxs[i,0]=res[0]+x[i]
            yys[i,0]=res[1]+y[i]
            zzs[i,0]=res[2]+z[i]
            res = r.dot(np.array([xxs[i,1],yys[i,1],zzs[i,1]]))
            xxs[i,1]=res[0]+x[i]
            yys[i,1]=res[1]+y[i]
            zzs[i,1]=res[2]+z[i]
            res = r.dot(np.array([xxs[i,2],yys[i,2],zzs[i,2]]))
            xxs[i,2]=res[0]+x[i]
            yys[i,2]=res[1]+y[i]
            zzs[i,2]=res[2]+z[i]
        c=0
        for i in xrange(0,x.shape[0]*3,3):
            triangles[c,:] = np.array([i,i+1,i+2])
            c=c+1
        xxs = xxs.flatten()
        yys = yys.flatten()
        zzs = zzs.flatten()
        self.xxs = xxs
        self.yys = yys
        self.zzs = zzs
        self.triangles = triangles
        
    def calculate_quiver_variables(self,u,v,w):
        self.u = u
        self.v = v
        self.w = w
        
    def change_my_label(self,labels):
        self.mylabel[:] = labels[:].astype('|S16')
        
    def calculate_itype(self):
        self.ibins = np.unique(self.data)
        if self.ibins.shape[0]<255:
            if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
            self.vtype='discrete'
            self.ibins = self.ibins.astype('uint8')
            self.idata = np.zeros((self.data.shape[0],self.ibins.shape[0]+1),dtype='uint8')
            for i in xrange(self.ibins.shape[0]):
                self.idata[np.where(self.data==self.ibins[i]),i] = 1
                self.imodel_checks.append(True)
                self.imodel1_ranges.append([1,1,1])
                self.imodel2_ranges.append([2,2,2])
                self.imodel3_ranges.append([3,3,3])
                self.imodel_sills.append([100,0,0])
                self.imodel_types.append(['Exponential','Exponential','Exponential'])
                self.imodel_full_sills.append(self.idata[:,i].var())
            for i in xrange(self.idata.shape[0]):
                self.idata[i,-1] = self.idata[i,:-1].sum()
            self.imodel_full_sills.append(self.idata[i,-1].var())
            self.imodel_checks.append(True)
            self.imodel1_ranges.append([1,1,1])
            self.imodel2_ranges.append([2,2,2])
            self.imodel3_ranges.append([3,3,3])
            self.imodel_sills.append([100,0,0])
            self.imodel_types.append(['Exponential','Exponential','Exponential'])
            return True
        else:
            return False
        
    def change_type(self):
        if self.vtype=='continuous':
            self.ibins = np.unique(self.data)
            if self.ibins.shape[0]<255:
                if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
                self.vtype = 'discrete'
                self.ibins = self.ibins.astype('uint8')
                self.idata = np.zeros((self.data.shape[0],self.ibins.shape[0]+1),dtype='uint8')
                for i in xrange(self.ibins.shape[0]):
                    self.idata[np.where(self.data==self.ibins[i]),i] = 1
                    self.imodel_checks.append(True)
                    self.imodel1_ranges.append([1,1,1])
                    self.imodel2_ranges.append([2,2,2])
                    self.imodel3_ranges.append([3,3,3])
                    self.imodel_sills.append([100,0,0])
                    self.imodel_types.append(['Exponential','Exponential','Exponential'])
                    self.imodel_full_sills.append(self.idata[:,i].var())
                #print self.idata
                for i in xrange(self.idata.shape[0]):
                    self.idata[i,-1] = self.idata[i,:-1].sum()
                self.imodel_full_sills.append(sum(self.imodel_full_sills))
                self.imodel_checks.append(True)
                self.imodel1_ranges.append([1,1,1])
                self.imodel2_ranges.append([2,2,2])
                self.imodel3_ranges.append([3,3,3])
                self.imodel_sills.append([100,0,0])
                self.imodel_types.append(['Exponential','Exponential','Exponential'])
                return True
            else:
                return False
        elif self.vtype=='discrete':
            self.vtype='continuous'
            return True
        
    def change_name(self,key,newkey):
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
        
    def update_colorbar_preferences(self,flag,tflag,title,orientation,labels,colors,fmt_flag,fmt):
        self.colorbar_flag = flag
        self.colorbar_title_flag = tflag
        self.colorbar_title = title
        self.colorbar_orientation = orientation
        self.colorbar_labels = labels
        self.colorbar_colors = colors
        self.colorbar_fmt_flag = fmt_flag
        self.colorbar_fmt = fmt
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def build_colorbar_preferences(self):
        # [ 0: SE EXISTE, 1: SE EXISTE TITULO, 2: TITULO, 3: orientacao, 4: nb_labels, 5: nb_colors, 6: string formatter, 7: format]
        self.colorbar_preferences = [self.colorbar_flag,self.colorbar_title_flag,self.colorbar_title,self.colorbar_orientation,self.colorbar_labels,self.colorbar_colors,self.colorbar_fmt_flag,self.colorbar_fmt]
        
    def directional_variogram(self,azimuth,dip,tolerance,bins,maximum=False,dz=False):
        '''
        directional_variogram(...)
            directional_variogram(azimuth,dip,tolerance,bins,maximum)
            
            Calculates a directional variogram from a variogram list
            (previously calculated).
            
        Parameters
        ----------
        azimuth : int or float.
            Angle in degrees in the horizontal plane.
            
        dip : float
            Angle in degrees in the vertical plane.
            
        tolerance : float
            Angle in degrees for the tolerance (both azimuth and dip).
            
        bins : int
            Number of lags starting at zero finishing at maximum.
            
        maximum : float
            Maximum number for distance.
            
        dz : bool or float
            Horizontal variogram parameter (maximum distance in Z).
            
        Returns
        -------
        out: tuple
            Tuple with X and Y values for a variogram plot.
        
        See also
        --------
        None
        '''
        if self.variogram_flag:
            if type(maximum)==bool: maximum = self.variogram_list[:,0].max()
            if dip==0:
                if type(dz)==bool:
                    ind0 = np.where((self.variogram_list[:,1]<=azimuth+tolerance) & (self.variogram_list[:,1] >= azimuth - tolerance) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance))
                    if azimuth+tolerance>90:
                        dif = -90 + (azimuth + tolerance - 90)
                        ind0b = np.where((self.variogram_list[:,1]<=dif) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                    elif azimuth-tolerance<-90:
                        dif = 90 - np.abs((azimuth - tolerance + 90))
                        ind0b = np.where((self.variogram_list[:,1]>=dif) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                else:
                    ind0 = np.where((self.variogram_list[:,1]<=azimuth+tolerance) & (self.variogram_list[:,1] >= azimuth - tolerance) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance) & (self.variogram_list[:,4]<=dz))
                    if azimuth+tolerance>90:
                        dif = -90 + (azimuth + tolerance - 90)
                        ind0b = np.where((self.variogram_list[:,1]<=dif) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance) & (self.variogram_list[:,4]<=dz))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                    elif azimuth-tolerance<-90:
                        dif = 90 - np.abs((azimuth - tolerance + 90))
                        ind0b = np.where((self.variogram_list[:,1]>=dif) & (self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance) & (self.variogram_list[:,4]<=dz))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
            else:
                ind0 = np.where((self.variogram_list[:,2]<=dip+tolerance) & (self.variogram_list[:,2] >= dip - tolerance))
            countsPerBin = np.histogram(self.variogram_list[ind0,0],bins=bins,range=[0,maximum])
            sumsPerBin = np.histogram(self.variogram_list[ind0,0],bins=bins,range=[0,maximum], weights=self.variogram_list[ind0,3])
            ind = np.where(countsPerBin[0]!=0)
            average = sumsPerBin[0][ind] / countsPerBin[0][ind]
            """
            print '###########################'
            print azimuth,dip
            print 'MYIND:', np.where((self.variogram_list[:,1]<-75))
            print 'AVERAGE:',average
            print 'IND:',ind
            print 'SUMSPERBINS:',sumsPerBin
            print 'DISTANCES:',self.variogram_list[ind0,0]
            print 'IND0:',ind0
            print 'INDH:',np.where((self.variogram_list[:,1]<=azimuth+tolerance) & (self.variogram_list[:,1] >= azimuth - tolerance))
            print '###########################'
            """
            if len(average)>0:
                return (average/2,sumsPerBin[1][ind]+(sumsPerBin[1][1]-sumsPerBin[1][0])/2)
            else:
                return (np.array([-10,-10,-10]),np.array([30,70,maximum]))
            
    def directional_multiphasic_variogram(self,azimuth,dip,tolerance,bins,maximum=False,dz=False):
        total,dist = self.directional_ivariogram(azimuth,dip,tolerance,bins,0,maximum,dz)
        c=1
        for i in xrange(1,len(self.imodel_checks)-1):
            if self.imodel_checks[i]:
                atotal,appex = self.directional_ivariogram(azimuth,dip,tolerance,bins,i,maximum,dz)
                dist = dist +appex
                total = total+atotal
                c=c+1
        return (total,dist/c)
                    
    def directional_ivariogram(self,azimuth,dip,tolerance,bins,itype,maximum=False,dz=False):
        '''
        directional_variogram(...)
            directional_variogram(azimuth,dip,tolerance,bins,maximum)
            
            Calculates a directional variogram from a variogram list
            (previously calculated).
            
        Parameters
        ----------
        azimuth : int or float.
            Angle in degrees in the horizontal plane.
            
        dip : float
            Angle in degrees in the vertical plane.
            
        tolerance : float
            Angle in degrees for the tolerance (both azimuth and dip).
            
        bins : int
            Number of lags starting at zero finishing at maximum.
            
        maximum : float
            Maximum number for distance.
            
        dz : bool or float
            Horizontal variogram parameter (maximum distance in Z).
            
        Returns
        -------
        out: tuple
            Tuple with X and Y values for a variogram plot.
        
        See also
        --------
        None
        '''
        #for i in xrange(self.ivariogram_list.shape[2]):
        #    print i
        #    print self.ivariogram_list[:,:,i]
        if self.ivariogram_flag:
            if itype == -1 or itype == self.idata.shape[1]-1:
                return self.directional_multiphasic_variogram(azimuth,dip,tolerance,bins,maximum,dz)
            if type(maximum)==bool: maximum = self.ivariogram_list[:,0,itype].max()
            if dip==0:
                if type(dz)==bool:
                    ind0 = np.where((self.ivariogram_list[:,1,itype]<=azimuth+tolerance) & (self.ivariogram_list[:,1,itype] >= azimuth - tolerance) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance))
                    if azimuth+tolerance>90:
                        dif = -90 + (azimuth + tolerance - 90)
                        ind0b = np.where((self.ivariogram_list[:,1,itype]<=dif) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                    elif azimuth-tolerance<-90:
                        dif = 90 - np.abs((azimuth - tolerance + 90))
                        ind0b = np.where((self.ivariogram_list[:,1,itype]>=dif) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                else:
                    ind0 = np.where((self.ivariogram_list[:,1,itype]<=azimuth+tolerance) & (self.ivariogram_list[:,1,itype] >= azimuth - tolerance) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance) & (self.ivariogram_list[:,4,itype]<=dz))
                    if azimuth+tolerance>90:
                        dif = -90 + (azimuth + tolerance - 90)
                        ind0b = np.where((self.ivariogram_list[:,1,itype]<=dif) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance) & (self.ivariogram_list[:,4,itype]<=dz))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
                    elif azimuth-tolerance<-90:
                        dif = 90 - np.abs((azimuth - tolerance + 90))
                        ind0b = np.where((self.ivariogram_list[:,1,itype]>=dif) & (self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance) & (self.ivariogram_list[:,4,itype]<=dz))
                        ind0 = (np.hstack((ind0[0],ind0b[0])),)
            else:
                ind0 = np.where((self.ivariogram_list[:,2,itype]<=dip+tolerance) & (self.ivariogram_list[:,2,itype] >= dip - tolerance))
            countsPerBin = np.histogram(self.ivariogram_list[ind0,0,itype],bins=bins,range=[0,maximum])
            sumsPerBin = np.histogram(self.ivariogram_list[ind0,0,itype],bins=bins,range=[0,maximum], weights=self.ivariogram_list[ind0,3,itype])
            ind = np.where(countsPerBin[0]!=0)
            #print ind
            average = sumsPerBin[0][ind] / countsPerBin[0][ind]
            """
            print '###########################'
            print azimuth,dip
            print 'MYIND:', np.where((self.variogram_list[:,1]<-75))
            print 'AVERAGE:',average
            print 'IND:',ind
            print 'SUMSPERBINS:',sumsPerBin
            print 'DISTANCES:',self.variogram_list[ind0,0]
            print 'IND0:',ind0
            print 'INDH:',np.where((self.variogram_list[:,1]<=azimuth+tolerance) & (self.variogram_list[:,1] >= azimuth - tolerance))
            print '###########################'
            """
            return (average/2,sumsPerBin[1][ind]+(sumsPerBin[1][1]-sumsPerBin[1][0])/2)
        
    def new_RGB(self,color):
        '''
        new_RGB(...)
            new_RGB(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of three integers (0 to 255)
            Tuple with the values of RED, GREEN and BLUE all 
            within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal managemente only.
        
        See also
        --------
        new_RGBA
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],255)
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,1)    
        
    def new_RGBA(self,color):
        '''
        new_RGBA(...)
            new_RGBA(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of four integers (0 to 255)
            Tuple with the values of RED, GREEN, BLUE and ALPHA
            (transparency) all within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        new_RGB
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],color[3])
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        
    def get_mpl_color(self):
        '''
        get_mpl_color()
            get_mpl_color(no arguments)
            
            Gives a tuple with the object color in matplotlib format (values
            from 0 to 1).
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: tuple
            tuple with four values 0 to 1 (reg,green,blue,alpha)
        
        See also
        --------
        new_RGB, new_RGBA
        '''
        return self.my_mpl_color
        
class data():
    def __init__(self,data,dtype,vtype,null_flag=False,null=-9):
        '''
        data(...)
            data(data,null_flag=False,null=-9)
            
            Creates a data variable to be used in a data collection.
            
        Parameters
        ----------
        data : array_like (numpy array)
            One dimensional array with only data (no spatial coordinates).
            Since its numpy array it follows the characteristics of the
            input numpy array.
        dtype : string
            dtype of the data numpy array.
        vtype : string
            variable type -> string, continuous, discrete
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations.
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : point class
            class with several variables including input ones.
        
        See Also
        --------
        mesh,block,polygon,stats        
        '''
        # EXPRESSING NULL VALUES FOR CLASS
        self.null = null
        self.null_flag = null_flag
        self.dtype = dtype
        self.vtype = vtype

        # LOADING DATA TO OBJECT CONSIDERING (OR NOT) NULL VALUE
        # self.data
        if self.null_flag: self.data = np.ma.masked_array(data,data==null)
        else: self.data = np.ma.masked_array(data,data!=data)

        # BUILDING THE BASIC STATISTICS
        # self.basic_statistcs [mean,variance,stdev,min,max,p25,p50,p75]
        if self.vtype != 'string':
            if self.null_flag:
                self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                         self.data.min(),self.data.max(),np.percentile(self.data.compressed(),25)
                                         ,np.percentile(self.data.compressed(),50),np.percentile(self.data.compressed(),75)]
            else:
                self.basic_statistics = [self.data.mean(),self.data.var(),self.data.std(),
                                         self.data.min(),self.data.max(),np.percentile(self.data,25)
                                         ,np.percentile(self.data,50),np.percentile(self.data,75)]
            self.variable_range = [self.basic_statistics[3],self.basic_statistics[4]]
        else:
            self.basic_statistics = [0,0,0,0,0,0,0,0]
            self.variable_range = [0,0]
        
        # SETTING UP SOME DEFINITIONS
        # COLOR DEFINITIONS
        self.myRGBAcolor=(0,0,255,255)   # DEFAULT FOR POINT IS BLUE
        self.myRGBcolor=(0,0,255)        # DEFAULT FOR POINT IS BLUE
        self.mycolormap='jet'            # DEFAULT FOR POINT IS JET
        self.my_mpl_color = (0,0,1,1)    # DEFAULT FOR POINT IS BLUE
        
        self.ibins = None
        self.idata  = None
        self.ivariogram_flag = False
        self.ivariogram_list = None
        
    def calculate_itype(self):
        self.ibins = np.unique(self.data)
        if self.ibins.shape[0]<255:
            if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
            self.vtype='discrete'
            self.ibins = self.ibins.astype('uint8')
            self.idata = np.zeros((self.data.shape[0],self.ibins.shape[0]),dtype='uint8')
            for i in xrange(self.ibins.shape[0]):
                self.idata[np.where(self.data==self.ibins[i]),i] = 1
            return True
        else: return False
        
    def change_type(self):
        if self.vtype=='continuous':
            self.ibins = np.unique(self.data)
            if self.ibins.shape[0]<255:
                if self.ibins.min()<0: self.ibins[np.where(self.ibins<0)]=0
                self.vtype='discrete'
                self.ibins = self.ibins.astype('uint8')
                self.idata = np.zeros((self.data.shape[0],self.ibins.shape[0]),dtype='uint8')
                for i in xrange(self.ibins.shape[0]):
                    self.idata[np.where(self.data==self.ibins[i]),i] = 1
                return True
            else:
                return False
        elif self.vtype=='discrete':
            self.vtype='continuous'
            return True
        
    def change_name(self,key,newkey):
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
        
    def new_RGB(self,color):
        '''
        new_RGB(...)
            new_RGB(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of three integers (0 to 255)
            Tuple with the values of RED, GREEN and BLUE all 
            within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal managemente only.
        
        See also
        --------
        new_RGBA
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],255)
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,1)    
        
    def new_RGBA(self,color):
        '''
        new_RGBA(...)
            new_RGBA(color)
            
            Changes object default color to the input one.
            
        Parameters
        ----------
        color : tuple of four integers (0 to 255)
            Tuple with the values of RED, GREEN, BLUE and ALPHA
            (transparency) all within range 0:255.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        new_RGB
        '''
        self.myRGBAcolor  = (color[0],color[1],color[2],color[3])
        self.myRGBcolor   = (color[0],color[1],color[2])
        self.my_mpl_color = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        
    def get_mpl_color(self):
        '''
        get_mpl_color()
            get_mpl_color(no arguments)
            
            Gives a tuple with the object color in matplotlib format (values
            from 0 to 1).
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: tuple
            tuple with four values 0 to 1 (reg,green,blue,alpha)
        
        See also
        --------
        new_RGB, new_RGBA
        '''
        return self.my_mpl_color

class surf_collection():
    def __init__(self,blocks,size,first,null_flag=False,null=-999):
        '''
        mesh_collection(...)
            mesh_collection(blocks,size,first,null_flag=False,null=-9)
            
            Creates a mesh collection which contains mesh objects (see mesh).
            
        Parameters
        ----------
        blocks: tuple of three integers
            Size of the array in each of the three directions (x,y,z).
        size: tuple of three integers or floats.
            Size of each cell in the array for each of the three
            directions (x,y,z).
        first: tuple of three integers of floats.
            First coordinate for the mesh in the three directions (x,y,z).
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations. The masking operation only
            happens upon mesh variable creation (see mesh object -> mesh).
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh collection class
            class to manage mesh variables.
        
        See Also
        --------
        point_collection       
        '''
        # MESH COLLECTION IS A MESH MANAGER AND SAVES ALL KINDS OF
        # MESH RELATED INFORMATION (SIZE;FIRST COORDINATE;BLOCK,etc.)
        self.blocks = blocks
        self.size   = size
        self.first  = first
        self.null   = null          # THIS INFO IS REPEATED FOR EACH VARIABLE
        self.null_flag = null_flag  # THIS INFO IS REPEATED FOR EACH VARIABLE
        
        self.xcoords, self.ycoords = self.__get_me_coordinate_arrays__()
        
        # THIS IS THE DICTIONARY THAT SAVES ALL VARIABLES BY NAME
        self.variable = {}
        
        # WARNING: THERES STILL WORK TO BE DONE ABOUT VARIOGRAPHY 
        
        # WARNING: IT MUST BE CREATED THE OBJECT THAT GIVES THE SPATIAL
        # COORDINATES FOR THE SPATIAL PLOTS.
        
    def __get_me_coordinate_arrays__(self):
        xcoords = np.zeros((self.blocks[0],self.blocks[1]))
        ycoords = np.zeros((self.blocks[0],self.blocks[1]))
        #zcoords = np.zeros(self.blocks)
        for i in xrange(self.blocks[0]):
            ycoords[i,:] = np.linspace(self.first[1],self.first[1]+self.blocks[1]*self.size[1],self.blocks[1]).astype('float32')+self.size[1]/2
        for j in xrange(self.blocks[1]):
            xcoords[:,j] = np.linspace(self.first[0],self.first[0]+self.blocks[0]*self.size[0],self.blocks[0]).astype('float32')+self.size[0]/2
        return xcoords,ycoords
        
        
    def translation(self,new_size,new_first):
        '''
        translation(...)
            translation(new_size,new_first)
            
            Changes the general dimensions and location of the surface.
            
        Parameters
        ----------
        new_size : tuple
            Tuple with the new size for the surface.
        new_first : tuple
            Tuple with the new first coordinate of the surface.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        self.size = (new_size[0],new_size[1],new_size[2])
        self.first = (new_first[0],new_first[1],new_first[2])
        self.xcoords, self.ycoords = self.__get_me_coordinate_arrays__()
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(no arguments)
            
            From the string key creates a newkey string and creates
            a new variable under the name key and deletes the previous
            one.
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to change
            the name.
        newkey : string
            string with the name of the variable previously with the
            name key.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        get_variable_names, get_newname_from_name
        '''
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
                    
    def get_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_numeric_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            From name creates a new name by adding a number.
            
        Parameters
        ----------
        name : string
            string of a name from an object.
            
        Returns
        -------
        out: string
            new string created from the input one
        
        See also
        --------
        change_name, get_variable_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.variable.keys():
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def add_variable(self,name,data):
        '''
        add_variable(...)
            add_variable(name,data)
            
            Creates a new variable to the collection as long as data is
            provided.
            
        Parameters
        ----------
        name : string
            string of a name from an object to be created.
        data : array
            array with the data from which to build the variable. In this
            case the data must have the desired shape.
            
        Returns
        -------
        out: None
            no return. New surf variable created to the collection.
        
        See also
        --------
        None
        '''
        if name in self.variable.keys(): name = self.get_newname_from_name(name)
        self.variable[name] = surf(data,self.null_flag,self.null)
        
    def __popupmessage__(self,info,title='Information',itype='error'):
        '''
        __popupmessage__(...)
            popupmessage(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
        
class mesh_collection():
    def __init__(self,blocks,size,first,null_flag=False,null=-999):
        '''
        mesh_collection(...)
            mesh_collection(blocks,size,first,null_flag=False,null=-9)
            
            Creates a mesh collection which contains mesh objects (see mesh).
            
        Parameters
        ----------
        blocks: tuple of three integers
            Size of the array in each of the three directions (x,y,z).
        size: tuple of three integers or floats.
            Size of each cell in the array for each of the three
            directions (x,y,z).
        first: tuple of three integers of floats.
            First coordinate for the mesh in the three directions (x,y,z).
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations. The masking operation only
            happens upon mesh variable creation (see mesh object -> mesh).
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh collection class
            class to manage mesh variables.
        
        See Also
        --------
        point_collection       
        '''
        # MESH COLLECTION IS A MESH MANAGER AND SAVES ALL KINDS OF
        # MESH RELATED INFORMATION (SIZE;FIRST COORDINATE;BLOCK,etc.)
        self.blocks = blocks
        self.size   = size
        self.first  = first
        self.null   = null          # THIS INFO IS REPEATED FOR EACH VARIABLE
        self.null_flag = null_flag  # THIS INFO IS REPEATED FOR EACH VARIABLE
        
        self.xcoords, self.ycoords, self.zcoords = self.__get_me_coordinate_arrays__()
        
        self.spherex = None
        self.spherey = None
        self.spherez = None
        self.sphere_diameter = 1
        self.sphere_flag = False
        # THIS IS THE DICTIONARY THAT SAVES ALL VARIABLES BY NAME
        self.variable = {}
        
        # WARNING: THERES STILL WORK TO BE DONE ABOUT VARIOGRAPHY 
        
        # WARNING: IT MUST BE CREATED THE OBJECT THAT GIVES THE SPATIAL
        # COORDINATES FOR THE SPATIAL PLOTS.
        
    def calculate_sphere_coordinates(self):
        theta = np.linspace(0, np.pi, self.blocks[1], endpoint=True)
        phi = np.linspace(0, 2 * np.pi, self.blocks[0], endpoint=True)
        theta, phi = np.meshgrid(theta, phi)
        self.spherex = self.sphere_diameter * np.sin(theta) * np.cos(phi) + self.first[0]
        self.spherey = self.sphere_diameter * np.sin(theta) * np.sin(phi) + self.first[1]
        self.spherez = self.sphere_diameter * np.cos(theta) + self.first[2]
        self.sphere_flag = True
        
    def zstretch(self,zscale,method):
        if method == 'Multiply':
            self.size = (self.size[0],self.size[1],self.size[2]*zscale)
            for i in xrange(self.blocks[0]):
                for j in xrange(self.blocks[1]):
                    self.zcoords[i,j,:] = np.linspace(self.first[2],self.first[2]+self.blocks[2]*self.size[2],self.blocks[2]).astype('float32')+self.size[2]/2
        elif method == 'Fraction':
            self.size = (self.size[0],self.size[1],self.size[2]/zscale)
            for i in xrange(self.blocks[0]):
                for j in xrange(self.blocks[1]):
                    self.zcoords[i,j,:] = np.linspace(self.first[2],self.first[2]+self.blocks[2]*self.size[2],self.blocks[2]).astype('float32')+self.size[2]/2
        
    def translation(self,new_size,new_first):
        '''
        translation(...)
            translation(new_size,new_first)
            
            Changes the general dimensions and location of the mesh.
            
        Parameters
        ----------
        new_size : tuple
            Tuple with the new size for the mesh.
        new_first : tuple
            Tuple with the new first coordinate of the mesh.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        self.size = (new_size[0],new_size[1],new_size[2])
        self.first = (new_first[0],new_first[1],new_first[2])
        self.xcoords, self.ycoords, self.zcoords = self.__get_me_coordinate_arrays__()
        if self.sphere_flag: self.calculate_sphere_coordinates()
            
        
    def __get_me_coordinate_arrays__(self):
        xcoords = np.zeros(self.blocks)
        ycoords = np.zeros(self.blocks)
        zcoords = np.zeros(self.blocks)
        for i in xrange(self.blocks[0]):
            for j in xrange(self.blocks[1]):
                zcoords[i,j,:] = np.linspace(self.first[2],self.first[2]+self.blocks[2]*self.size[2],self.blocks[2]).astype('float32')-self.size[2]/2
        for k in xrange(self.blocks[2]):
            for j in xrange(self.blocks[1]):
                xcoords[:,j,k] = np.linspace(self.first[0],self.first[0]+self.blocks[0]*self.size[0],self.blocks[0]).astype('float32')-self.size[0]/2
        for k in xrange(self.blocks[2]):
            for i in xrange(self.blocks[0]):
                ycoords[i,:,k] = np.linspace(self.first[1],self.first[1]+self.blocks[1]*self.size[1],self.blocks[1]).astype('float32')-self.size[1]/2
        return xcoords,ycoords,zcoords
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(no arguments)
            
            From the string key creates a newkey string and creates
            a new variable under the name key and deletes the previous
            one.
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to change
            the name.
        newkey : string
            string with the name of the variable previously with the
            name key.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        get_variable_names, get_newname_from_name
        '''
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
                    
    def get_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_numeric_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            From name creates a new name by adding a number.
            
        Parameters
        ----------
        name : string
            string of a name from an object.
            
        Returns
        -------
        out: string
            new string created from the input one
        
        See also
        --------
        change_name, get_variable_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.variable.keys():
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def add_variable(self,name,data):
        '''
        add_variable(...)
            add_variable(name,data)
            
            Creates a new variable to the collection as long as data is
            provided.
            
        Parameters
        ----------
        name : string
            string of a name from an object to be created.
        data : array
            array with the data from which to build the variable. In this
            case the data must have the desired shape.
            
        Returns
        -------
        out: None
            no return. New mesh variable created to the collection.
        
        See also
        --------
        None
        '''
        #print name
        if name in self.variable.keys(): name = self.get_newname_from_name(name)
        self.variable[name] = mesh(data,self.null_flag,self.null)
        
    def __popupmessage__(self,info,title='Information',itype='error'):
        '''
        __popupmessage__(...)
            popupmessage(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
            
class triangle_mesh_collection():
    def __init__(self,x,y,z,triangles,null_flag=False,null=-999):
        '''
        triangle_mesh_collection(...)
            triangle_mesh_collection(x,y,z,triangles,null_flag=False,null=-9)
            
            Creates a triangle mesh collection which contains triangle objects.
            
        Parameters
        ----------
        x: float array
            Float array with x coordinates.
        y: float array
            Float array with y coordinates.
        x: float array
            Float array with z coordinates.
        triangles: int array
            Int array with x size on rows and 3 size on columns. Each row has
            the 3 indexes for a triangle in the x,y,z arrays.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations. The masking operation only
            happens upon mesh variable creation (see mesh object -> mesh).
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : mesh collection class
            class to manage mesh variables.
        
        See Also
        --------
        point_collection,mesh_collection,data_collection       
        '''
        # MESH COLLECTION IS A MESH MANAGER AND SAVES ALL KINDS OF
        # MESH RELATED INFORMATION (SIZE;FIRST COORDINATE;BLOCK,etc.)
        self.null   = null          # THIS INFO IS REPEATED FOR EACH VARIABLE
        self.null_flag = null_flag  # THIS INFO IS REPEATED FOR EACH VARIABLE
        
        self.xcoords, self.ycoords, self.zcoords = x,y,z
        self.color_variable = np.zeros(x.shape[0],dtype='float32')
        self.triangles = triangles
        self.mycolor = (1,0,0)
        self.mycolormap = 'jet'
        self.colorbar_preferences = [False,False,'None','vertical',8,20,False,'%.3f']
        
        self.variable = {}
        
        # WARNING: THERES STILL WORK TO BE DONE ABOUT VARIOGRAPHY 
        
        # WARNING: IT MUST BE CREATED THE OBJECT THAT GIVES THE SPATIAL
        # COORDINATES FOR THE SPATIAL PLOTS.
        
    def translation(self,new_size,new_first):
        '''
        translation(...)
            translation(new_size,new_first)
            
            Changes the general dimensions and location of the mesh.
            
        Parameters
        ----------
        new_size : tuple
            Tuple with the new size for the mesh.
        new_first : tuple
            Tuple with the new first coordinate of the mesh.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        self.size = (new_size[0],new_size[1],new_size[2])
        self.first = (new_first[0],new_first[1],new_first[2])
        self.xcoords, self.ycoords, self.zcoords = self.__get_me_coordinate_arrays__()
        if self.sphere_flag: self.calculate_sphere_coordinates()
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(no arguments)
            
            From the string key creates a newkey string and creates
            a new variable under the name key and deletes the previous
            one.
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to change
            the name.
        newkey : string
            string with the name of the variable previously with the
            name key.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        get_variable_names, get_newname_from_name
        '''
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
                    
    def get_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_numeric_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            From name creates a new name by adding a number.
            
        Parameters
        ----------
        name : string
            string of a name from an object.
            
        Returns
        -------
        out: string
            new string created from the input one
        
        See also
        --------
        change_name, get_variable_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.variable.keys():
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def add_variable(self,name,data):
        '''
        add_variable(...)
            add_variable(name,data)
            
            Creates a new variable to the collection as long as data is
            provided.
            
        Parameters
        ----------
        name : string
            string of a name from an object to be created.
        data : array
            array with the data from which to build the variable. In this
            case the data must have the desired shape.
            
        Returns
        -------
        out: None
            no return. New mesh variable created to the collection.
        
        See also
        --------
        None
        '''
        #print name
        if name in self.variable.keys(): name = self.get_newname_from_name(name)
        self.variable[name] = triangle_mesh(data,self.null_flag,self.null)
        
    def __popupmessage__(self,info,title='Information',itype='error'):
        '''
        __popupmessage__(...)
            popupmessage(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
            
class point_collection():
    def __init__(self,x,y,z,null_flag=False,null=-999):
        '''
        point_collection(...)
            point_collection(x,y,z,null_flag=False,null=-9)
            
            Creates a point collection which contains point objects (see point).
            
        Parameters
        ----------
        x: array of integers or floats.
            Array of coordinates for each of the points in X axis.
        y: array of integers or floats.
            Array of coordinates for each of the points in Y axis.
        z: array of integers or floats.
            Array of coordinates for each of the points in Z axis.
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations. The masking operation only
            happens upon mesh variable creation (see mesh object -> mesh).
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : point collection class
            class to manage point variables.
        
        See Also
        --------
        mesh_collection       
        '''
        # POINT COLLECTION IS A POINT MANAGER AND SAVES ALL KINDS OF
        # POINT RELATED INFORMATION (X;Y;Z,etc.)
        self.x = x
        self.y = y
        self.z = z
        self.null = null            # THIS INFO IS REPEATED FOR EACH VARIABLE
        self.null_flag = null_flag  # THIS INFO IS REPEATED FOR EACH VARIABLE
        
        # THIS IS THE DICTIONARY THAT SAVES ALL VARIABLES BY NAME
        self.variable = {}

        # WARNING: THERES STILL WORK TO BE DONE ABOUT VARIOGRAPHY
        
    def zstretch(self,zscale,method):
        if method == 'Multiply':
            dmin = self.z.min()
            dmax = self.z.max()
            zmax = self.z.max()+(self.z.max()-self.z.min())*zscale
            zmin = self.z.min()
            self.z = (self.z-dmin)*(zmax-zmin)/(dmax-dmin)+zmin
        elif method == 'Fraction':
            dmin = self.z.min()
            dmax = self.z.max()
            zmax = self.z.max()/zscale
            zmin = self.z.min()
            self.z = (self.z-dmin)*(zmax-zmin)/(dmax-dmin)+zmin
        
    def translation(self,new_const):
        '''
        translation(...)
            translation(new_first)
            
            Changes the general location of the poin by summing a constant.
            
        Parameters
        ----------
        
        new_const : tuple
            Tuple with the constants to be added to the x,y,z locations.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        self.x = self.x + new_const[0]
        self.y = self.y + new_const[1]
        self.z = self.z + new_const[2]
        
    def calculate_variogram(self,key):
        '''
        calculate_variogram(...)
            calculate_variogram(key)
            
            Calculates a variogram array of values for distance,
            azimuth, dip and variance (each in its own column).
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to calculate
            the variogram.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        if self.null_flag==True:
            ind = np.where(self.variable[key].data.mask!=True)
            x = self.x[ind]
            y = self.y[ind]
            z = self.z[ind]
            v = self.variable[key].data[ind]
        else:
            x = self.x
            y = self.y
            z = self.z
            v = self.variable[key].data
        full_size = np.sum(xrange(x.shape[0]))-x.shape[0]+1
        variogram_list = np.zeros((full_size,5),dtype='float32')
        variogram_list[:,:] = -99
        counter = 0
        l = x.shape[0]
        dlg = wx.ProgressDialog("Progress dialog",
                               "Calculating variogram list for this variable...",
                               maximum = l-2,
                               parent=None,
                               style = wx.PD_APP_MODAL
                                | wx.PD_ELAPSED_TIME
                                #| wx.PD_ESTIMATED_TIME
                                | wx.PD_REMAINING_TIME
                                | wx.PD_AUTO_HIDE
                                )
        for i in xrange(1,l-1):
            distx = (x[i+1:]-x[i])
            disty = (y[i+1:]-y[i])
            distz = (z[i+1:]-z[i])
            azimuth = np.arctan2(np.abs(distx),np.abs(disty))*180/np.pi
            Q2 = np.where((distx<0) & (disty>0))
            Q4 = np.where((distx>0) & (disty<0))
            azimuth[Q2[0]] = azimuth[Q2[0]]*-1
            azimuth[Q4[0]] = azimuth[Q4[0]]*-1
            dip = np.arctan2(distz,np.sqrt(distx**2+disty**2))*180/np.pi #np.arctan2(distz,np.sqrt(distx**2+disty**2))*180/np.pi
            dist = np.sqrt(distx**2+disty**2+distz**2)
            var = (v[i+1:]-v[i])**2
            variogram_list[counter:counter+(l-i-1),0] = dist[:]
            variogram_list[counter:counter+(l-i-1),1] = azimuth[:]
            variogram_list[counter:counter+(l-i-1),2] = dip[:]
            variogram_list[counter:counter+(l-i-1),3] = var[:]
            variogram_list[counter:counter+(l-i-1),4] = np.abs(distz[:])
            counter = counter+(l-i-1)
            dlg.Update(i)
        dlg.Destroy()
        self.variable[key].variogram_list = variogram_list
        self.variable[key].variogram_flag = True
        return True
        
    def calculate_ivariogram(self,key):
        '''
        calculate_ivariogram(...)
            calculate_ivariogram(key)
            
            Calculates a indicator variogram array of values for distance,
            azimuth, dip and variance (each in its own column).
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to calculate
            the variogram.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        None
        '''
        if self.null_flag==True:
            ind = np.where(self.variable[key].data.mask!=True)
            x = self.x[ind]
            y = self.y[ind]
            z = self.z[ind]
            v = self.variable[key].idata[ind]
        else:
            x = self.x
            y = self.y
            z = self.z
            v = self.variable[key].idata
        #for i in xrange(self.variable[key].idata.shape[1]):
        #    print i
        #    print self.variable[key].idata[:,i]
        full_size = np.sum(xrange(x.shape[0]))-x.shape[0]+1
        variogram_list = np.zeros((full_size,5,self.variable[key].idata.shape[1]),dtype='float32')
        variogram_list[:,:,:] = -99
        counter = 0
        l = x.shape[0]
        #"""
        dlg = wx.ProgressDialog("Progress dialog",
                               "Calculating variogram list for this variable...",
                               maximum = l-2,
                               parent=None,
                               style = wx.PD_APP_MODAL
                                | wx.PD_ELAPSED_TIME
                                #| wx.PD_ESTIMATED_TIME
                                | wx.PD_REMAINING_TIME
                                | wx.PD_AUTO_HIDE
                                )
        #"""
        for i in xrange(1,l-1):
            distx = (x[i+1:]-x[i])
            disty = (y[i+1:]-y[i])
            distz = (z[i+1:]-z[i])
            azimuth = np.arctan2(np.abs(distx),np.abs(disty))*180/np.pi
            Q2 = np.where((distx<0) & (disty>0))
            Q4 = np.where((distx>0) & (disty<0))
            azimuth[Q2[0]] = azimuth[Q2[0]]*-1
            azimuth[Q4[0]] = azimuth[Q4[0]]*-1
            dip = np.arctan2(distz,np.sqrt(distx**2+disty**2))*180/np.pi #np.arctan2(distz,np.sqrt(distx**2+disty**2))*180/np.pi
            dist = np.sqrt(distx**2+disty**2+distz**2)
            for j in xrange(self.variable[key].idata.shape[1]):
                var = (v[i+1:,j]-v[i,j])**2
                variogram_list[counter:counter+(l-i-1),0,j] = dist[:]
                variogram_list[counter:counter+(l-i-1),1,j] = azimuth[:]
                variogram_list[counter:counter+(l-i-1),2,j] = dip[:]
                variogram_list[counter:counter+(l-i-1),3,j] = var[:]
                variogram_list[counter:counter+(l-i-1),4,j] = np.abs(distz[:])
            counter = counter+(l-i-1)
            dlg.Update(i)
        dlg.Destroy()
        #for i in xrange(variogram_list.shape[2]):
        #    print i
        #    print variogram_list[:,3,i]
        #for i in xrange(self.variable[key].idata.shape[1]):
        #    print i
        #    print variogram_list[:,3,i].min(),variogram_list[:,3,i].max()
        #print variogram_list
        self.variable[key].ivariogram_list = variogram_list
        self.variable[key].ivariogram_flag = True
        return True
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(key,newkey)
            
            From the string key creates a newkey string and creates
            a new variable under the name key and deletes the previous
            one.
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to change
            the name.
        newkey : string
            string with the name of the variable previously with the
            name key.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        get_variable_names, get_newname_from_name
        '''
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
                    
    def get_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_numeric_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            From name creates a new name by adding a number.
            
        Parameters
        ----------
        name : string
            string of a name from an object.
            
        Returns
        -------
        out: string
            new string created from the input one
        
        See also
        --------
        change_name, get_variable_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.variable.keys():
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def add_variable(self,name,data):
        '''
        add_variable(...)
            add_variable(name,data)
            
            Creates a new variable to the collection as long as data is
            provided.
            
        Parameters
        ----------
        name : string
            string of a name from an object to be created.
        data : array
            array with the data from which to build the variable.
            
        Returns
        -------
        out: None
            no return. New point variable created to the collection.
        
        See also
        --------
        None
        '''
        if name in self.variable.keys(): name = self.get_newname_from_name(name)
        self.variable[name] = point(data,self.null_flag,self.null)
        
    def __popupmessage__(self,info,title='Information',itype='error'):
        '''
        __popupmessage__(...)
            popupmessage(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
            
class data_collection():
    def __init__(self,null_flag=False,null=-999):
        '''
        data_collection(...)
            data_collection(null_flag=False,null=-9)
            
            Creates a point collection which contains point objects (see point).
            
        Parameters
        ----------
        null_flag : bool,optional
            Default is False. If False a variable will be created with
            all mask values False (but it still has mask) which means all
            values in array are valid. If True than all values equal to
            null will mean cells where null exists are mask True and not
            valid for posterior operations. The masking operation only
            happens upon mesh variable creation (see mesh object -> mesh).
        null : integer, float, etc.
            Value by which cells are masked. See null_flag parameter.
            
        Returns
        -------
        out : point collection class
            class to manage point variables.
        
        See Also
        --------
        mesh_collection       
        '''
        # DATA COLLECTION IS A POINT MANAGER AND SAVES ALL KINDS OF
        # DATA RELATED INFORMATION 
        self.null = null            # THIS INFO IS REPEATED FOR EACH VARIABLE
        self.null_flag = null_flag  # THIS INFO IS REPEATED FOR EACH VARIABLE
        
        # THIS IS THE DICTIONARY THAT SAVES ALL VARIABLES BY NAME
        self.variable_dtype = []  # DTYPE IS EITHER string OR ANY OTHER TYPE OF ACCEPTED NUMERIC DATA
        self.variable_type = []   # TYPE IS EITHER string, continuous OR discrete
        self.variable = {}
        self.x = np.zeros(1,dtype='bool') # DOING THIS TO MAKE IT COMPATIBLE WITH POINT
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(key,newkey)
            
            From the string key creates a newkey string and creates
            a new variable under the name key and deletes the previous
            one.
            
        Parameters
        ----------
        key : string
            string with the name of the variable we intend to change
            the name.
        newkey : string
            string with the name of the variable previously with the
            name key.
            
        Returns
        -------
        out: None
            does not have return. Internal management only.
        
        See also
        --------
        get_variable_names, get_newname_from_name
        '''
        if newkey not in self.variable.keys():
            self.variable[newkey] = self.variable.pop(key)
                    
    def get_variable_names(self):
        '''
        get_variable_names()
            get_variable_names(key,newkey)
            
            Gets all names from all variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        return self.variable.keys()
        
    def get_numeric_variable_names(self,vtype='all'):
        '''
        get_numeric_variable_names()
            get_numeric_variable_names(key,newkey)
            
            Gets all names from all or particular numeric variables in collection.
            
        Parameters
        ----------
        vtype : String
            all, continuous or discrete
            
        Returns
        -------
        out: list
            list with the required names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        possible_dtypes = ['int','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float','float16','float32','float64']
        vlist = self.variable.keys()
        if vtype == 'all':
            numeric_variables = []
            for i in xrange(len(vlist)):
                #print vlist[i],self.variable[vlist[i]].dtype
                if self.variable[vlist[i]].dtype in possible_dtypes:
                    numeric_variables.append(vlist[i])
        elif vtype == 'continuous':
            numeric_variables = []
            for i in xrange(len(vlist)):
                if self.variable[vlist[i]].dtype in possible_dtypes:
                    if self.variable[vlist[i]].vtype == 'continuous':
                        numeric_variables.append(vlist[i])
        elif vtype == 'discrete':
            numeric_variables = []
            for i in xrange(len(vlist)):
                if self.variable[vlist[i]].dtype in possible_dtypes:
                    if self.variable[vlist[i]].vtype == 'discrete':
                        numeric_variables.append(vlist[i])
        #print numeric_variables
        return numeric_variables
        
    def get_string_variable_names(self):
        '''
        get_string_variable_names()
            get_string_variable_names(key,newkey)
            
            Gets all names from all or particular string variables in collection.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list
            list with the required names of all variables in collection.
        
        See also
        --------
        change_name, get_newname_from_name
        '''
        vlist = self.variable.keys()
        string_variables = []
        for i in xrange(len(vlist)):
            if self.variable[vlist[i]].dtype == 'string':
                string_variables.append(vlist[i])
        return string_variables
                
        
    def get_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            From name creates a new name by adding a number.
            
        Parameters
        ----------
        name : string
            string of a name from an object.
            
        Returns
        -------
        out: string
            new string created from the input one
        
        See also
        --------
        change_name, get_variable_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.variable.keys():
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def add_variable(self,name,nsdata,dtype,vtype):
        '''
        add_variable(...)
            add_variable(name,data)
            
            Creates a new variable to the collection as long as data is
            provided.
            
        Parameters
        ----------
        name : string
            string of a name from an object to be created.
        data : array
            array with the data from which to build the variable.
            
        Returns
        -------
        out: None
            no return. New point variable created to the collection.
        
        See also
        --------
        None
        '''
        if name in self.variable.keys(): name = self.get_newname_from_name(name)
        self.variable[name] = data(nsdata,dtype,vtype,self.null_flag,self.null)
        self.x = np.zeros(nsdata.shape,dtype='bool')
        self.variable_dtype.append(dtype)
        self.variable_type.append(vtype)
        
    def __popupmessage__(self,info,title='Information',itype='error'):
        '''
        __popupmessage__(...)
            popupmessage(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
                 
class object_manager():
    def __init__(self):
        '''
        object_manager(...)
            object_manager(no arguments)
            
            Creates a manager for all objects and several operations.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out : object manager class
            class to manage cerena objects.
        
        See Also
        --------
        mesh_collection, point_collection     
        '''
        # USER DEFINITONS
        #self.user_definitions = user_definitions()        
        
        # GENERALISTIC
        self.object_name    = {}  # NAMES OF ALL OBJECTS IN object_manager
        self.object_type    = {}  # TYPES OF ALL OBJECTS IN object_types
        self.object_list    = {}  # DATA OF ALL OBJECTS IN object_manager
        self.object_counter = 0   # COUNTER SO THAT ALL OBJECTS GET AN ID.
        self.last_object    = ''
        self.last_variable  = ''
        self.update_objects = []
        
        self.mayavi_colormaps = ['Accent','Blues','BrBG','BuGn','BuPu','Dark2',
                                 'GnBu','Greens','Greys','OrRd','Oranges'
                                 ,'PRGn','Paired','Pastel1','Pastel2','PiYG'
                                 ,'PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu'
                                 ,'RdGy','RdPu','RdYlBu','RdYlGn','Reds','Set1'
                                 ,'Set2','Set3','Spectral','YlGn','YlGnBu'
                                 ,'YlOrBr','YlOrRd','autumn','binary'
                                 ,'black-white','blue-red','bone','cool'
                                 ,'copper','file','flag','gist_earth'
                                 ,'gist_gray','gist_heat','gist_ncar'
                                 ,'gist_rainbow','gist_stern','gist_yarg','gray'
                                 ,'hot','hsv','jet','pink','prism','spectral'
                                 ,'spring','summer','winter']        
                                 
        self.mayavi_glyphs = ['2darrow','2dcircle','2dcross','2ddash',
                              '2ddiamond','2dhooked_arrow','2dsquare'
                              ,'2dthick_arrow','2dthick_cross','2dtriangle'
                              ,'2dvertex','arrow','axes','cone','cube'
                              ,'cylinder','point','sphere']
                              
        self.matplotlib_colormaps = ['Accent','Blues','BrBG','BuGn','BuPu','Dark2',
                                     'GnBu','Greens','Greys','OrRd','Oranges','PRGn'
                                     ,'Paired','Pastel1','Pastel2','PiYG','PuBu','PuBuGn'
                                     ,'PuOr','PuRd','Purples','RdBu','RdGy','RdPu','RdYlBu'
                                     ,'RdYlGn','Reds','Set1','Set2','Set3','Spectral'
                                     ,'YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot','autumn'
                                     ,'binary','bone','brg','bwr','cool','coolwarm'
                                     ,'copper','cubehelix','flag','gist_earth','gist_gray'
                                     ,'gist_heat','gist_ncar','gist_rainbow','gist_stern'
                                     ,'gist_yarg','gnuplot','gnuplot2','gray','hot'
                                     ,'hsv','jet','ocean','pink','prism','rainbow'
                                     ,'seismic','spectral','spring','summer','terrain'
                                     ,'winter']
                                     
    def change_name(self,key,newkey):
        if newkey not in self.object_name.keys():
            self.object_list[newkey] = self.object_list.pop(key)
            self.object_name[newkey] = self.object_name.pop(key)
            self.object_type[newkey] = self.object_type.pop(key)
        else:
            self.popupmessage("That name already exists on object manager",'Error',itype='error')
        
    def call_top(self,pieces):
        '''
        call_top(...)
            call_top(pieces)
            
            Returns the top object with the name on the pieces list. If pieces
            has also variable name this will be ignored.
            
        Parameters
        ----------
        pieces : list
            List of strings with lenght 1 or 2. The first position is always
            is the one considered the top object (second is variable). Only
            the first position is used.
            
        Returns
        -------
        out: object class
            Returns top object refered to in the list of names.
        
        See also
        --------
        call
        '''
        return self.object_list[pieces[0]]
        
    def call(self,pieces):
        '''
        call(...)
            call(pieces)
            
            Returns the object with the name on the pieces list. If pieces
            has only object name than it will return the data of the first
            variable in that object.
            
        Parameters
        ----------
        pieces : list
            List of strings with lenght 1 or 2. The first position is always
            the name of the object. The second (if exists) is the name of the
            variable on the object which you want to call.
            
        Returns
        -------
        out: object class
            Returns the object variable refered to in the list of names.
        
        See also
        --------
        call_top
        '''
        if len(pieces)==1:
            variable = self.object_list[pieces[0]].get_variable_names()[0]
            return self.object_list[pieces[0]].variable[variable]
        elif len(pieces)==2:
            return self.object_list[pieces[0]].variable[pieces[1]]
        
    def change_name(self,key,newkey):
        '''
        change_name(...)
            change_name(key,newkey)
            
            Changes the name of an existing object.
            
        Parameters
        ----------
        key : string
            String name for the object whose name to be changed.
            
        newkey: string
            String name of the new name for the object key.
            
        Returns
        -------
        out: None
            no return. The object is copied for a new variable and old one is 
            deleted.
        
        See also
        --------
        get_object_names,do_newname_from_name
        '''
        if newkey not in self.object_name.keys():
            self.object_list[newkey] = self.object_list.pop(key)
            self.object_name[newkey] = self.object_name.pop(key)
            self.object_type[newkey] = self.object_type.pop(key)
        else:
            self.__popupmessage__("That name already exists on object manager",'Error',itype='error')
            
    def get_object_names(self):
        '''
        get_object_names(...)
            get_object_names(no arguments)
            
            Gives list with all objects names.
            
        Parameters
        ----------
        None
            
        Returns
        -------
        out: list with objects names (all in object manager).
        
        See also
        --------
        change_name,do_newname_from_name,do_variables_names
        '''
        return self.object_list.keys()
        
    def do_newname_from_name(self,name):
        '''
        get_newname_from_name(...)
            get_newname_from_name(name)
            
            Changes the name by adding a number string.
            
        Parameters
        ----------
        name : string
            String name to be changed into new name.
            
        Returns
        -------
        out: String with the new name
        
        See also
        --------
        change_name, get_object_names, do_variables_names
        '''
        newname = name+'_0'
        i = 1
        while newname in self.object_name:
            newname = name+'_'+str(i)
            i = i + 1
        return newname
        
    def do_variables_names(number):
        '''
        do_variables_names(...)
            do_variables_names(number)
            
            Gives list with numbered strings with Variable_X.
            
        Parameters
        ----------
        number : int
            Length of list of variables names to be created.
            
        Returns
        -------
        out: List with variables names.
        
        See also
        --------
        change_name, get_object_names,do_newname_from_name
        '''
        names = []
        for i in xrange(number):
            names.append('Variable_'+str(i))
        return names
    
    def get_newname_from_type(self,name):
        '''
        get_newname_from_type(...)
            get_newname_from_type(name)
            
            Changes the name by adding a number string.
            Dub that calls get_newname_from_name.
            
        Parameters
        ----------
        name : string
            String name to be changed into new name.
            
        Returns
        -------
        out: String with the new name
        
        See also
        --------
        change_name, get_object_names, do_newname_from_name, do_variables_names
        '''
        return self.get_newname_from_name(name)
        
    def remove_selection(self,selection):
        '''
        remove_selection(...)
            remove_selection(selection)
            
            Deletes object or variable from object manager.
            
        Parameters
        ----------
        selection : list with a string.
            List with string name of the object to be deleted.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if len(selection)==1:
            del(self.object_list[selection[0]])
            del(self.object_type[selection[0]])
            del(self.object_name[selection[0]])
        else:
            if len(self.object_list[selection[0]].variable.keys())==1:
                del(self.object_list[selection[0]])
                del(self.object_type[selection[0]])
                del(self.object_name[selection[0]])
            else:
                del(self.object_list[selection[0]].variable[selection[1]])
                
    def grid_interpolation(self,selection,blocks,size,first,method,name,vname):
        x = (self.call_top(selection).x-first[0])/size[0]
        y = (self.call_top(selection).y-first[1])/size[1]
        z = (self.call_top(selection).z-first[2])/size[2]
        v = self.call(selection).data
        ind = np.where((x > 0) & (x<blocks[0]) & (y > 0) & (y<blocks[1]) & (z > 0) & (z<blocks[2]))
        #print ind,blocks,len(ind[0]),ind[0]
        if len(ind[0])==0:
            return False
        else:
            data = cgrid.grid_interpolation(x,y,z,v,blocks,method)
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_mesh_object(blocks,size,first,-999,name,vname,data)
            self.last_object = name
            return True
                
    def operation_mirror_mesh(self,selection,name):
        '''
        operation_mirror_mesh(...)
            operation_mirror_mesh(selection,name)
            
            Mirrors object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be mirrored.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
            self.object_list[selection[0]].variable[selection[1]].data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
            data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].add_variable(name,data.data)
            self.last_object = selection[0]
            self.last_variable = name

    def operation_categorize_mesh(self,selection,categories,name):
        '''
        operation_categorize_mesh(...)
            operation_categorize_mesh(selection,name)
            
            Categorizes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be categorized.
            
        categories : list
            List with the bins for the categorizations. Minimum and maximum not
            required.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append(np.percentile(self.object_list[selection[0]].variable[selection[1]].data,categories[i])) #(categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            #self.object_list[selection[0]].variable[selection[1]].data.data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            #self.object_list[selection[0]].variable[selection[1]].update_to_masked_array(np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks))            
            data = self.object_list[selection[0]].variable[selection[1]].data.copy()             
            for i in xrange(1,len(ncategories)):
                ind = np.where((data>=ncategories[i-1]) & (data<=ncategories[i]))
                self.object_list[selection[0]].variable[selection[1]].data[ind] = i
                
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append((categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            #data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            data = self.object_list[selection[0]].variable[selection[1]].data.copy()            
            for i in xrange(1,len(ncategories)):
                ind = np.where((self.object_list[selection[0]].variable[selection[1]].data>=ncategories[i-1]) & (self.object_list[selection[0]].variable[selection[1]].data<=ncategories[i]))
                data[ind] = i            
            self.object_list[selection[0]].add_variable(name,data)
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_crop_mesh(self,selection,initial,final,name):
        '''
        operation_crop_mesh(...)
            operation_crop_mesh(selection,name)
            
            Crops object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be cropped.
            
        initial : tuple of ints
            Tuple with the initial position of the crop in the array.
        
        final : tuple of ints
            Tuple with the final position of the crop in the array.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        initialo = (initial[0]-1,initial[1]-1,initial[2]-1)
        data = self.object_list[selection[0]].variable[selection[1]].data.data[initialo[0]:final[0],initialo[1]:final[1],initialo[2]:final[2]]
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (initial[0]*self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  initial[1]*self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,initial[2]*self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def operation_linear_transform_mesh(self,selection,minimum,maximum,name):
        '''
        operation_linear_transform_mesh(...)
            operation_linear_transform_mesh(selection,name)
            
            Give a linear transformation to the object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transformed.
            
        minimum : float
            Minimum value for transformed variable.

        maximum : float
            Maximum value for transformed variable.            
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            self.object_list[selection[0]].variable[selection[1]].data = (self.object_list[selection[0]].variable[selection[1]].data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            data = (self.object_list[selection[0]].variable[selection[1]].data.data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].add_variable(name,data)
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_mirror_point(self,selection,name):
        '''
        operation_mirror_point(...)
            operation_mirror_point(selection,name)
            
            Mirrors object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be mirrored.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()                        
            self.object_list[selection[0]].variable[selection[1]].data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
            data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].add_variable(name,data.data)
            self.last_object = selection[0]
            self.last_variable = name

    def operation_categorize_point(self,selection,categories,name):
        '''
        operation_categorize_point(...)
            operation_categorize_point(selection,name)
            
            Categorizes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be categorized.
            
        categories : list
            List with the bins for the categorizations. Minimum and maximum not
            required.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append(np.percentile(self.object_list[selection[0]].variable[selection[1]],categories[i]))  #(categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            #self.object_list[selection[0]].variable[selection[1]].data.data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            self.object_list[selection[0]].variable[selection[1]].update_to_masked_array(np.digitize(self.object_list[selection[0]].variable[selection[1]].data,ncategories))           
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append((categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data,ncategories)
            self.object_list[selection[0]].add_variable(name,data.astype('float32'))
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_crop_point(self,selection,initial,final,name):
        '''
        operation_crop_point(...)
            operation_crop_point(selection,name)
            
            Crops object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be cropped.
            
        initial : tuple of ints
            Tuple with the initial position of the crop in the array.
        
        final : tuple of ints
            Tuple with the final position of the crop in the array.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        
        x = self.object_list[selection[0]].x
        y = self.object_list[selection[0]].y
        z = self.object_list[selection[0]].z
        ind = np.where((x>=initial[0]) & (x<=final[0]) & (y>=initial[1]) & (y<=final[1]) & (z>=initial[2]) & (z<=final[2]))

        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x[ind],y[ind],z[ind],null,name,vname,self.object_list[selection[0]].variable[selection[1]].data.data[ind])
        self.last_object = name
        
    def operation_linear_transform_point(self,selection,minimum,maximum,name):
        '''
        operation_linear_transform_point(...)
            operation_linear_transform_point(selection,name)
            
            Give a linear transformation to the object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transformed.
            
        minimum : float
            Minimum value for transformed variable.

        maximum : float
            Maximum value for transformed variable.            
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            self.object_list[selection[0]].variable[selection[1]].data = (self.object_list[selection[0]].variable[selection[1]].data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            data = (self.object_list[selection[0]].variable[selection[1]].data.data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].add_variable(name,data)
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_mirror_data(self,selection,name):
        '''
        operation_mirror_data(...)
            operation_mirror_data(selection,name)
            
            Mirrors object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be mirrored.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()                        
            self.object_list[selection[0]].variable[selection[1]].data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
            data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].add_variable(name,data.data,data.dtype,'continuous')
            self.last_object = selection[0]
            self.last_variable = name

    def operation_categorize_data(self,selection,categories,name):
        '''
        operation_categorize_data(...)
            operation_categorize_data(selection,name)
            
            Categorizes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be categorized.
            
        categories : list
            List with the bins for the categorizations. Minimum and maximum not
            required.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append(np.percentile(self.object_list[selection[0]].variable[selection[1]],categories[i])) #(categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            #self.object_list[selection[0]].variable[selection[1]].data.data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            self.object_list[selection[0]].variable[selection[1]].update_to_masked_array(np.digitize(self.object_list[selection[0]].variable[selection[1]].data,ncategories))           
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append((categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data,ncategories)
            self.object_list[selection[0]].add_variable(name,data.astype('float32'),data.dtype,'discrete')
            self.last_object = selection[0]
            self.last_variable = name
        
    def operation_linear_transform_data(self,selection,minimum,maximum,name):
        '''
        operation_linear_transform_point(...)
            operation_linear_transform_point(selection,name)
            
            Give a linear transformation to the object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transformed.
            
        minimum : float
            Minimum value for transformed variable.

        maximum : float
            Maximum value for transformed variable.            
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            self.object_list[selection[0]].variable[selection[1]].data = (self.object_list[selection[0]].variable[selection[1]].data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            data = (self.object_list[selection[0]].variable[selection[1]].data.data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].add_variable(name,data,data.dtype,'continuous')
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_mirror_surf(self,selection,name):
        '''
        operation_mirror_surf(...)
            operation_mirror_surf(selection,name)
            
            Mirrors object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be mirrored.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()                        
            self.object_list[selection[0]].variable[selection[1]].data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            middle = (self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min())/2+self.object_list[selection[0]].variable[selection[1]].data.min()
            data = 2*middle-self.object_list[selection[0]].variable[selection[1]].data
            self.object_list[selection[0]].add_variable(name,data.data)
            self.last_object = selection[0]
            self.last_variable = name

    def operation_categorize_surf(self,selection,categories,name):
        '''
        operation_categorize_surf(...)
            operation_categorize_surf(selection,name)
            
            Categorizes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be categorized.
            
        categories : list
            List with the bins for the categorizations. Minimum and maximum not
            required.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append(np.percentile(self.object_list[selection[0]].variable[selection[1]],categories[i])) #(categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            #self.object_list[selection[0]].variable[selection[1]].data.data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            self.object_list[selection[0]].variable[selection[1]].update_to_masked_array(np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks))            
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            ncategories = [self.object_list[selection[0]].variable[selection[1]].data.min()-0.00001]
            trange = self.object_list[selection[0]].variable[selection[1]].data.max()-self.object_list[selection[0]].variable[selection[1]].data.min()
            for i in xrange(len(categories)):
                ncategories.append((categories[i]/100)*trange+ncategories[0])
            ncategories.append(self.object_list[selection[0]].variable[selection[1]].data.max()+0.00001)
            data = np.digitize(self.object_list[selection[0]].variable[selection[1]].data.flatten(),ncategories).reshape(self.object_list[selection[0]].blocks)
            self.object_list[selection[0]].add_variable(name,data.astype('float32'))
            self.last_object = selection[0]
            self.last_variable = name
            
    def operation_crop_surf(self,selection,initial,final,name):
        '''
        operation_crop_surf(...)
            operation_crop_surf(selection,name)
            
            Crops object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be cropped.
            
        initial : tuple of ints
            Tuple with the initial position of the crop in the array.
        
        final : tuple of ints
            Tuple with the final position of the crop in the array.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        initialo = (initial[0]-1,initial[1]-1,initial[2]-1)
        data = self.object_list[selection[0]].variable[selection[1]].data.data[initialo[0]:final[0],initial[1]:final[1],initialo[2]:final[2]]
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (initial[0]*self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  initial[1]*self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,initial[2]*self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def operation_linear_transform_surf(self,selection,minimum,maximum,name):
        '''
        operation_linear_transform_surf(...)
            operation_linear_transform_surf(selection,name)
            
            Give a linear transformation to the object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transformed.
            
        minimum : float
            Minimum value for transformed variable.

        maximum : float
            Maximum value for transformed variable.            
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # middle = (mesh.max()-mesh.min())/2+mesh.min()
        # mesh[:]=2*middle-mesh[:]
        if type(name)==bool:
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            self.object_list[selection[0]].variable[selection[1]].data = (self.object_list[selection[0]].variable[selection[1]].data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].variable[selection[1]].update_basic_statistics()
        else:
            g = 0
            while name in self.object_list[selection[0]].get_variable_names(): 
                name = name+'_'+str(g)
                g=g+1
            dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            data = (self.object_list[selection[0]].variable[selection[1]].data.data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            self.object_list[selection[0]].add_variable(name,data)
            self.last_object = selection[0]
            self.last_variable = name
            
    def geometric_pickle_mesh(self,selection,stepx,stepy,stepz,name):
        '''
        geometric_pickle_mesh(...)
            geometric_pickle_mesh(selection,stepx,stepy,stepz,name)
            
            Pickles object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be pickled.
            
        stepx : int
            Step for the mesh pickling in the X direction (0).
            
        stepy : int
            Step for the mesh pickling in the Y direction (0).
            
        stepz : int
            Step for the mesh pickling in the Z direction (0).
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks_appex = self.object_list[selection[0]].blocks
        data = self.object_list[selection[0]].variable[selection[1]].data.data[0:blocks_appex[0]:stepx,0:blocks_appex[1]:stepy,0:blocks_appex[2]:stepz]
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0]*stepx,self.object_list[selection[0]].size[1]*stepy,self.object_list[selection[0]].size[2]*stepz)
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_repeat_mesh(self,selection,repx,repy,repz,name):
        '''
        geometric_repeat_mesh(...)
            geometric_repeat_mesh(selection,newx,newy,newz,name)
            
            Downscales object variable of object manager using repetition.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be repeated.
            
        repx : int
             Number of repetitions for each X node.
            
        repy : int
            Number of repetitions for each Y node.
            
        repz : int
            Number of repetitions for each Z node.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.repeat(np.repeat(np.repeat(self.object_list[selection[0]].variable[selection[1]].data.data,repx, axis=0),repy, axis=1), repz, axis=2)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0]/repx,self.object_list[selection[0]].size[1]/repy,self.object_list[selection[0]].size[2]/repz)
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_expand_mesh(self,selection,newx,newy,newz,name):
        '''
        geometric_expand_mesh(...)
            geometric_expand_mesh(selection,newx,newy,newz,name)
            
            Expand (resizish) object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be expanded.
            
        newx : int
             New size for number of blocks in X (0).
            
        newy : int
            New size for number of blocks in Y (1).
            
        newz : int
            New size for number of blocks in Z (2).
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.resize(self.object_list[selection[0]].variable[selection[1]].data.data,(newx,newy,newz))
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_tile_mesh(self,selection,tilex,tiley,tilez,name):
        '''
        geometric_repeat_mesh(...)
            geometric_repeat_mesh(selection,newx,newy,newz,name)
            
            Tiles object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be tilled.
            
        tilex : int
             Number of tiles for each X node.
            
        tiley : int
            Number of tiles for each Y node.
            
        tilez : int
            Number of tiles for each Z node.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.tile(self.object_list[selection[0]].variable[selection[1]].data.data,(tilex,tiley,tilez))
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_transpose_mesh(self,selection,name):
        '''
        geometric_transpose_mesh(...)
            geometric_transpose_mesh(selection,name)
            
            Transposes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transposed.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.transpose(self.object_list[selection[0]].variable[selection[1]].data.data)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_shift_mesh(self,selection,shiftx,shifty,shiftz,name):
        '''
        geometric_shift_mesh(...)
            geometric_shift_mesh(selection,shiftx,shifty,shiftz,name)
            
            Shifts elements of object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be shifted.
            
        shiftx : int
            Size of shift for X.
            
        shifty : int
            Size of shift for Y.
            
        shiftz : int
            Size of shift for Z.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.roll(self.object_list[selection[0]].variable[selection[1]].data.data,shiftx,axis=0)
        data = np.roll(data,shifty,axis=1)
        data = np.roll(data,shiftz,axis=2)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipx_mesh(self,selection,name):
        '''
        geometric_flipx_mesh(...)
            geometric_flipx_mesh(selection,name)
            
            Flip object variable of object manager for axis X.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.flipud(self.object_list[selection[0]].variable[selection[1]].data.data)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipy_mesh(self,selection,name):
        '''
        geometric_flipy_mesh(...)
            geometric_flipy_mesh(selection,name)
            
            Flip object variable of object manager for axis Y.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.fliplr(self.object_list[selection[0]].variable[selection[1]].data.data)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_rotate_mesh(self,selection,rotations,name):
        '''
        geometric_rotate_mesh(...)
            geometric_rotate_mesh(selection,rotations,name)
            
            Rotates object variable of object manager by rotations times 90.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be rotated.
            
        rotations: int
            Number of rotations of 90 degrees.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.rot90(self.object_list[selection[0]].variable[selection[1]].data.data,rotations)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_swap_mesh(self,selection,axis0,axis1,name):
        '''
        geometric_swap_mesh(...)
            geometric_swap_mesh(selection,axis1,axis2,name)
            
            Swaps axis in object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be rotated.
            
        axis0: string
            String (X,Y or Z) of the axis to be swapped.
            
        axis1: string
            String (X,Y or Z) of the axis to be swapped to.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        if axis0 == 'X': a0 = 0
        elif axis0 == 'Y': a0 = 1
        elif axis0 == 'Z': a0 = 2
        if axis1 == 'X': a1 = 0
        elif axis1 == 'Y': a1 = 1
        elif axis1 == 'Z': a1 = 2
        data = np.swapaxes(self.object_list[selection[0]].variable[selection[1]].data.data,a0,a1)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_mesh_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_pad_mesh(self,selection,edges,mode,name,reflect_type):
        '''
        geometric_pad_mesh(...)
            geometric_pad_mesh(selection,edges,mode,name,reflect_type)
            
            Pads (adds to edges) the object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be padded.
            
        edges : tuple
            Tuple of tuples of ints with the number of values to be added in
            all (X, Y and Z) directions. One tuple for each.
            
        mode: string
            String (mean,median,minimum,maximum,reflect,symmetric) of the pad mode.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        reflect_type: string
            String (even or odd) of the rflect type for reflect or symmetric modes.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        # Returns False since in this numpy version there is no numpy pad.
        return False
        
    def geometric_pickle_point(self,selection,stepx,name):
        '''
        geometric_pickle_point(...)
            geometric_pickle_point(selection,stepx,name)
            
            Pickles object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be pickled.
            
        stepx : int
            Step for the mesh pickling in any direction (0).
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks_appex = self.object_list[selection[0]].variable[selection[1]].data.shape[0]
        data = self.object_list[selection[0]].variable[selection[1]].data.data[0:blocks_appex:stepx]
        x = self.object_list[selection[0]].x[0:blocks_appex:stepx]
        y = self.object_list[selection[0]].y[0:blocks_appex:stepx]
        z = self.object_list[selection[0]].z[0:blocks_appex:stepx]
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,data)
        self.last_object = name
        
    def geometric_repeat_point(self,selection,repx,name):
        '''
        geometric_repeat_point(...)
            geometric_repeat_point(selection,newx,newy,newz,name)
            
            Downscales object variable of object manager using repetition.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be repeated.
            
        repx : int
             Number of repetitions value.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.repeat(self.object_list[selection[0]].variable[selection[1]].data.data,repx, axis=0)
        x = np.repeat(self.object_list[selection[0]].x,repx, axis=0)
        y = np.repeat(self.object_list[selection[0]].y,repx, axis=0)
        z = np.repeat(self.object_list[selection[0]].z,repx, axis=0)
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipx_point(self,selection,name):
        '''
        geometric_flipx_point(...)
            geometric_flipx_point(selection,name)
            
            Flip object variable of object manager for axis X.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = self.object_list[selection[0]].variable[selection[1]].data.data
        x = self.object_list[selection[0]].x[::-1]
        y = self.object_list[selection[0]].y.copy()
        z = self.object_list[selection[0]].z.copy()
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipy_point(self,selection,name):
        '''
        geometric_flipy_point(...)
            geometric_flipy_point(selection,name)
            
            Flip object variable of object manager for axis Y.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = self.object_list[selection[0]].variable[selection[1]].data.data
        x = self.object_list[selection[0]].x.copy()
        y = self.object_list[selection[0]].y[::-1]
        z = self.object_list[selection[0]].z.copy()
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipz_point(self,selection,name):
        '''
        geometric_flipz_point(...)
            geometric_flipz_point(selection,name)
            
            Flip object variable of object manager for axis Z.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = self.object_list[selection[0]].variable[selection[1]].data.data
        x = self.object_list[selection[0]].x.copy()
        y = self.object_list[selection[0]].y.copy()
        z = self.object_list[selection[0]].z[::-1]
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,data)
        self.last_object = name
        
    def geometric_pickle_surf(self,selection,stepx,stepy,name):
        '''
        geometric_pickle_surf(...)
            geometric_pickle_surf(selection,stepx,stepy,name)
            
            Pickles object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be pickled.
            
        stepx : int
            Step for the mesh pickling in the X direction (0).
            
        stepy : int
            Step for the mesh pickling in the Y direction (0).
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks_appex = self.object_list[selection[0]].blocks
        data = self.object_list[selection[0]].variable[selection[1]].data.data[0:blocks_appex[0]:stepx,0:blocks_appex[1]:stepy,:]
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0]*stepx,self.object_list[selection[0]].size[1]*stepy,self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_repeat_surf(self,selection,repx,repy,name):
        '''
        geometric_repeat_surf(...)
            geometric_repeat_surf(selection,newx,newy,newz,name)
            
            Downscales object variable of object manager using repetition.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be repeated.
            
        repx : int
             Number of repetitions for each X node.
            
        repy : int
            Number of repetitions for each Y node.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.repeat(np.repeat(self.object_list[selection[0]].variable[selection[1]].data.data,repx, axis=0),repy, axis=1)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0]/repx,self.object_list[selection[0]].size[1]/repy,self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_expand_surf(self,selection,newx,newy,name):
        '''
        geometric_expand_surf(...)
            geometric_expand_surf(selection,newx,newy,newz,name)
            
            Expand (resizish) object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be expanded.
            
        newx : int
             New size for number of blocks in X (0).
            
        newy : int
            New size for number of blocks in Y (1).
            
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.resize(self.object_list[selection[0]].variable[selection[1]].data.data,(newx,newy,1))
        blocks = data.shape #(final[0]-initialo[0]-1,final[1]-initialo[1]-1,final[2]-initialo[2]-1)
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_tile_surf(self,selection,tilex,tiley,name):
        '''
        geometric_repeat_surf(...)
            geometric_repeat_surf(selection,newx,newy,newz,name)
            
            Tiles object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be tilled.
            
        tilex : int
             Number of tiles for each X node.
            
        tiley : int
            Number of tiles for each Y node.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.tile(self.object_list[selection[0]].variable[selection[1]].data.data,(tilex,tiley,1))
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_transpose_surf(self,selection,name):
        '''
        geometric_transpose_surf(...)
            geometric_transpose_surf(selection,name)
            
            Transposes object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be transposed.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.transpose(self.object_list[selection[0]].variable[selection[1]].data.data)
        
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_shift_surf(self,selection,shiftx,shifty,name):
        '''
        geometric_shift_surf(...)
            geometric_shift_surf(selection,shiftx,shifty,shiftz,name)
            
            Shifts elements of object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be shifted.
            
        shiftx : int
            Size of shift for X.
            
        shifty : int
            Size of shift for Y.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.roll(self.object_list[selection[0]].variable[selection[1]].data.data,shiftx,axis=0)
        data = np.roll(data,shifty,axis=1)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipx_surf(self,selection,name):
        '''
        geometric_flipx_surf(...)
            geometric_flipx_surf(selection,name)
            
            Flip object variable of object manager for axis X.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.flipud(self.object_list[selection[0]].variable[selection[1]].data.data)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_flipy_surf(self,selection,name):
        '''
        geometric_flipy_surf(...)
            geometric_flipy_surf(selection,name)
            
            Flip object variable of object manager for axis Y.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be flipped.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.fliplr(self.object_list[selection[0]].variable[selection[1]].data.data)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def geometric_rotate_surf(self,selection,rotations,name):
        '''
        geometric_rotate_surf(...)
            geometric_rotate_surf(selection,rotations,name)
            
            Rotates object variable of object manager by rotations times 90.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be rotated.
            
        rotations: int
            Number of rotations of 90 degrees.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        data = np.rot90(self.object_list[selection[0]].variable[selection[1]].data.data,rotations)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
            
    def geometric_swap_surf(self,selection,axis0,axis1,name):
        '''
        geometric_swap_surf(...)
            geometric_swap_surf(selection,axis1,axis2,name)
            
            Swaps axis in object variable of object manager.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be swaped.
            
        axis0: string
            String (X,Y) of the axis to be swapped.
            
        axis1: string
            String (X,Y) of the axis to be swapped to.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        if axis0 == 'X': a0 = 0
        elif axis0 == 'Y': a0 = 1
        #elif axis0 == 'Z': a0 = 2
        if axis1 == 'X': a1 = 0
        elif axis1 == 'Y': a1 = 1
        #elif axis1 == 'Z': a1 = 2
        data = np.swapaxes(self.object_list[selection[0]].variable[selection[1]].data.data,a0,a1)
        blocks = data.shape
        first  = (self.object_list[selection[0]].size[0]+self.object_list[selection[0]].first[0],
                  self.object_list[selection[0]].size[1]+self.object_list[selection[0]].first[1]
                  ,self.object_list[selection[0]].size[2]+self.object_list[selection[0]].first[2])
        size   = (self.object_list[selection[0]].size[0],self.object_list[selection[0]].size[1],self.object_list[selection[0]].size[2])
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,vname,data)
        self.last_object = name
        
    def calculator_constant(self,selection,operation,constant,name):
        '''
        calculator_constant(...)
            calculator_constant(selection,operation,constant,name)
            
            Calculates the result between an object and a constant.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be used.
            
        operation: string
            String add,subtract,multiply,divide.
            
        constant: float
            Constant to be used in operations.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if self.object_type[selection[0]] == 'mesh':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + constant
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - constant
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*constant
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/constant
            blocks = data.shape
            first  = self.object_list[selection[0]].first
            size   = self.object_list[selection[0]].size
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_mesh_object(blocks,size,first,null,name,vname,data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'surf':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + constant
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - constant
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*constant
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/constant
            blocks = data.shape
            first  = self.object_list[selection[0]].first
            size   = self.object_list[selection[0]].size
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_surf_object(blocks,size,first,null,name,vname,data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'point':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + constant
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - constant
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*constant
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/constant      
            x = self.object_list[selection[0]].x
            y = self.object_list[selection[0]].y
            z = self.object_list[selection[0]].z   
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_point_object(x,y,z,null,name,vname,data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'data':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + constant
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - constant
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*constant
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/constant      
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_data_object(null,name,vname,data,data.dtype,self.object_list[selection[0]].variable[selection[1]].vtype)
            self.last_object = name
            
    def calculator_object(self,selection,operation,selection2,name):
        '''
        calculator_object(...)
            calculator_object(selection,operation,selection2,name)
            
            Calculates the result between an object and another object.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be used.
            
        operation: string
            String add,subtract,multiply,divide.
            
        selection2: float
            List with string name of the object and the variable to be used.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if self.object_type[selection[0]] == 'mesh' and self.object_type[selection2[0]] == 'mesh':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data
            blocks = data.shape
            first  = self.object_list[selection[0]].first
            size   = self.object_list[selection[0]].size
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_mesh_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'surf' and self.object_type[selection2[0]] == 'surf':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data
            blocks = data.shape
            first  = self.object_list[selection[0]].first
            size   = self.object_list[selection[0]].size
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_surf_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'point' and self.object_type[selection2[0]] == 'point':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data     
            x = self.object_list[selection[0]].x
            y = self.object_list[selection[0]].y
            z = self.object_list[selection[0]].z   
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_point_object(x,y,z,null,name,vname+'_'+selection2[1],data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'point' and self.object_type[selection2[0]] == 'data':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data     
            x = self.object_list[selection[0]].x
            y = self.object_list[selection[0]].y
            z = self.object_list[selection[0]].z   
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_point_object(x,y,z,null,name,vname+'_'+selection2[1],data)
            self.last_object = name
        elif self.object_type[selection[0]] == 'data' and self.object_type[selection2[0]] == 'point':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data     
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_data_object(null,name,vname+'_'+selection2[1],data,data.dtype,self.object_list[selection[0]].variable[selection[1]].vtype)
            self.last_object = name
        elif self.object_type[selection[0]] == 'data' and self.object_type[selection2[0]] == 'data':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            if operation == 'add': data = self.object_list[selection[0]].variable[selection[1]].data + self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'subtract': data = self.object_list[selection[0]].variable[selection[1]].data - self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'multiply': data = self.object_list[selection[0]].variable[selection[1]].data*self.object_list[selection2[0]].variable[selection2[1]].data
            elif operation == 'divide': data = self.object_list[selection[0]].variable[selection[1]].data/self.object_list[selection2[0]].variable[selection2[1]].data     
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_data_object(null,name,vname+'_'+selection2[1],data,data.dtype,self.object_list[selection[0]].variable[selection[1]].vtype)
            self.last_object = name
            
    def calculator_global(self,selection,operation):
        '''
        calculator_global(...)
            calculator_global(selection,operation,name)
            
            Calculates the global result of an object.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be used.
            
        operation: string
            String sum,product,stdeviation.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if self.object_type[selection[0]] == 'mesh':
            if operation == 'sum': data = self.object_list[selection[0]].variable[selection[1]].data.sum()
            elif operation == 'product': data = self.object_list[selection[0]].variable[selection[1]].data.prod()
            elif operation == 'stdeviation': data = self.object_list[selection[0]].variable[selection[1]].data.std()
            return data
        elif self.object_type[selection[0]] == 'surf':
            if operation == 'sum': data = self.object_list[selection[0]].variable[selection[1]].data.sum()
            elif operation == 'product': data = self.object_list[selection[0]].variable[selection[1]].data.prod()
            elif operation == 'stdeviation': data = self.object_list[selection[0]].variable[selection[1]].data.std()
            return data
        elif self.object_type[selection[0]] == 'point':
            if operation == 'sum': data = self.object_list[selection[0]].variable[selection[1]].data.sum()
            elif operation == 'product': data = self.object_list[selection[0]].variable[selection[1]].data.prod()
            elif operation == 'stdeviation': data = self.object_list[selection[0]].variable[selection[1]].data.std()
            return data
        elif self.object_type[selection[0]] == 'data':
            if operation == 'sum': data = self.object_list[selection[0]].variable[selection[1]].data.sum()
            elif operation == 'product': data = self.object_list[selection[0]].variable[selection[1]].data.prod()
            elif operation == 'stdeviation': data = self.object_list[selection[0]].variable[selection[1]].data.std()
            return data
            
    def related_above_or_below_surface(self,selection,selection2,method,const):
        surf = self.call(selection).data
        mesh = self.call(selection2).data.data
        zcoords = self.call_top(selection2).zcoords
        if self.object_type[selection[0]]=='surf' and self.object_type[selection2[0]]=='mesh':
            if method == 'Above':
                dialog = wx.ProgressDialog ( 'Progress', 'Applying constant above surface.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
                for x in xrange(self.call_top(selection2).blocks[0]):
                    for y in xrange(self.call_top(selection2).blocks[1]):
                        for z in xrange(self.call_top(selection2).blocks[2]):
                            if zcoords[x,y,z]>surf[x,y]:
                                mesh[x,y,z] = const
                    dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
                self.call(selection2).update_to_masked_array(mesh)
                self.call(selection2).update_basic_statistics()
            elif method == 'Below':
                dialog = wx.ProgressDialog ( 'Progress', 'Applying constant below surface.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
                for x in xrange(self.call_top(selection2).blocks[0]):
                    for y in xrange(self.call_top(selection2).blocks[1]):
                        for z in xrange(self.call_top(selection2).blocks[2]):
                            if zcoords[x,y,z]<surf[x,y]:
                                mesh[x,y,z] = const
                    dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
                self.call(selection2).update_to_masked_array(mesh)
                self.call(selection2).update_basic_statistics()
                
    def related_in_between_surface(self,selection,selection2,selection3,method,const):
        surf = self.call(selection).data
        surf2 = self.call(selection3).data
        mesh = self.call(selection2).data.data
        zcoords = self.call_top(selection2).zcoords
        if self.object_type[selection[0]]=='surf' and self.object_type[selection2[0]]=='mesh' and self.object_type[selection3[0]]=='surf':
            #print 1
            if method == 'Inside':
                #print 2
                dialog = wx.ProgressDialog ( 'Progress', 'Applying constant inside surfaces.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
                for x in xrange(self.call_top(selection2).blocks[0]):
                    for y in xrange(self.call_top(selection2).blocks[1]):
                        for z in xrange(self.call_top(selection2).blocks[2]):
                            if zcoords[x,y,z]<surf[x,y] and zcoords[x,y,z]>surf2[x,y]:
                                mesh[x,y,z] = const
                    dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
                self.call(selection2).update_to_masked_array(mesh)
                self.call(selection2).update_basic_statistics()
            elif method == 'Outside':
                dialog = wx.ProgressDialog ( 'Progress', 'Applying constant outside surfaces.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
                for x in xrange(self.call_top(selection2).blocks[0]):
                    for y in xrange(self.call_top(selection2).blocks[1]):
                        for z in xrange(self.call_top(selection2).blocks[2]):
                            if zcoords[x,y,z]>surf[x,y]:
                                mesh[x,y,z] = const
                            elif zcoords[x,y,z]<surf2[x,y]:
                                mesh[x,y,z] = const
                    dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
                self.call(selection2).update_to_masked_array(mesh)
                self.call(selection2).update_basic_statistics()
                
    def related_flatten_surface(self,selection,selection2,name):
        surf = self.call(selection).data
        surf = np.int_(surf-surf.min())
        mesh = self.call(selection2).data.data
        sizez = mesh.shape[2]
        new_mesh = np.ones((mesh.shape[0],mesh.shape[1],mesh.shape[2]+surf.max()),dtype = mesh.dtype)*self.call(selection2).null
        if self.object_type[selection[0]]=='surf' and self.object_type[selection2[0]]=='mesh':
            dialog = wx.ProgressDialog ( 'Progress', 'Flattening with surface.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for x in xrange(self.call_top(selection2).blocks[0]):
                for y in xrange(self.call_top(selection2).blocks[1]):
                    #for z in xrange(self.call_top(selection2).blocks[2]):
                    new_mesh[x,y,surf[x,y,0]:(surf[x,y,0]+sizez)] = mesh[x,y,:]
                dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
            blocks = new_mesh.shape
            size = self.call_top(selection2).size
            first = self.call_top(selection2).first
            null = self.call_top(selection2).null
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_mesh_object(blocks,size,first,null,name,selection[1],new_mesh)
            self.last_object = name
            
    def related_unflatten_surface(self,selection,selection2,name):
        surf = self.call(selection).data.copy()
        surf = np.int_((surf-surf.min()))
        surf = np.abs(surf-surf.max())
        mesh = self.call(selection2).data.data
        sizez = mesh.shape[2]
        new_mesh = np.ones((mesh.shape[0],mesh.shape[1],mesh.shape[2]+surf.max()),dtype = mesh.dtype)*self.call(selection2).null
        if self.object_type[selection[0]]=='surf' and self.object_type[selection2[0]]=='mesh':
            dialog = wx.ProgressDialog ( 'Progress', 'unFlattening with surface.', maximum = self.call_top(selection2).blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
            for x in xrange(self.call_top(selection2).blocks[0]):
                for y in xrange(self.call_top(selection2).blocks[1]):
                    #print (surf[x,y,0]-sizez)
                    #for z in xrange(self.call_top(selection2).blocks[2]):
                    new_mesh[x,y,surf[x,y,0]:(surf[x,y,0]+sizez)] = mesh[x,y,:]
                dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(self.call_top(selection2).blocks[0]-1) )
            blocks = new_mesh.shape
            size = self.call_top(selection2).size
            first = self.call_top(selection2).first
            null = self.call_top(selection2).null
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_mesh_object(blocks,size,first,null,name,selection[1],new_mesh)
            self.last_object = name
            
    def related_flatten_point_surface(self,selection,selection2,name):
        surf = self.call(selection).data.copy()
        surf = np.int_(surf-surf.min())
        x = self.call_top(selection2).x
        y = self.call_top(selection2).y
        z = self.call_top(selection2).z
        ind = np.where((x>=self.call_top(selection).first[0]) & (x<=self.call_top(selection).first[0]+self.call_top(selection).size[0]*self.call_top(selection).blocks[0]) & (y>=self.call_top(selection).first[1]) & (y<=self.call_top(selection).first[1]+self.call_top(selection).size[1]*self.call_top(selection).blocks[1]))
        x = x[ind]
        y = y[ind]
        z = z[ind]
        indx = np.int_((x-self.call_top(selection).first[0])/self.call_top(selection).size[0])
        indy = np.int_((y-self.call_top(selection).first[1])/self.call_top(selection).size[1])
        data = self.call(selection2).data.data[ind]
        z = z+surf[indx,indy,0]
        null = self.call_top(selection2).null
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.add_point_object(x,y,z,null,name,selection2[1],data)
        self.last_object = name
        
    def related_unflatten_point_surface(self,selection,selection2,name):
        surf = self.call(selection).data.copy()
        surf = np.int_(surf-surf.min())
        x = self.call_top(selection2).x
        y = self.call_top(selection2).y
        z = self.call_top(selection2).z
        ind = np.where((x>=self.call_top(selection).first[0]) & (x<=self.call_top(selection).first[0]+self.call_top(selection).size[0]*self.call_top(selection).blocks[0]) & (y>=self.call_top(selection).first[1]) & (y<=self.call_top(selection).first[1]+self.call_top(selection).size[1]*self.call_top(selection).blocks[1]))
        x = x[ind]
        y = y[ind]
        z = z[ind]
        indx = np.int_((x-self.call_top(selection).first[0])/self.call_top(selection).size[0])
        indy = np.int_((y-self.call_top(selection).first[1])/self.call_top(selection).size[1])
        data = self.call(selection2).data.data[ind]
        z = z-surf[indx,indy,0]
        null = self.call_top(selection2).null
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.add_point_object(x,y,z,null,name,selection2[1],data)
        self.last_object = name
        
    def related_shovel_surface(self,selection,selection2,selection3,name,const):
        blocks = self.call_top(selection).blocks
        coords = np.zeros((np.prod(blocks),2),dtype='int32')
        i = 0
        for x in xrange(blocks[0]):
            for y in xrange(blocks[1]):
                coords[i,0] = x
                coords[i,1] = y
                i = i + 1
        x = np.int_((self.call(selection2).data-self.call_top(selection).first[0])/self.call_top(selection).size[0])
        y = np.int_((self.call(selection3).data-self.call_top(selection).first[1])/self.call_top(selection).size[1])
        shape = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
        res = pympl.points_inside_polygon(coords,shape)
        ind = np.where(res==True)
        data = self.call(selection).data.data.copy()
        data[coords[ind[0],0],coords[ind[0],1]] = const
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks = data.shape
        first  = self.object_list[selection[0]].first
        size   = self.object_list[selection[0]].size
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,selection[1],data)
        self.last_object = name
        
    def related_balanced_shovel_surface(self,selection,selection2,selection3,name,const):
        blocks = self.call_top(selection).blocks
        coords = np.zeros((np.prod(blocks),2),dtype='int32')
        i = 0
        for x in xrange(blocks[0]):
            for y in xrange(blocks[1]):
                coords[i,0] = x
                coords[i,1] = y
                i = i + 1
        x = np.int_((self.call(selection2).data-self.call_top(selection).first[0])/self.call_top(selection).size[0])
        y = np.int_((self.call(selection3).data-self.call_top(selection).first[1])/self.call_top(selection).size[1])
        shape = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
        res = pympl.points_inside_polygon(coords,shape)
        ind = np.where(res==True)
        data = self.call(selection).data.data.copy()
        data[coords[ind[0],0],coords[ind[0],1]] = data[coords[ind[0],0],coords[ind[0],1]]-const
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks = data.shape
        first  = self.object_list[selection[0]].first
        size   = self.object_list[selection[0]].size
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        vname = selection[1]
        self.add_surf_object(blocks,size,first,null,name,selection[1],data)
        self.last_object = name
        
    def related_slope_map_surface(self,selection,name):
        blocks = self.call_top(selection).blocks
        surf = self.call(selection).data
        xcoords = self.call_top(selection).xcoords
        ycoords = self.call_top(selection).ycoords
        data = np.zeros(blocks,dtype='float32')
        m = np.max(blocks)
        dialog = wx.ProgressDialog ( 'Progress', 'Calculating surface slope.', maximum = blocks[0]-1, style = wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_AUTO_HIDE )
        for x in xrange(blocks[0]):
            for y in xrange(blocks[1]):
                a = np.clip((x-1,y-1),0,m)
                b = np.array([x+2,y+2])
                #print a,b,x,y,data.shape,xcoords.shape,ycoords.shape
                if surf.mask[x,y,0]: data[x,y,0] = surf.data[x,y,0]
                else:
                    ext = np.sqrt((xcoords[a[0]:b[0],a[1]:b[1]]-xcoords[x,y])**2+(ycoords[a[0]:b[0],a[1]:b[1]]-ycoords[x,y])**2)
                    ext[np.where(ext==0)] = 1000000
                    data[x,y,0] = np.max(np.abs((surf.data[a[0]:b[0],a[1]:b[1],0]-surf.data[x,y,0])/ext))
            dialog.Update ( x, 'Step...'+'  '+repr(x)+'   of   '+repr(blocks[0]-1) )
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        blocks = data.shape
        first  = [self.object_list[selection[0]].first[0],self.object_list[selection[0]].first[1],self.call(selection).data.min()]
        size   = self.object_list[selection[0]].size
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        self.add_mesh_object(blocks,size,first,null,name,selection[1],data)
        self.last_object = name
        
    def related_upscale_mesh(self,selection,name,choice,steps):
        data = cgrid.grid_upscale(self.call(selection).data,choice,steps)
        blocks = data.shape
        size = (self.call_top(selection).size[0]*steps[0],self.call_top(selection).size[1]*steps[1],self.call_top(selection).size[2]*steps[2])
        first = self.call_top(selection).first
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        self.add_mesh_object(blocks,size,first,null,name,selection[1],data)
        self.last_object = name
        
    def related_attribute_mesh(self,selection,name,choice,window,order,percentile,rank,axis):
        """
        choice = ['Uniform','Gaussian','Minimum','Maximum','Median','Percentile',
                  'Rank','Prewitt','Sobel','Laplace','Gaussian Laplace','Fourier uniform'
                  ,'Fourier gaussian','Fourier shift','Spline','Local MoranI'
                  ,'Local MoranI with distance','Local GearyC','Local GearyC with distance'
                  ,'G statistics','G statistics with distance','Equalization'
                  ,'Sigmoid','Minimum isoVariance','Maximum isoVariance','Mean isoVariance']
        """
        if choice == 'Uniform': data = cgrid.uniform_filter(self.call(selection).data,window)
        elif choice == 'Gaussian': data = cgrid.gaussian_filter(self.call(selection).data,window,order)
        elif choice == 'Minimum': data = cgrid.order_statistics_filter(self.call(selection).data,window,'minimum', rank)
        elif choice == 'Maximum': data = cgrid.order_statistics_filter(self.call(selection).data,window,'maximum', rank)
        elif choice == 'Median': data = cgrid.order_statistics_filter(self.call(selection).data,window,'median', rank)
        elif choice == 'Percentile': data = cgrid.order_statistics_filter(self.call(selection).data,window,'percentile'+str(percentile), rank)
        elif choice == 'Rank': data = cgrid.order_statistics_filter(self.call(selection).data,window,'rank', rank)
        elif choice == 'Prewitt': data = cgrid.prewitt_filter(self.call(selection).data,axis)
        elif choice == 'Sobel': data = cgrid.sobel_filter(self.call(selection).data,axis)
        elif choice == 'Laplace': data = cgrid.laplace_filter(self.call(selection).data)
        elif choice == 'Gaussian Laplace': data = cgrid.gaussian_laplace_filter(self.call(selection).data,window)
        elif choice == 'Fourier uniform': data = cgrid.fourier_uniform_filter(self.call(selection).data,window,axis)
        elif choice == 'Fourier gaussian': data = cgrid.fourier_gaussian_filter(self.call(selection).data,window,axis)
        elif choice == 'Fourier shift': data = cgrid.fourier_shift_filter(self.call(selection).data,window,axis)
        elif choice == 'Spline': data = cgrid.spline_filter(self.call(selection).data,order)
        elif choice == 'Local MoranI': data = cgrid.local_moranI(self.call(selection).data,window,'constant')
        elif choice == 'Local MoranI with distance': data = cgrid.local_moranI(self.call(selection).data,window,'distance')
        elif choice == 'Local GearyC': data = cgrid.local_gearyC(self.call(selection).data,window,'constant')
        elif choice == 'Local GearyC with distance': data = cgrid.local_gearyC(self.call(selection).data,window,'distance')
        elif choice == 'G statistics': data = cgrid.local_Gstatistics(self.call(selection).data,window,'constant')
        elif choice == 'G statistics with distance': data = cgrid.local_Gstatistics(self.call(selection).data,window,'distance')
        elif choice == 'Equalization': data = cgrid.equalization_procedure(self.call(selection).data)
        elif choice == 'Sigmoid': data = cgrid.sigmoid_procedure(self.call(selection).data)
        elif choice == 'Minimum isoVariance': data = cgrid.local_isoVariance(self.call(selection).data,window,'minimum')
        elif choice == 'Mean isoVariance': data = cgrid.local_isoVariance(self.call(selection).data,window,'mean')
        elif choice == 'Maximum isoVariance': data = cgrid.local_isoVariance(self.call(selection).data,window,'maximum')
        elif choice in ['mean','variance','std','sum','product','peak2peak','signal2noise','skewness','kurtosis']:
            data = cgrid.moving_window_atribute(self.call(selection).data,choice,window)
        else: return False
        blocks = data.shape
        size = self.call_top(selection).size
        first = self.call_top(selection).first
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        if self.object_list[selection[0]].null_flag == False: null = None
        else: null = self.object_list[selection[0]].null
        self.add_mesh_object(blocks,size,first,null,name,selection[1],data)
        self.last_object = name
        
    def related_create_point_set_mesh(self,selection,locations):
        full_size = self.call_top(selection).blocks[2]*locations.shape[0]
        size = self.call_top(selection).size
        first = self.call_top(selection).first
        zsize = self.call_top(selection).blocks[2]
        x = np.zeros(full_size)
        y = np.zeros(full_size)
        z = np.zeros(full_size)
        v = np.zeros(full_size)
        for i in xrange(locations.shape[0]):
            x[i*zsize:(i+1)*zsize] = locations[i,0]
            y[i*zsize:(i+1)*zsize] = locations[i,1]
            z[i*zsize:(i+1)*zsize] = np.arange(zsize)
            v[i*zsize:(i+1)*zsize] = self.call(selection).data.data[np.int_(x[i*zsize:(i+1)*zsize]),np.int_(y[i*zsize:(i+1)*zsize]),np.int_(z[i*zsize:(i+1)*zsize])]
            x[i*zsize:(i+1)*zsize] = x[i*zsize:(i+1)*zsize]*size[0]+first[0]+size[0]
            y[i*zsize:(i+1)*zsize] = y[i*zsize:(i+1)*zsize]*size[1]+first[1]+size[1]
            z[i*zsize:(i+1)*zsize] = z[i*zsize:(i+1)*zsize]*size[2]+first[2]+size[2]
        name = 'POINT_from_'+selection[0]
        if name in self.object_name: 
                name = self.do_newname_from_name(name)
        null = self.call_top(selection).null
        vname = selection[1]
        self.add_point_object(x,y,z,null,name,vname,v)
        self.last_object = name        
        
    def related_add_from_mesh_point(self,selection,selection2,name):
        blocks = self.call_top(selection2).blocks
        size = self.call_top(selection2).size
        first = self.call_top(selection2).first
        x = np.int_((self.call_top(selection).x-first[0])/size[0])
        y = np.int_((self.call_top(selection).y-first[1])/size[1])
        z = np.int_((self.call_top(selection).z-first[2])/size[2])
        v = np.zeros(x.shape[0],dtype='float32')
        v[:] = self.call_top(selection).null
        for i in xrange(x.shape[0]):
            if x[i]>=0 and x[i]<blocks[0] and y[i]>=0 and y[i]<blocks[1] and z[i]>=0 and z[i]<blocks[2]:
                if self.call(selection2).data.mask[x[i],y[i],z[i]] == False:
                    v[i] = self.call(selection2).data[x[i],y[i],z[i]]
        g = 0
        while name in self.object_list[selection[0]].get_variable_names(): 
            name = name+'_'+str(g)
            g=g+1
        self.call_top(selection).add_variable(name,v)
        self.last_object = selection[0]
        self.last_variable = name
        
    def convert_to_world_referential_point(self,selection,selection2,name):
        if self.object_list[selection2[0]].sphere_flag:
            diameter = self.object_list[selection2[0]].sphere_diameter
            xi = self.object_list[selection2[0]].first[0]
            yi = self.object_list[selection2[0]].first[1]
            zi = self.object_list[selection2[0]].first[2]
            xo = self.object_list[selection2[0]].first[0]+self.object_list[selection2[0]].size[0]*self.object_list[selection2[0]].blocks[0]
            yo = self.object_list[selection2[0]].first[1]+self.object_list[selection2[0]].size[1]*self.object_list[selection2[0]].blocks[1]
            # dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            # dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            # data = (self.object_list[selection[0]].variable[selection[1]].data.data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum            
            # theta = np.linspace(0, np.pi, self.blocks[1], endpoint=True)
            # phi = np.linspace(0, 2 * np.pi, self.blocks[0], endpoint=True)
            # theta, phi = np.meshgrid(theta, phi)            
            # self.spherex = self.sphere_diameter * np.sin(theta) * np.cos(phi) + self.first[0]
            # self.spherey = self.sphere_diameter * np.sin(theta) * np.sin(phi) + self.first[1]
            # self.spherez = self.sphere_diameter * np.cos(theta) + self.first[2]            
            phi  = (self.object_list[selection[0]].x-xi)*(np.pi*2-0)/(xo-xi)+0
            theta  = (self.object_list[selection[0]].y-yi)*(np.pi-0)/(yo-yi)+0
            #diameter = self.object_list[selection[0]].z*diameter/zi
            x = (diameter+(self.object_list[selection[0]].z-zi))*np.sin(theta)*np.cos(phi)+xi
            y = (diameter+(self.object_list[selection[0]].z-zi))*np.sin(theta)*np.sin(phi)+yi
            z = (diameter+(self.object_list[selection[0]].z-zi))*np.cos(theta)+zi
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            null = self.call_top(selection).null
            vname = selection[1]
            v = self.call(selection).data
            self.add_point_object(x,y,z,null,name,vname,v)
            self.last_object = name   
        
    def related_convert_point2data(self,selection,name,inherit):
        # add_data_object(self,null,name,vname,data,dtype,vtype)
        data = []
        if inherit:
            x = self.call_top(selection).x.reshape((self.call_top(selection).x.shape[0],1))
            y = self.call_top(selection).y.reshape((self.call_top(selection).x.shape[0],1))
            z = self.call_top(selection).z.reshape((self.call_top(selection).x.shape[0],1))
            data = np.hstack((x,y,z))
            vname = ['X','Y','Z',selection[1]]
            
            data = np.hstack((data,self.call(selection).data.data.reshape(self.call(selection).data.data.shape[0],1)))
            dtype = [x.dtype,y.dtype,z.dtype,data.dtype]
            null = self.call_top(selection).null
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_data_object(null,name,vname,data,dtype,self.call(selection).vtype)
            self.last_object = name
        else:
            vname = selection[1]
            data = self.call(selection).data.data
            null = self.call_top(selection).null
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_data_object(null,name,vname,data,data.dtype,self.call(selection).vtype)
            self.last_object = name
            
    def related_full_convert_point2data(self,selection,name):
        # add_data_object(self,null,name,vname,data,dtype,vtype)
        data = []
        x = self.call_top(selection).x.reshape((self.call_top(selection).x.shape[0],1))
        y = self.call_top(selection).y.reshape((self.call_top(selection).x.shape[0],1))
        z = self.call_top(selection).z.reshape((self.call_top(selection).x.shape[0],1))
        data = np.hstack((x,y,z))
        vname = ['X','Y','Z']
        dtype = [x.dtype,y.dtype,z.dtype,data.dtype]
        vtype = ['continuous','continuous','continuous']
        for i in self.call_top(selection).get_variable_names():
            vname.append(i)
            dtype.append(self.call([selection[0],i]).data.dtype)
            vtype.append(self.call([selection[0],i]).vtype)
            data = np.hstack((data,self.call([selection[0],i]).data.data.reshape(self.call([selection[0],i]).data.data.shape[0],1)))
        null = self.call_top(selection).null
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.add_data_object(null,name,vname,data,dtype,vtype)
        self.last_object = name
        
    def related_convert_data2point(self,selection,name,xselection,yselection,zselection):
        # add_data_object(self,null,name,vname,data,dtype,vtype)
        #print xselection,yselection,zselection
        x = self.call(xselection).data.data.astype('float32') #.reshape((self.call(xselection).data.shape[0],1))
        y = self.call(yselection).data.data.astype('float32') #.reshape((self.call(yselection).data.shape[0],1))
        if type(zselection[0])==bool:
            #z = np.zeros((self.call(xselection).data.shape[0],1),dtype='float32')
            z = np.zeros(self.call(xselection).data.shape[0],dtype='float32')
        else:
            z = self.call(zselection).data.data.astype('float32') #.reshape((self.call(zselection).data.shape[0],1))
        #data = np.hstack((x,y,z))
        vname = [selection[1]]
        
        #data = np.hstack((data,self.call(selection).data.data.reshape(self.call(selection).data.data.shape[0],1)))
        data = self.call(selection).data.data
        null = self.call_top(selection).null
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.add_point_object(x,y,z,null,name,selection[1],data)
        self.last_object = name
            
    def related_full_convert_data2point(self,selection,name,xselection,yselection,zselection):
        # add_data_object(self,null,name,vname,data,dtype,vtype)
        data = []
        choices = self.call_top(selection).get_numeric_variable_names()
        x = self.call(xselection).data.data.astype('float32') #.reshape((self.call(xselection).data.shape[0],1))
        y = self.call(yselection).data.data.astype('float32') #.reshape((self.call(yselection).data.shape[0],1))
        if type(zselection[0])==bool:
            #z = np.zeros((self.call(xselection).data.shape[0],1),dtype='float32')
            z = np.zeros(self.call(xselection).data.shape[0],dtype='float32')
        else:
            z = self.call(zselection).data.data.astype('float32') #.reshape((self.call(zselection).data.shape[0],1))
            choices.remove(zselection[1])
        choices.remove(xselection[1])
        choices.remove(yselection[1])
        #data = np.hstack((x,y,z))
        #choices = self.call_top(selection).get_numeric_variable_names()
        if len(choices)>=0:
            if len(choices)>1:
                data = self.call([selection[0],choices[0]]).data.data.reshape((self.call([selection[0],choices[0]]).data.shape[0],1))
                vname = [choices[0]]
                for i in xrange(1,len(choices)):
                    #if choices[i] not in [xselection[0],yselection[0],zselection[0]]:
                    data = np.hstack((data,self.call([selection[0],choices[i]]).data.data.reshape((self.call([selection[0],choices[i]]).data.shape[0],1))))
                    vname.append(choices[i])
            else:
                data = self.call([selection[0],choices[0]]).data.data
                vname = choices[0]
            #data = np.hstack((data,self.call(selection).data.data.reshape(self.call(selection).data.data.shape[0],1)))
            #data = self.call(selection).data.data
            null = self.call_top(selection).null
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_point_object(x,y,z,null,name,vname,data)
            self.last_object = name
            
    def related_convert_mesh2data(self,selection,name,pickle=False,pickle_value=1):
        if pickle:
            data = self.call(selection).data.data.copy().flatten()[::pickle_value]
        else:
            data = self.call(selection).data.data.copy().flatten()
        null = self.call_top(selection).null
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.add_data_object(null,name,selection[1],data,data.dtype,self.call(selection).vtype)
        self.last_object = name
        
    def related_full_convert_mesh2data(self,selection,name,pickle=False,pickle_value=1):
        if len(self.call_top(selection).get_variable_names())==1:
            self.related_convert_mesh2data([selection[0],self.call_top(selection).get_variable_names()[0]],name,pickle,pickle_value)
        else:
            variables = self.call_top(selection).get_variable_names()
            vname = [variables[0]]
            dtype = [self.call([selection[0],variables[0]]).dtype]
            if pickle:
                data  = self.call([selection[0],self.call_top(selection).get_variable_names()[0]]).data.data.copy().flatten()[::pickle_value]
                data = data.reshape((data.shape[0],1))
            else:
                data  = self.call([selection[0],self.call_top(selection).get_variable_names()[0]]).data.data.copy().flatten()
                data = data.reshape((data.shape[0],1))
            vtype = [self.call([selection[0],variables[0]]).vtype]
            null = self.call_top(selection).null
            for i in xrange(1,len(variables)):
                vname.append(variables[i])
                dtype.append(self.call([selection[0],variables[i]]).dtype)
                vtype.append(self.call([selection[0],variables[i]]).vtype)
                if pickle:
                    data2 = self.call([selection[0],variables[i]]).data.data.copy().flatten()[::pickle_value]
                    data2 = data2.reshape((data.shape[0],1))
                    data = np.hstack((data,data2))
                else:
                    data2 = self.call([selection[0],variables[i]]).data.data.copy().flatten()
                    data2 = data2.reshape((data.shape[0],1))
                    data = np.hstack((data,data2))
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            self.add_data_object(null,name,vname,data,dtype,vtype)
            self.last_object = name
            
    def related_classify_data(self,x,y,number,cmap,bin_array):
        shapes_list = pympl.create_bidistribution_class(x,y,bin_array,cmap)
        coords = np.hstack((x.reshape((x.shape[0],1)),y.reshape((y.shape[0],1))))
        for i in xrange(len(shapes_list)):
            res = pympl.points_inside_polygon(coords,shapes_list[i])
            ind = np.where(res==True)
            bin_array[ind] = number
            
        return bin_array
        
        
    def related_cluster_analysis_data(self,x,y,ctype,clusters,eps,samples,bin_array):
        if ctype == 'KMeans': bin_array = cmlt.Kmeans_cluster_analysis(x,y,clusters)
        elif ctype == 'Mean Shift': bin_array = cmlt.mean_shift_cluster_analysis(x,y,eps,samples)
        elif ctype == 'DBSCAN': bin_array = cmlt.DBSCAN_cluster_analysis(x,y,eps,samples)            
        return bin_array     
            
    def manipulate_join(self,selection,axis,selection2,name):
        '''
        manipulate_join(...)
            manipulate_join(selection,axis,selection2,name)
            
            Joins two objects togheter considering the chosen axis.
            
        Parameters
        ----------
        selection : list with two strings.
            List with string name of the object and the variable to be used.
            
        operation: string
            String X,Y,Z.
            
        selection2: float
            List with string name of the object and the variable to be used.
            
        name : bool or string
            if bool the target variable will be changed. If string a new
            variable will be created with the name the string has. ONLY NEW
            VARIABLE IS WORKING IN THIS VERSION.
            
        Returns
        -------
        out: None
        
        See also
        --------
        None
        '''
        if self.object_type[selection[0]] == 'mesh' and self.object_type[selection2[0]] == 'mesh':
            flag = False
            if axis=='X':
                if self.call_top(selection).blocks[1]==self.call_top(selection2).blocks[1] and self.call_top(selection).blocks[2]==self.call_top(selection2).blocks[2]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.vstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_mesh_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
            elif axis=='Y':
                if self.call_top(selection).blocks[0]==self.call_top(selection2).blocks[0] and self.call_top(selection).blocks[2]==self.call_top(selection2).blocks[2]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.hstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_mesh_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
            elif axis=='Z':
                if self.call_top(selection).blocks[1]==self.call_top(selection2).blocks[1] and self.call_top(selection).blocks[0]==self.call_top(selection2).blocks[0]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.dstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_mesh_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
        elif self.object_type[selection[0]] == 'surf' and self.object_type[selection2[0]] == 'surf':
            flag = False
            if axis=='X':
                if self.call_top(selection).blocks[1]==self.call_top(selection2).blocks[1] and self.call_top(selection).blocks[2]==self.call_top(selection2).blocks[2]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.vstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_surf_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
            elif axis=='Y':
                if self.call_top(selection).blocks[0]==self.call_top(selection2).blocks[0] and self.call_top(selection).blocks[2]==self.call_top(selection2).blocks[2]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.hstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_surf_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
            """
            elif axis=='Z':
                if self.call_top(selection).blocks[1]==self.call_top(selection2).blocks[1] and self.call_top(selection).blocks[0]==self.call_top(selection2).blocks[0]: flag = True
                if flag:
                    if name in self.object_name: 
                        name = self.do_newname_from_name(name)
                    data = np.dstack((self.call(selection).data.data,self.call(selection2).data.data))
                    blocks = data.shape
                    first  = self.object_list[selection[0]].first
                    size   = self.object_list[selection[0]].size
                    if self.object_list[selection[0]].null_flag == False: null = None
                    else: null = self.object_list[selection[0]].null
                    vname = selection[1]
                    self.add_surf_object(blocks,size,first,null,name,vname+'_'+selection2[1],data)
                    self.last_object = name
            """
        elif self.object_type[selection[0]] == 'point' and self.object_type[selection2[0]] == 'point':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            data = np.hstack((self.object_list[selection[0]].variable[selection[1]].data.data,self.object_list[selection2[0]].variable[selection2[1]].data.data))            
            x = np.hstack((self.object_list[selection[0]].x,self.object_list[selection2[0]].x))
            y = np.hstack((self.object_list[selection[0]].y,self.object_list[selection2[0]].y))
            z = np.hstack((self.object_list[selection[0]].z,self.object_list[selection2[0]].z))
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_point_object(x,y,z,null,name,vname+'_'+selection2[1],data)
        elif self.object_type[selection[0]] == 'data' and self.object_type[selection2[0]] == 'data':
            if name in self.object_name: 
                name = self.do_newname_from_name(name)
            data = np.hstack((self.object_list[selection[0]].variable[selection[1]].data.data,self.object_list[selection2[0]].variable[selection2[1]].data.data))            
            if self.object_list[selection[0]].null_flag == False: null = None
            else: null = self.object_list[selection[0]].null
            vname = selection[1]
            self.add_data_object(null,name,vname+'_'+selection2[1],data,self.object_list[selection[0]].variable[selection[1]].dtype,self.object_list[selection[0]].variable[selection[1]].vtype)
            
        
    def add_mesh_object(self,blocks,size,first,null,name,vname,data):
        '''
        add_mesh_object(...)
            add_mesh_object(blocks,size,first,null,name,vname,data)
            
            Creates a mesh collection directly from an object.
            
        Parameters
        ----------
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        size : tuple
            Tuple of floats with size of blocks in (X,Y,Z).
            
        first : tuple
            Tuple of floats with first coordinate of blocks in (X,Y,Z).
            
        null : float
            Null data value. If none null = None (default).
            
        name : string
            string with the name of the object collection.
            
        vname : string or list
            Name of the variable, or , if several, list of names of the variables.
            
        data : numpy array
            If several variables array must have shape length of 4. The last dimension
            is the variable.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if name in self.object_name:
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'mesh'
        if null!=None:
            self.object_list[name]  = mesh_collection(blocks,size,first,null=null,null_flag=True)
        else:
            self.object_list[name]  = mesh_collection(blocks,size,first)
        if len(data.shape)==4:
            columns = data.shape[3]
        else:
            columns = 1
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(vname[i],data[:,:,:,i])
        else: self.object_list[name].add_variable(vname,data)
        self.last_object = name
        return True
        
    def add_triangle_mesh_object(self,x,y,z,triangles,null,name,vname):
        '''
        add_triangle_mesh_object(...)
            add_triangle_mesh_object(x,y,z,triangles,null,name,vname)
            
            Creates a triangle mesh collection directly from an object.
            
        Parameters
        ----------
            
        x: float array
            Float array with x coordinates.
            
        y: float array
            Float array with y coordinates.
            
        x: float array
            Float array with z coordinates.
            
        triangles: int array
            Int array with x size on rows and 3 size on columns. Each row has
            the 3 indexes for a triangle in the x,y,z arrays.
            
        null : float
            Null data value. If none null = None (default).
            
        name : string
            string with the name of the object collection.
            
        vname : string or list
            Name of the variable, or , if several, list of names of the variables.
            
        data : numpy array
            If several variables array must have shape length of 4. The last dimension
            is the variable.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if name in self.object_name:
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'triangle_mesh'
        if null!=None:
            self.object_list[name]  = triangle_mesh_collection(x,y,z,triangles,null=null,null_flag=True)
        else:
            self.object_list[name]  = triangle_mesh_collection(x,y,z,triangles)
        self.object_list[name].add_variable(vname,np.zeros(x.shape[0]))
        self.last_object = name
        return True
        
    def add_point_object(self,x,y,z,null,name,vname,data):
        '''
        add_point_object(...)
            add_point_object(blocks,size,first,null,name,vname,data)
            
            Creates a point collection directly from an object.
            
        Parameters
        ----------
        
        x : numpy array
            X coordinates array.
            
        y : numpy array
            Y coordinates array.
            
        z : numpy array
            Z coordinates array.
            
        null : float
            Null data value. If none null = None (default).
            
        name : string
            string with the name of the object collection.
            
        vname : string or list
            Name of the variable, or , if several, list of names of the variables.
            
        data : numpy array
            Several variable not implemented. Single dimension array.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if name in self.object_name:
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'point'
        if null!=None:
            self.object_list[name]  = point_collection(x,y,z,null=null,null_flag=True)
        else:
            self.object_list[name]  = point_collection(x,y,z)
        #print data.shape,len(data.shape)
        if len(data.shape)>1:
            columns = len(vname)
        else: columns = 1
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(vname[i],data[:,i])
        else: self.object_list[name].add_variable(vname,data)
        self.last_object = name
        return True
        
    def add_surf_object(self,blocks,size,first,null,name,vname,data):
        '''
        add_surf_object(...)
            add_surf_object(blocks,size,first,null,name,vname,data)
            
            Creates a surf collection directly from an object.
            
        Parameters
        ----------
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        size : tuple
            Tuple of floats with size of blocks in (X,Y,Z).
            
        first : tuple
            Tuple of floats with first coordinate of blocks in (X,Y,Z).
            
        null : float
            Null data value. If none null = None (default).
            
        name : string
            string with the name of the object collection.
            
        vname : string or list
            Name of the variable, or , if several, list of names of the variables.
            
        data : numpy array
            If several variables array must have shape length of 4. The last dimension
            is the variable.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if name in self.object_name:
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'surf'
        if null!=None:
            self.object_list[name]  = surf_collection(blocks,size,first,null=null,null_flag=True)
        else:
            self.object_list[name]  = surf_collection(blocks,size,first)
        if len(data.shape)==4:
            columns = data.shape[3]
        else:
            columns = 1
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(vname[i],data[:,:,:,i])
        else: self.object_list[name].add_variable(vname,data)
        self.last_object = name
        return True
        
    def add_data_object(self,null,name,vname,data,dtype,vtype):
        '''
        add_point_object(...)
            add_point_object(blocks,size,first,null,name,vname,data)
            
            Creates a point collection directly from an object.
            
        Parameters
        ----------
        
        x : numpy array
            X coordinates array.
            
        y : numpy array
            Y coordinates array.
            
        z : numpy array
            Z coordinates array.
            
        null : float
            Null data value. If none null = None (default).
            
        name : string
            string with the name of the object collection.
            
        vname : string or list
            Name of the variable, or , if several, list of names of the variables.
            
        data : numpy array
            Several variable not implemented. Single dimension array.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if name in self.object_name:
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'data'
        if null!=None:
            self.object_list[name]  = data_collection(null=null,null_flag=True)
        else:
            self.object_list[name]  = data_collection()
        if type(vname)==list:
            columns = len(vname)
        else:
            columns = 1
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(vname[i],data[:,i],dtype[i],vtype[i])
        else: self.object_list[name].add_variable(vname,data,dtype,vtype)
        self.last_object = name
        return True
            
    def __load_ascii_mesh__(self,path,blocks,dtype,at_least):
        '''
        __load_ascii_mesh__(...)
            __load_ascii_mesh__(path,blocks,dtype,at_least)
            
            Loads mesh data and other information from ASCII file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_npy_mesh__,add_mesh_object_from_file
        '''
        header = cfile.check_header_on_file(path,at_least)
        columns = cfile.check_number_of_columns(path,header)
        data = cfile.load_ascii_grid(path,blocks,header=header,dtype=dtype,columns=columns,swap=False,swap_directory='TMP')
        header_type = cfile.pygeo_determine_header_format(path,at_least)
        if header_type[:4] == 'cmrp' or header_type == 'geoeas': header_info = cfile.pygeo_get_names_from_header(path,columns)
        else: header_info = ['MESH',cfile.pygeo_create_variables_names(columns)]
        keys = []        
        return data,header,header_type,columns,header_info,keys
        
    def __load_npy_mesh__(self,path,dtype):
        '''
        __load_npy_mesh__(...)
            __load_npy_mesh__(path,dtype)
            
            Loads mesh data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        dtype : string
            String with data type (usually float32).
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        data = cfile.load_npy_grid(path,swap=False,swap_directory='TMP')
        if data.dtype!=dtype: data = data.astype(dtype)
        header=0
        header_type = 'npy'
        if len(data.shape) == 3: columns = 1
        elif len(data.shape)==2:
            columns = 1
            data = data.reshape((data.shape[0],data.shape[1],1))
        elif len(data.shape)==1:
            columns = 1
            data = data.reshape((data.shape[0],1,1))
        elif len(data.shape)==4:
            columns = data.shape[3]
        else:
            return ValueError
        header_info = ['MESH',cfile.pygeo_create_variables_names(columns)]
        keys = []
        return data,header,header_type,columns,header_info,keys
        
    def export_mesh_object_to_file(self,selection,opath,filetype='ASCII',fmt='%15.3f'):
        '''
        export_mesh_object_to_file(...)
            export_mesh_object_to_file(selection,path,filetype,fmt)
            
            Saves mesh to file.
            
        Parameters
        ----------
        selection: list
            List of strings of objects to save to file.        
        
        opath : string
            String with file path.
            
        filetype : string
            String with file format (ASCII or NPY). Currently working
            automatically by file extension.
            
        fmt : string
            String with number format.
            
        Returns
        -------
        out: None
            Exports mesh information to file.
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        header = selection[0]+'\n'
        if len(selection)==1:
            header = header + str(len(self.object_list[selection[0]].variable.keys())) + '\n'
            for i in self.object_list[selection[0]].variable.keys():
                header = header + i + '\n'
            cfile.cerena_save_grid_by_dictionary(self.object_list[selection[0]].variable,opath=opath,fmt=fmt,header=header)
        else:
            header = header + '1\n' + selection[1] +'\n'
            cfile.save_grid(self.object_list[selection[0]].variable[selection[1]].data.data,opath=opath,fmt=fmt,header=header)
        
    def add_mesh_object_from_file(self,path,blocks,size,first,null=None,filetype='ASCII',dtype='float32',at_least=3):
        '''
        __add_mesh_object_from_file__(...)
            __add_mesh_object_from_file__(path,blocks,size,first,null,filetype,dtype,at_least)
            
            Loads mesh data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        size : tuple
            Tuple of floats with size of blocks in (X,Y,Z).
            
        first : tuple
            Tuple of floats with first coordinate of blocks in (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        null : float
            Null data value. If none null = None (default).
            
        filetype : string
            string with filetype of the file to be loaded (ASCII or NPY).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if filetype == 'ASCII':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_ascii_mesh__(path,blocks,dtype,at_least)
                # SETTING GENERALISTIC DATABASE
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                self.object_name[header_info[0]] = self.object_counter
                self.object_counter = self.object_counter + 1
                self.object_type[header_info[0]]  = 'mesh'
                if null!=None:
                    self.object_list[header_info[0]]  = mesh_collection(blocks,size,first,null=null,null_flag=True)
                else:
                    self.object_list[header_info[0]]  = mesh_collection(blocks,size,first)
                if columns > 1:
                    for i in xrange(columns):
                        self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,:,:,i])
                else: self.object_list[header_info[0]].add_variable(header_info[1][0],data)
                self.last_object = header_info[0]
                return True
            except ValueError:
                self.__popupmessage__('Not possible to load ascii file. Please check file format.','Error') # NEEDS FUNCTION
                return False
        elif filetype == 'NPY':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_npy_mesh__(path,dtype)
                if len(data.shape)==3: blocks = data.shape
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                self.object_name[header_info[0]] = self.object_counter
                self.object_counter = self.object_counter + 1
                self.object_type[header_info[0]]  = 'mesh'
                if null!=None:
                    self.object_list[header_info[0]]  = mesh_collection(blocks,size,first,null=null,null_flag=True)
                else:
                    self.object_list[header_info[0]]  = mesh_collection(blocks,size,first)
                if columns > 1:
                    for i in xrange(columns):
                        self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,:,:,i])
                else: self.object_list[header_info[0]].add_variable(header_info[1][0],data)
                self.last_object = header_info[0]
                return True
            except ValueError:
                self.__popupmessage__('Not possible to load NPY file. Please check file format.','Error')
                return False
                
    def __load_ascii_surf__(self,path,blocks,dtype,at_least):
        '''
        __load_ascii_mesh__(...)
            __load_ascii_mesh__(path,blocks,dtype,at_least)
            
            Loads mesh data and other information from ASCII file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_npy_mesh__,add_mesh_object_from_file
        '''
        header = cfile.check_header_on_file(path,at_least)
        columns = cfile.check_number_of_columns(path,header)
        data = cfile.load_ascii_grid(path,blocks,header=header,dtype=dtype,columns=columns,swap=False,swap_directory='TMP')
        header_type = cfile.pygeo_determine_header_format(path,at_least)
        if header_type[:4] == 'cmrp' or header_type == 'geoeas': header_info = cfile.pygeo_get_names_from_header(path,columns)
        else: header_info = ['SURF',cfile.pygeo_create_variables_names(columns)]
        keys = []        
        return data,header,header_type,columns,header_info,keys
        
    def __load_npy_surf__(self,path,dtype):
        '''
        __load_npy_mesh__(...)
            __load_npy_mesh__(path,dtype)
            
            Loads mesh data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        dtype : string
            String with data type (usually float32).
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        data = cfile.load_npy_grid(path,swap=False,swap_directory='TMP')
        if data.dtype!=dtype: data = data.astype(dtype)
        header=0
        header_type = 'npy'
        if len(data.shape) == 3: columns = 1
        elif len(data.shape)==2:
            columns = 1
            data = data.reshape((data.shape[0],data.shape[1],1))
        elif len(data.shape)==1:
            columns = 1
            data = data.reshape((data.shape[0],1,1))
        elif len(data.shape)==4:
            columns = data.shape[3]
        else:
            return ValueError
        header_info = ['SURF',cfile.pygeo_create_variables_names(columns)]
        keys = []
        return data,header,header_type,columns,header_info,keys
        
    def export_surf_object_to_file(self,selection,opath,filetype='ASCII',fmt='%15.3f'):
        '''
        export_surf_object_to_file(...)
            export_surf_object_to_file(selection,path,filetype,fmt)
            
            Saves surface to file.
            
        Parameters
        ----------
        selection: list
            List of strings of objects to save to file.        
        
        opath : string
            String with file path.
            
        filetype : string
            String with file format (ASCII or NPY). Currently working
            automatically by file extension.
            
        fmt : string
            String with number format.
            
        Returns
        -------
        out: None
            Exports mesh information to file.
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        header = selection[0]+'\n'
        if len(selection)==1:
            header = header + str(len(self.object_list[selection[0]].variable.keys())) + '\n'
            for i in self.object_list[selection[0]].variable.keys():
                header = header + i + '\n'
            cfile.cerena_save_grid_by_dictionary(self.object_list[selection[0]].variable,opath=opath,fmt=fmt,header=header)
        else:
            header = header + '1\n' + selection[1] +'\n'
            cfile.save_grid(self.object_list[selection[0]].variable[selection[1]].data.data,opath=opath,fmt=fmt,header=header)
                
    def add_surf_object_from_file(self,path,blocks,size,first,null=None,filetype='ASCII',dtype='float32',at_least=3):
        '''
        __add_surf_object_from_file__(...)
            __add_surf_object_from_file__(path,blocks,size,first,null,filetype,dtype,at_least)
            
            Loads surf data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        blocks : tuple
            Tuple of ints with number of blocks in (X,Y,Z).
            
        size : tuple
            Tuple of floats with size of blocks in (X,Y,Z).
            
        first : tuple
            Tuple of floats with first coordinate of blocks in (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        null : float
            Null data value. If none null = None (default).
            
        filetype : string
            string with filetype of the file to be loaded (ASCII or NPY).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        __load_ascii_mesh__, __load_npy_mesh__
        '''
        if filetype == 'ASCII':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_ascii_surf__(path,blocks,dtype,at_least)
                # SETTING GENERALISTIC DATABASE
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                self.object_name[header_info[0]] = self.object_counter
                self.object_counter = self.object_counter + 1
                self.object_type[header_info[0]]  = 'surf'
                if null!=None:
                    self.object_list[header_info[0]]  = surf_collection(blocks,size,first,null=null,null_flag=True)
                else:
                    self.object_list[header_info[0]]  = surf_collection(blocks,size,first)
                if columns > 1:
                    for i in xrange(columns):
                        self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,:,:,i])
                else: self.object_list[header_info[0]].add_variable(header_info[1][0],data)
                self.last_object = header_info[0]
                return True
            except ValueError:
                self.__popupmessage__('Not possible to load ascii file. Please check file format.','Error') # NEEDS FUNCTION
                return False
        elif filetype == 'NPY':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_npy_surf__(path,dtype)
                #if len(data.shape)==2: blocks = data.shape
                blocks = data.shape
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                self.object_name[header_info[0]] = self.object_counter
                self.object_counter = self.object_counter + 1
                self.object_type[header_info[0]]  = 'surf'
                if null!=None:
                    self.object_list[header_info[0]]  = surf_collection(blocks,size,first,null=null,null_flag=True)
                else:
                    self.object_list[header_info[0]]  = surf_collection(blocks,size,first)
                if columns > 1:
                    for i in xrange(columns):
                        self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,:,:,i])
                else: self.object_list[header_info[0]].add_variable(header_info[1][0],data)
                self.last_object = header_info[0]
                return True
            except ValueError:
                self.__popupmessage__('Not possible to load NPY file. Please check file format.','Error')
                return False
                
    def __load_ascii_point__(self,path,coordinate_columns,dtype,at_least):
        '''
        __load_ascii_point__(...)
            __load_point_mesh__(path,coordinate_columns,dtype,at_least)
            
            Loads point data and other information from ASCII file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        coordinate_columns : tuple
            Tuple of ints with columns of coordinates (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_npy_mesh__,add_mesh_object_from_file, __load_npy_point__,
        add_point_object_from_file
        '''
        header = cfile.check_header_on_file(path,at_least)
        columns = cfile.check_number_of_columns(path,header)
        data = cfile.load_ascii_point(path,coordinate_columns,header,dtype=dtype,swap=False,swap_directory='TMP')
        header_type = cfile.pygeo_determine_header_format(path,at_least)
        if header_type[:4] == 'cmrp' or header_type == 'geoeas': header_info = cfile.pygeo_get_point_names_from_header(path,columns,coordinate_columns)
        else: header_info = ['POINT',cfile.pygeo_create_variables_names(columns)]
        keys = []
        number = coordinate_columns.count(0)        
        columns = columns + number
        return data,header,header_type,columns,header_info,keys
        
    def __load_npy_point__(self,path,dtype):
        '''
        __load_npy_point__(...)
            __load_npy_point__(path,dtype)
            
            Loads point data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        dtype : string
            String with data type (usually float32).
            
        Returns
        -------
        out: 3D numpy array, number of header rows, string with type of header,
             number of columns of data, list with name of object its variables,
             list of other possible instructions (not being used).
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file, __load_ascii_point__,
        add_point_object_from_file
        '''
        data = cfile.load_npy_point(path,swap=False,swap_directory='TMP')
        if data.dtype!=dtype: data = data.astype(dtype)
        header=0
        header_type = 'npy'
        if len(data.shape) == 2:
            columns = data.shape[1]
        else:
            return ValueError
        header_info = ['POINT',cfile.pygeo_create_variables_names(columns)]
        keys = []
        return data,header,header_type,columns,header_info,keys
        
    def export_point_object_to_file(self,selection,opath,filetype='ASCII',fmt='%15.3f'):
        '''
        export_point_object_to_file(...)
            export_point_object_to_file(selection,path,filetype,fmt)
            
            Saves point to file.
            
        Parameters
        ----------
        selection: list
            List of strings of objects to save to file.        
        
        opath : string
            String with file path.
            
        filetype : string
            String with file format (ASCII or NPY). Currently working
            automatically by file extension.
            
        fmt : string
            String with number format.
            
        Returns
        -------
        out: None
            Exports point information to file.
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        header = selection[0]+'\n'+ str(len(self.object_list[selection[0]].variable.keys())+3) +'\nX\nY\nZ\n'
        if len(selection)==1:
            #header = header + str(len(self.object_list[selection[0]].variable.keys())) + '\n'
            for i in self.object_list[selection[0]].variable.keys():
                header = header + i + '\n'
            cfile.cerena_save_point_by_dictionary(self.object_list[selection[0]].variable,
                                                  self.object_list[selection[0]].x
                                                  ,self.object_list[selection[0]].y
                                                  ,self.object_list[selection[0]].z
                                                  ,opath=opath,fmt=fmt,header=header)
        else:
            header = selection[0] + '\n4\nX\nY\nZ\n' + selection[1] +'\n'
            point = np.hstack((self.object_list[selection[0]].x[:,np.newaxis],self.object_list[selection[0]].y[:,np.newaxis],self.object_list[selection[0]].z[:,np.newaxis],self.object_list[selection[0]].variable[selection[1]].data.data[:,np.newaxis]))
            cfile.save_point(point,opath=opath,fmt=fmt,header=header)
                
    def add_point_object_from_file(self,path,coordinate_columns,null=None,filetype='ASCII',dtype='float32',at_least=3):
        '''
        __add_point_object_from_file__(...)
            __add_point_object_from_file__(path,coordinate_columns,null,filetype,dtype,at_least)
            
            Loads mesh data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        coordinate_columns : tuple
            Tuple of ints with columns of coordinates (X,Y,Z).
            
        dtype : string
            String with data type (usually float32).
            
        null : float
            Null data value. If none null = None (default).
            
        filetype : string
            string with filetype of the file to be loaded (ASCII or NPY).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        add_mesh_object_from_file, __load_npy_point__, __load_ascii_mesh__,
        __load_npy_mesh__,  __load_ascii_point__
        '''
        if filetype == 'ASCII':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_ascii_point__(path,coordinate_columns,dtype,at_least)
                # SETTING GENERALISTIC DATABASE
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                if len(header_info[1])!=0: 
                    self.object_name[header_info[0]] = self.object_counter
                    self.object_counter = self.object_counter + 1
                    self.object_type[header_info[0]]  = 'point'
                    if null!=None:
                        self.object_list[header_info[0]]  = point_collection(data[:,0],data[:,1],data[:,2],null=null,null_flag=True)
                    else:
                        self.object_list[header_info[0]]  = point_collection(data[:,0],data[:,1],data[:,2]) 
                    if columns > 1:
                        for i in xrange(columns-3):
                            self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,i+3])
                    else: self.object_list[header_info[0]].add_variable(header_info[1][0],data[:,-1])
                    self.last_object = header_info[0]
                    return True
                else:
                    self.__popupmessage__('Cant have a point object without a variable.','Error')
                    return False
            except ValueError:
                self.__popupmessage__('Not possible to load ascii file. Please check file format.','Error')
                return False
        elif filetype == 'NPY':
            try:
                data,header,header_type,columns,header_info,keys = self.__load_npy_point__(path,dtype)
                # SETTING GENERALISTIC DATABASE
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                self.object_name[header_info[0]] = self.object_counter
                self.object_counter = self.object_counter + 1
                self.object_type[header_info[0]]  = 'point'
                if null!=None:
                    self.object_list[header_info[0]]  = point_collection(data[:,0],data[:,1],data[:,2],null=null,null_flag=True)
                else:
                    self.object_list[header_info[0]]  = point_collection(data[:,0],data[:,1],data[:,2]) 
                if columns > 1:
                    for i in xrange(columns-3):
                        self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,i+3])
                else: self.object_list[header_info[0]].add_variable(header_info[1][0],data[:,-1])
                self.last_object = header_info[0]
                return True
            except ValueError:
                self.__popupmessage__('Not possible to load NPY file. Please check file format.','Error')
                return False
    
    def export_data_object_to_file(self,selection,opath,filetype='ASCII',fmt='%15.3f'):
        '''
        export_data_object_to_file(...)
            export_data_object_to_file(selection,path,filetype,fmt)
            
            Saves non-spatial to file.
            
        Parameters
        ----------
        selection: list
            List of strings of objects to save to file.        
        
        opath : string
            String with file path.
            
        filetype : string
            String with file format (ASCII or NPY). Currently working
            automatically by file extension.
            
        fmt : string
            String with number format.
            
        Returns
        -------
        out: None
            Exports mesh information to file.
        
        See also
        --------
        __load_ascii_mesh__,add_mesh_object_from_file
        '''
        header = selection[0]+'\n'
        if len(selection)==1:
            header = header + str(len(self.object_list[selection[0]].variable.keys())) + '\n'
            for i in self.object_list[selection[0]].variable.keys():
                header = header + i + '\n'
            cfile.cerena_save_flexible_data_by_dictionary(self.object_list[selection[0]].variable,opath=opath,fmt=fmt,header=header)
        else:
            header = header + '1\n' + selection[1] +'\n'
            cfile.save_flexible_data(self.object_list[selection[0]].variable[selection[1]].data,opath=opath,fmt=fmt,header=header)
                
    def add_data_object_from_file(self,path,null=None,filetype='ASCII',dtype='float32',at_least=3):
        '''
        __add_data_object_from_file__(...)
            __add_data_object_from_file__(path,coordinate_columns,null,filetype,dtype,at_least)
            
            Loads mesh data and other information from NPY file.
            
        Parameters
        ----------
        path : string
            String with file path.
            
        dtype : string
            String with data type (usually float32).
            
        null : float
            Null data value. If none null = None (default).
            
        filetype : string
            string with filetype of the file to be loaded (ASCII or NPY).
            
        dtype : string
            String with data type (usually float32).
            
        at_least : int
            Number of without string to consider header information finished.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        add_mesh_object_from_file, __load_npy_point__, __load_ascii_mesh__,
        __load_npy_mesh__,  __load_ascii_point__
        '''
        if filetype == 'ASCII':
            try:
                header = cfile.check_header_on_flexible_file(path,at_least)
                columns = cfile.check_number_of_columns(path,header)
                # load_flexible_data(path,swap=False,swap_directory='TMP',at_least=3)
                data = cfile.load_flexible_data(path,swap=False,swap_directory='TMP')
                header_type = cfile.pygeo_determine_header_in_flexible_format(path,at_least)
                if header_type[:4] == 'cmrp' or header_type == 'geoeas': header_info = cfile.pygeo_get_names_from_header(path,columns)
                else: header_info = ['nsDATA',cfile.pygeo_create_variables_names(columns)]
                keys = []
                # SETTING GENERALISTIC DATABASE
                if header_info[0] in self.object_name: 
                    header_info[0] = self.do_newname_from_name(header_info[0])
                if len(header_info[1])!=0: 
                    self.object_name[header_info[0]] = self.object_counter
                    self.object_counter = self.object_counter + 1
                    self.object_type[header_info[0]]  = 'data'
                    if null!=None:
                        self.object_list[header_info[0]]  = data_collection(null=null,null_flag=True)
                    else:
                        self.object_list[header_info[0]]  = data_collection() 
                    if columns > 1:
                        for i in xrange(columns):
                            if data[0,i].replace('.','').replace('-','').isdigit(): self.object_list[header_info[0]].add_variable(header_info[1][i],np.float_(data[:,i]).astype(dtype),dtype,'continuous')
                            else: self.object_list[header_info[0]].add_variable(header_info[1][i],data[:,i],'string','string')
                    else: 
                        if data[0].replace('.','').replace('-','').isdigit(): self.object_list[header_info[0]].add_variable(header_info[1][0],np.float_(data).astype(dtype),dtype,'continuous')
                        else: self.object_list[header_info[0]].add_variable(header_info[1][0],data,'string','string')
                    self.last_object = header_info[0]
                    return True
                else:
                    self.__popupmessage__('Cant have a data object without a variable.','Error')
                    return False
            except ValueError:
                self.__popupmessage__('Not possible to load ascii file. Please check file format.','Error')
                return False
        elif filetype == 'NPY':
            self.__popupmessage__('Data NPY loading not implemented. Please check file format.','Error')
            return False
                
    def copy(self,selection,end_selection):
        '''
        copy(...)
            copy(selection,end_selection)
            
            Copys an object variable to another object (variable).
            
        Parameters
        ----------
        selection : list
            List of strings that identify object and its variable to be copied.
            
        end_selection: list
            List of string that identify object.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        True if copied, False if not.
        '''
        if self.object_type[selection[0]]=='mesh' and self.object_type[end_selection[0]]=='mesh':
            if not np.any(np.array(self.call_top(selection).blocks)-np.array(self.call_top(end_selection).blocks)):
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.last_object = end_selection[0]
                self.last_variable = newname               
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='point' and self.object_type[end_selection[0]]=='point':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.last_object = end_selection[0]
                self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='point' and self.object_type[end_selection[0]]=='data':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data,self.call(selection).dtype,self.call(selection).vtype)
                self.last_object = end_selection[0]
                self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='data' and self.object_type[end_selection[0]]=='point':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if self.call(selection).vtype!='string':
                    if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                    else: newname = selection[1]
                    self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                    self.last_object = end_selection[0]
                    self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='data' and self.object_type[end_selection[0]]=='data':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data,self.call(selection).dtype,self.call(selection).vtype)
                self.last_object = end_selection[0]
                self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='surf' and self.object_type[end_selection[0]]=='surf':
            if not np.any(np.array(self.call_top(selection).blocks)-np.array(self.call_top(end_selection).blocks)):
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.last_object = end_selection[0]
                self.last_variable = newname                
                return True
            else:
                return False
        else:
            return False
                
        
    def cut(self,selection,end_selection):
        '''
        cut(...)
            cut(selection,end_selection)
            
            Cut an object variable to another object (variable).
            
        Parameters
        ----------
        selection : list
            List of strings that identify object and its variable to be cutted.
            
        end_selection: list
            List of string that identify object.
            
        Returns
        -------
        out: None
            Internal command for object manager class.
        
        See also
        --------
        True if cutted, False if not.
        '''
        if self.object_type[selection[0]]=='mesh' and self.object_type[end_selection[0]]=='mesh':
            if not np.all(np.array(self.call_top(selection).blocks)-np.array(self.call_top(end_selection).blocks)):
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]                
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.remove_selection(selection)                
                self.last_object = end_selection[0]
                self.last_variable = newname                 
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='point' and self.object_type[end_selection[0]]=='point':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]                
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.remove_selection(selection)
                self.last_object = end_selection[0]
                self.last_variable = newname                 
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='point' and self.object_type[end_selection[0]]=='data':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data,self.call(selection).dtype,self.call(selection).vtype)
                self.remove_selection(selection)                
                self.last_object = end_selection[0]
                self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='data' and self.object_type[end_selection[0]]=='point':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if self.call(selection).vtype!='string':
                    if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                    else: newname = selection[1]
                    self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                    self.remove_selection(selection)                    
                    self.last_object = end_selection[0]
                    self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='data' and self.object_type[end_selection[0]]=='data':
            if (self.call_top(selection).x.shape[0]-self.call_top(end_selection).x.shape[0])==0:
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data,self.call(selection).dtype,self.call(selection).vtype)
                self.remove_selection(selection)                
                self.last_object = end_selection[0]
                self.last_variable = newname
                return True
            else:
                return False
        elif self.object_type[selection[0]]=='surf' and self.object_type[end_selection[0]]=='surf':
            if not np.all(np.array(self.call_top(selection).blocks)-np.array(self.call_top(end_selection).blocks)):
                if selection[1] in self.object_list[end_selection[0]].get_variable_names(): newname = self.object_list[end_selection[0]].get_newname_from_name(selection[1])
                else: newname = selection[1]                
                self.object_list[end_selection[0]].add_variable(newname,self.object_list[selection[0]].variable[selection[1]].data.data)
                self.remove_selection(selection)                 
                self.last_object = end_selection[0]
                self.last_variable = newname                 
                return True
            else:
                return False
        else:
            return False
            
    def __popupmessage__(self,info,title,itype='error'):
        '''
        __popupmessage__(...)
            __popupmessage__(info,title='Information',itype='error')
            
            Popup message to inform or to warn about an error occured.
            
        Parameters
        ----------
        info : string
            String that appears in the body of the frame.
        title: string
            String that appears in the title of the frame.
        itype: string
            information type. If error frame has an error icon, if info
            an information icon appears.
            
        Returns
        -------
        out: None
            no return. Just popups a frame.
        
        See also
        --------
        None
        '''
        if itype=='info':
            wx.MessageBox(info, title, wx.OK | wx.ICON_INFORMATION)
        elif itype=='error':
            wx.MessageBox(info, title, wx.OK | wx.ICON_ERROR)
            
    def save_project(self,opath,obj_names):
        if opath[-7:] != '.geoms2': opath = opath + '.geoms2'
        if os.path.exists(opath): shutil.rmtree(opath)
        if not os.path.exists(opath):
            os.mkdir(opath)
            npath = opath+'\\'
            number_of_objects = len(self.object_list.keys())
            name_of_objects = obj_names #self.object_list.keys()
            type_of_objects = [] #self.object_type.keys()
            for i in name_of_objects:
                type_of_objects.append(self.object_type[i])
            name_of_variables = []
            for i in name_of_objects:
                name_of_variables.append(self.object_list[i].variable.keys())
            fid = open(npath+'init.ini','w')
            fid.write('%i\n'%number_of_objects)
            for i in xrange(len(name_of_objects)):
                fid.write(name_of_objects[i].replace(' ','_')+'\t'+type_of_objects[i]+'\t')
                for j in name_of_variables[i]:
                    fid.write(j.replace(' ','_')+';')
                fid.write('end\n')
            fid.write('FILES:\n')
            files = []
            for i in xrange(len(name_of_objects)):
                if type_of_objects[i]!='data':
                    fid.write(str(i)+'.npy\n')
                    files.append(str(i)+'.npy')
                else:
                    fid.write(str(i)+'.prn\n')
                    files.append(str(i)+'.prn')
            fid.close()
            c=0
            for i in name_of_objects:
                if type_of_objects[c] == 'mesh' or type_of_objects[c] == 'surf':
                    l = []
                    if len(self.object_list[i].variable.keys())>1:
                        for j in self.object_list[i].variable.keys():
                            l.append(self.object_list[i].variable[j].data.data[:,:,:,np.newaxis])
                        np.save(npath+files[c],np.concatenate(l,axis=3))
                    else:
                        j = self.object_list[i].variable.keys()[0]
                        np.save(npath+files[c],self.object_list[i].variable[j].data.data)
                elif type_of_objects[c] == 'point':
                    p = np.hstack((self.object_list[i].x[:,np.newaxis],self.object_list[i].y[:,np.newaxis],self.object_list[i].z[:,np.newaxis]))
                    for j in self.object_list[i].variable.keys():
                        p = np.hstack((p,self.object_list[i].variable[j].data.data[:,np.newaxis]))
                    np.save(npath+files[c],p)
                elif type_of_objects[c] == 'data':
                    cfile.cerena_save_flexible_data_by_dictionary(self.object_list[i].variable,opath=npath+files[c],fmt='%15.3f',header=False)
                c = c + 1
            c = 0
            for i in name_of_objects:
                if type_of_objects[c] == 'mesh' or type_of_objects[c] == 'surf':
                    fid = open(npath+files[c]+'.conf','w')
                    fid.write('%i     %i     %i\n'%(self.object_list[i].blocks[0],self.object_list[i].blocks[1],self.object_list[i].blocks[2]))
                    fid.write('%f     %f     %f\n'%(self.object_list[i].size[0],self.object_list[i].size[1],self.object_list[i].size[2]))
                    fid.write('%f     %f     %f\n'%(self.object_list[i].first[0],self.object_list[i].first[1],self.object_list[i].first[2]))
                    fid.write('%f\n'%(self.object_list[i].null))
                    fid.close()
                elif type_of_objects[c] == 'point' or type_of_objects[c] == 'data':
                    fid = open(npath+files[c]+'.conf','w')
                    fid.write('%f\n'%(self.object_list[i].null))
                    fid.close()
                c = c + 1
            return True
            
    def load_project_mesh(self,path,name,variables,blocks,size,first,null):
        data = np.load(path)
        if len(data.shape)==3: blocks = data.shape
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'mesh'
        if null!=None:
            self.object_list[name]  = mesh_collection(blocks,size,first,null=null,null_flag=True)
        else:
            self.object_list[name]  = mesh_collection(blocks,size,first)
        columns = len(variables)
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(variables[i],data[:,:,:,i])
        else: self.object_list[name].add_variable(variables[0],data)
        self.last_object = name
        self.update_objects.append(name)
        return True
        
    def load_project_surf(self,path,name,variables,blocks,size,first,null):
        data = np.load(path)
        if len(data.shape)==3: blocks = data.shape
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'surf'
        if null!=None:
            self.object_list[name]  = surf_collection(blocks,size,first,null=null,null_flag=True)
        else:
            self.object_list[name]  = surf_collection(blocks,size,first)
        columns = len(variables)
        if columns > 1:
            for i in xrange(columns):
                self.object_list[name].add_variable(variables[i],data[:,:,:,i])
        else: self.object_list[name].add_variable(variables[0],data)
        self.last_object = name
        self.update_objects.append(name)
        return True
        
    def load_project_point(self,path,name,variables,null):
        data = np.load(path)
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        self.object_name[name] = self.object_counter
        self.object_counter = self.object_counter + 1
        self.object_type[name]  = 'point'
        if null!=None:
            self.object_list[name]  = point_collection(data[:,0],data[:,1],data[:,2],null=null,null_flag=True)
        else:
            self.object_list[name]  = point_collection(data[:,0],data[:,1],data[:,2])
        columns = data.shape[1]
        if columns > 1:
            for i in xrange(columns-3):
                #print variables,variables[i],data.shape
                self.object_list[name].add_variable(variables[i],data[:,i+3])
        else: self.object_list[name].add_variable(variables[0],data[:,-1])
        self.last_object = name
        self.update_objects.append(name)
        return True
        
    def load_project_data(self,path,name,variables,null):
        data = cfile.load_flexible_data(path,swap=False,swap_directory='TMP')
        if name in self.object_name: 
            name = self.do_newname_from_name(name)
        if len(variables)!=0: 
            self.object_name[name] = self.object_counter
            self.object_counter = self.object_counter + 1
            self.object_type[name]  = 'data'
            if null!=None:
                self.object_list[name]  = data_collection(null=null,null_flag=True)
            else:
                self.object_list[name]  = data_collection()
            columns = len(variables)
            if columns > 1:
                for i in xrange(columns):
                    if data[0,i].replace('.','').replace('-','').isdigit(): self.object_list[name].add_variable(variables[i],np.float_(data[:,i]).astype('float32'),'float32','continuous')
                    else: self.object_list[name].add_variable(variables[i],data[:,i],'string','string')
            else: 
                if data[0].replace('.','').replace('-','').isdigit(): self.object_list[name].add_variable(variables[0],np.float_(data).astype('float32'),'float32','continuous')
                else: self.object_list[name].add_variable(variables[0],data,'string','string')
            self.last_object = name
            self.update_objects.append(name)
            return True
        
        
    def load_project(self,ipath):
        if ipath[-7:] == '.geoms2':
            fid = open(ipath+'\\init.ini','r')
            number_of_objects = int(fid.readline())
            names = []
            variables = []
            types = []
            for i in xrange(number_of_objects):
                appex = fid.readline().split()
                names.append(appex[0])
                types.append(appex[1])
                variables.append(appex[2].split(';')[:-1])
            fid.readline()
            files = []
            for i in xrange(number_of_objects):
                files.append(fid.readline().replace('\n',''))
            fid.close()
            for i in xrange(number_of_objects):
                if types[i]=='mesh':
                    fid = open(ipath+'\\'+files[i]+'.conf','r')
                    blocks = np.int_(fid.readline().split())
                    size   = np.float_(fid.readline().split())
                    first  = np.float_(fid.readline().split())
                    null   = np.float(fid.readline())
                    fid.close()
                    self.load_project_mesh(ipath+'\\'+files[i],names[i],variables[i],blocks,size,first,null)
                elif types[i]=='surf':
                    fid = open(ipath+'\\'+files[i]+'.conf','r')
                    blocks = np.int_(fid.readline().split())
                    size   = np.float_(fid.readline().split())
                    first  = np.float_(fid.readline().split())
                    null   = np.float(fid.readline())
                    fid.close()
                    self.load_project_surf(ipath+'\\'+files[i],names[i],variables[i],blocks,size,first,null)
                elif types[i]=='point':
                    fid = open(ipath+'\\'+files[i]+'.conf','r')
                    null   = np.float(fid.readline())
                    fid.close()
                    self.load_project_point(ipath+'\\'+files[i],names[i],variables[i],null)
                elif types[i]=='data':
                    fid = open(ipath+'\\'+files[i]+'.conf','r')
                    null   = np.float(fid.readline())
                    fid.close()
                    self.load_project_data(ipath+'\\'+files[i],names[i],variables[i],null)
            return True
            
    def build_distribution(self,mode,minimum,maximum,peak,std,number,tmode):
        #print mode,tmode
        if tmode == 'Stochastic': tmode = 'stochastic'
        def linear_transform(this,minimum,maximum):
            return ((this-this.min())*(maximum-minimum))/(this.max()-this.min())+minimum
        
        def make_gaussian_distribution(number,mean,std,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.zeros(number)
                for i in xrange(result.shape[0]):
                    flag = True
                    while flag:
                        appex = np.random.normal(mean,std)
                        if appex >= minimum and appex <= maximum:
                            result[i] = appex
                            flag = False
                return result
            
        def make_lognormal_distribution(number,mean,std,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.zeros(number)
                for i in xrange(result.shape[0]):
                    flag = True
                    while flag:
                        appex = np.random.lognormal(mean,std)
                        if appex >= minimum and appex <= maximum:
                            result[i] = appex
                            flag = False
                return result
            
        def make_triangular_distribution(number,peak,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.triangular(minimum,peak,maximum,size=number)
                return result
                
        def make_uniform_distribution(number,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.uniform(minimum,maximum,size=number)
                return result
                
        def make_beta_distribution(number,alpha,beta,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.beta(alpha,beta,number)
                return linear_transform(result,minimum,maximum)
        
        def make_binomial_distribution(number,n,p,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.binomial(n,p,number)
                return linear_transform(result,minimum,maximum)
            
        def make_chisquare_distribution(number,degrees,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.chisquare(degrees,number)
                return linear_transform(result,minimum,maximum)
                
        def make_exponential_distribution(number,scale,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.exponential(scale,number)
                return linear_transform(result,minimum,maximum)
                
        def make_fisher_distribution(number,dg_num,dg_den,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.f(dg_num,dg_den,number)
                return linear_transform(result,minimum,maximum)
        
        def make_gamma_distribution(number,shape,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.gamma(shape,size=number)
                return linear_transform(result,minimum,maximum)
                
        def make_geometric_distribution(number,p,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.geometric(p,number)
                return linear_transform(result,minimum,maximum)
                
        def make_gumbel_distribution(number,loc,scale,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.gumbel(loc,scale,number)
                return linear_transform(result,minimum,maximum)
                
        def make_hypergeometric_distribution(number,ngood,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                nbad = number-ngood
                result = np.random.hypergeometric(ngood,nbad,ngood+nbad,number)
                #print result
                return linear_transform(result,minimum,maximum)
        
        def make_laplace_distribution(number,peak,decay,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.laplace(peak,decay,number)
                return linear_transform(result,minimum,maximum)
                
        def make_logistic_distribution(number,loc,scale,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.logistic(loc,scale,number)
                return linear_transform(result,minimum,maximum)
                
        def make_pareto_distribution(number,a,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.pareto(a,number)
                return linear_transform(result,minimum,maximum)
                
        def make_poisson_distribution(number,lam,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.poisson(lam,number)
                return linear_transform(result,minimum,maximum)
                
        def make_power_distribution(number,a,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.power(a,number)
                return linear_transform(result,minimum,maximum)
                
        def make_rayleigh_distribution(number,scale,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.rayleigh(scale,number)
                return linear_transform(result,minimum,maximum)
                
        def make_vonmises_distribution(number,mu,kappa,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.vonmises(mu,kappa,number)
                return linear_transform(result,minimum,maximum)
                
        def make_wald_distribution(number,mean,scale,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.wald(mean,scale,number)
                #print result
                return linear_transform(result,minimum,maximum)
                
        def make_weibull_distribution(number,a,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.weibull(a,number)
                return linear_transform(result,minimum,maximum)
                
        def make_zipf_distribution(number,a,minimum,maximum,mode='stochastic'):
            if mode=='stochastic':
                result = np.random.zipf(a,number)
                return linear_transform(result,minimum,maximum)
        
        # ['Gaussian','Exponential','Rayleigh','Triangular','Von Mises','Uniform','Log-normal',
        #  'Weibull','Power','Pareto','Logistic','Laplace','Beta','Chi-square','Fisher','Gamma','Gumbel']
        
        #t = make_zipf_distribution(10000,30,20,770) # DONT KNOW HOW IT WORKS X
        #t = make_weibull_distribution(10000,30,20,770) # GOOD ASSYMETRY TO THE RIGHT 
        #t = make_wald_distribution(10000,30,30,1,20,770) # DONT KNOW HOW IT WORKS X
        #t = make_vonmises_distribution(10000,10,1,20,770) # GOOD CAN DO BIMODAL KAPPA LESS THAN MEAN
        #t = make_rayleigh_distribution(10000,170,20,770) # GOOD
        #t = make_power_distribution(10000,30,20,770) # GOOD
        #t = make_poisson_distribution(10000,10,20,770) # INTEGER X
        #t = make_pareto_distribution(10000,30,20,770) # GOOD
        #t = make_logistic_distribution(10000,10,500,20,770) # GOOD
        #t = make_laplace_distribution(10000,10,10,20,770) # GOOD     
        #t = make_hypergeometric_distribution(10000,100,20,770) # DONT KNOW HOW IT WORKS X
        #t = make_beta_distribution(10000,100,100,20,770) # GOOD
        #t = make_binomial_distribution(10000,10,0.7,20,770) # INTEGER X
        #t = make_chisquare_distribution(10000,10,5,770) # GOOD FLEXIBLE
        #t = make_exponential_distribution(10000,300,300,770) # GOOD
        #t = make_fisher_distribution(10000,300,7000,20,770) # GOOD FLEXIBLE
        #t = make_gamma_distribution(10000,100,100,770)  # GOOD SIMILAR TO NORMAL
        #t = make_geometric_distribution(10000,0.7,20,770)  # INTEGER X   
        #t = make_gumbel_distribution(10000,300,200,20,770) # GOOD
        #t = make_triangular_distribution(10000,300,20,770)
        #t = make_uniform_distribution(10000,20,770)
        # ,minimum,maximum,peak,std,number,tmode
        if mode=='Gaussian':
            t = make_gaussian_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Exponential':
            t = make_exponential_distribution(number,std,minimum,maximum,tmode)
        elif mode=='Rayleigh':
            t = make_rayleigh_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Triangular':
            t = make_triangular_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Von Mises':
            t = make_vonmises_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Uniform':
            t = make_uniform_distribution(number,minimum,maximum,tmode)
        elif mode=='Log-normal':
            t = make_lognormal_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Weibull':
            t = make_weibull_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Power':
            t = make_power_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Pareto':
            t = make_pareto_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Logistic':
            t = make_logistic_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Laplace':
            t = make_laplace_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Beta':
            t = make_beta_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Chi-square':
            t = make_chisquare_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Fisher':
            t = make_fisher_distribution(number,peak,std,minimum,maximum,tmode)
        elif mode=='Gamma':
            t = make_gamma_distribution(number,peak,minimum,maximum,tmode)
        elif mode=='Gumbel':
            t = make_gumbel_distribution(number,peak,std,minimum,maximum,tmode)
        #print t
        return t
    

    