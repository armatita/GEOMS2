# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:28:40 2013

@author: pedro.correia
"""

from __future__ import division
import numpy as np
from mayavi import mlab

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
#from mayavi.sources.api import ArraySource
#from mayavi.modules.api import IsoSurface
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from tvtk.util.ctf import PiecewiseFunction

class MayaviView(HasTraits):
    
    scene = Instance(MlabSceneModel, ())
    
    #scene.disable_render = True

    # The layout of the panel created by traits.
    view = View(Item('scene', editor=SceneEditor(),
                    resizable=True,
                    show_label=False),
                resizable=True)

    def __init__(self):
        HasTraits.__init__(self)
        self.model_flag = False
        
        #self._f = mlab.figure('',bgcolor=(1,1,1))
        # SOME USER PREFERENCES.
        self.use_texture_interpolate = False
        self.texture_interpolator = 'nearest_neighbour'

        self.view_objects_names = {}
        self.view_variables_names = {}
        self.view_objects_attributes = {}
        
        self.scene.anti_aliasing_frames = 0
        
        self.me = None
        
        #print self.scene.mayavi_scene

    def volume_mesh(self,data,cmap='jet'):
        vmin = data.min()
        vmax = data.max()
        self.me = mlab.pipeline.volume(mlab.pipeline.scalar_field(data),vmin=vmin,vmax=vmax)
        mlab.show()
        
    def volume_mesh_and_slices(self,data,cmap='jet',cb=False):
        vmin = data.min()
        vmax = data.max()
        sclr = mlab.pipeline.scalar_field(data)
        self.me = mlab.pipeline.volume(sclr,vmin=vmin,vmax=vmax)
        a = mlab.pipeline.image_plane_widget(sclr,
                                        plane_orientation='x_axes',
                                        slice_index=data.shape[0],colormap=cmap,vmin=vmin,vmax=vmax,figure = self.scene.mayavi_scene
                                    )
        b = mlab.pipeline.image_plane_widget(sclr,
                                    plane_orientation='y_axes',
                                    slice_index=data.shape[1],colormap=cmap,vmin=vmin,vmax=vmax,figure = self.scene.mayavi_scene
                                )
        c = mlab.pipeline.image_plane_widget(sclr,
                                    plane_orientation='z_axes',
                                    slice_index=data.shape[2],colormap=cmap,vmin=vmin,vmax=vmax,figure = self.scene.mayavi_scene
                                )
        a.ipw.margin_size_x = 0.0
        a.ipw.margin_size_y = 0.0
        b.ipw.margin_size_x = 0.0
        b.ipw.margin_size_y = 0.0
        c.ipw.margin_size_x = 0.0
        c.ipw.margin_size_y = 0.0
        a.ipw.texture_interpolate = self.use_texture_interpolate
        b.ipw.texture_interpolate = self.use_texture_interpolate
        c.ipw.texture_interpolate = self.use_texture_interpolate
        a.ipw.reslice_interpolate = self.texture_interpolator
        b.ipw.reslice_interpolate = self.texture_interpolator
        c.ipw.reslice_interpolate = self.texture_interpolator
        mlab.show()
        
def mesh_volume(data,opacity=0,interpolation_type='nearest'):
    vmin = data.min()
    vmax = data.max()
    fig = mlab.figure('',bgcolor=(1,1,1))
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data),vmin=vmin,vmax=vmax)
    vol.volume_property.scalar_opacity_unit_distance = opacity
    vol.volume_property.interpolation_type = interpolation_type
    mlab.show()
    
def mesh_volume_and_slices(data,cmap='jet',opacity=0,interpolation_type='nearest'):
    vmin = data.min()
    vmax = data.max()
    fig = mlab.figure('',bgcolor=(1,1,1))
    sclr = mlab.pipeline.scalar_field(data)
    vol = mlab.pipeline.volume(sclr,vmin=vmin,vmax=vmax)
    vol.volume_property.scalar_opacity_unit_distance = opacity
    vol.volume_property.interpolation_type = interpolation_type
    a = mlab.pipeline.image_plane_widget(sclr,
                                    plane_orientation='x_axes',
                                    slice_index=data.shape[0],colormap=cmap,vmin=vmin,vmax=vmax,figure = fig)
    b = mlab.pipeline.image_plane_widget(sclr,
                                plane_orientation='y_axes',
                                slice_index=data.shape[1],colormap=cmap,vmin=vmin,vmax=vmax,figure = fig)
    c = mlab.pipeline.image_plane_widget(sclr,
                                plane_orientation='z_axes',
                                slice_index=data.shape[2],colormap=cmap,vmin=vmin,vmax=vmax,figure = fig)
    a.ipw.margin_size_x = 0.0
    a.ipw.margin_size_y = 0.0
    b.ipw.margin_size_x = 0.0
    b.ipw.margin_size_y = 0.0
    c.ipw.margin_size_x = 0.0
    c.ipw.margin_size_y = 0.0
    a.ipw.texture_interpolate = False
    b.ipw.texture_interpolate = False
    c.ipw.texture_interpolate = False
    a.ipw.reslice_interpolate = 'nearest_neighbour'
    b.ipw.reslice_interpolate = 'nearest_neighbour'
    c.ipw.reslice_interpolate = 'nearest_neighbour'
    mlab.show()
    
"""
############TESTING FUNCTIONS############################    
fid = open(r'C:\PedroCorreia\PyWorkspace\GSI STUDENT TOOLBOX2\EXAMPLE\Tutorial\reservoir\SY_cut.out','r')
for i in xrange(3): fid.readline()
data = np.loadtxt(fid).reshape((50,50,61),order='F')
fid.close()
data = np.ma.masked_array(data,data<0)
mesh_volume_and_slices(data,'jet',0.0)
#mv = MayaviView()
#control = mv.edit_traits(parent=None,kind='panel').control
#control.SetSize((600,400))
#mv.volume_mesh(data)
#mv.volume_mesh_and_slices(data)
"""