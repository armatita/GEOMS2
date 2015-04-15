# -*- coding: utf-8 -*-
"""
Created on Tue Jun 04 18:19:26 2013

@author: pedro.correia
"""

from __future__ import division
import numpy as np
from scipy import interpolate, signal
import scipy
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import matplotlib.nxutils as pip
import matplotlib.path as path

# HISTOGRAM
"""
'best'            0
  'upper right'     1
  'upper left'      2
  'lower left'      3
  'lower right'     4
  'right'           5
  'center left'     6
  'center right'    7
  'lower center'    8
  'upper center'    9
  'center'  
['best','upper right','upper left','lower left','lower right','right','center left','center right',
 'lower center','upper center','center']
 'bar' | 'barstacked' | 'step' | 'stepfilled'
"""

def xkcd_line(x, y, xlim=None, ylim=None,
              mag=1.0, f1=30, f2=0.05, f3=15):
    """
    Mimic a hand-drawn line from (x, y) data

    Parameters
    ----------
    x, y : array_like
        arrays to be modified
    xlim, ylim : data range
        the assumed plot range for the modification.  If not specified,
        they will be guessed from the  data
    mag : float
        magnitude of distortions
    f1, f2, f3 : int, float, int
        filtering parameters.  f1 gives the size of the window, f2 gives
        the high-frequency cutoff, f3 gives the size of the filter
    
    Returns
    -------
    x, y : ndarrays
        The modified lines
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # get limits for rescaling
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    if xlim[1] == xlim[0]:
        xlim = ylim
        
    if ylim[1] == ylim[0]:
        ylim = xlim

    # scale the data
    x_scaled = (x - xlim[0]) * 1. / (xlim[1] - xlim[0])
    y_scaled = (y - ylim[0]) * 1. / (ylim[1] - ylim[0])

    # compute the total distance along the path
    dx = x_scaled[1:] - x_scaled[:-1]
    dy = y_scaled[1:] - y_scaled[:-1]
    dist_tot = np.sum(np.sqrt(dx * dx + dy * dy))

    # number of interpolated points is proportional to the distance
    Nu = int(200 * dist_tot)
    u = np.arange(-1, Nu + 1) * 1. / (Nu - 1)

    # interpolate curve at sampled points
    k = min(3, len(x) - 1)
    res = interpolate.splprep([x_scaled, y_scaled], s=0, k=k)
    x_int, y_int = interpolate.splev(u, res[0]) 

    # we'll perturb perpendicular to the drawn line
    dx = x_int[2:] - x_int[:-2]
    dy = y_int[2:] - y_int[:-2]
    dist = np.sqrt(dx * dx + dy * dy)

    # create a filtered perturbation
    coeffs = mag * np.random.normal(0, 0.01, len(x_int) - 2)
    b = signal.firwin(f1, f2 * dist_tot/Nu, window=('kaiser', f3))
    response = signal.lfilter(b, 1, coeffs)

    x_int[1:-1] += response * dy / dist
    y_int[1:-1] += response * dx / dist

    # un-scale data
    x_int = x_int[1:-1] * (xlim[1] - xlim[0]) + xlim[0]
    y_int = y_int[1:-1] * (ylim[1] - ylim[0]) + ylim[0]
    
    return x_int, y_int


def XKCDify(ax, mag=0.1,
            f1=3, f2=10, f3=3,
            bgcolor='w',
            xaxis_loc=None,
            yaxis_loc=None,
            xaxis_arrow='+',
            yaxis_arrow='+',
            ax_extend=0.1,
            expand_axes=False):
    """Make axis look hand-drawn

    This adjusts all lines, text, legends, and axes in the figure to look
    like xkcd plots.  Other plot elements are not modified.
    
    Parameters
    ----------
    ax : Axes instance
        the axes to be modified.
    mag : float
        the magnitude of the distortion
    f1, f2, f3 : int, float, int
        filtering parameters.  f1 gives the size of the window, f2 gives
        the high-frequency cutoff, f3 gives the size of the filter
    xaxis_loc, yaxis_log : float
        The locations to draw the x and y axes.  If not specified, they
        will be drawn from the bottom left of the plot
    xaxis_arrow, yaxis_arrow : str
        where to draw arrows on the x/y axes.  Options are '+', '-', '+-', or ''
    ax_extend : float
        How far (fractionally) to extend the drawn axes beyond the original
        axes limits
    expand_axes : bool
        if True, then expand axes to fill the figure (useful if there is only
        a single axes in the figure)
    """
    # Get axes aspect
    ext = ax.get_window_extent().extents
    aspect = (ext[3] - ext[1]) / (ext[2] - ext[0])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - xlim[0]

    xax_lim = (xlim[0] - ax_extend * xspan,
               xlim[1] + ax_extend * xspan)
    yax_lim = (ylim[0] - ax_extend * yspan,
               ylim[1] + ax_extend * yspan)

    if xaxis_loc is None:
        xaxis_loc = ylim[0]

    if yaxis_loc is None:
        yaxis_loc = xlim[0]

    # Draw axes
    xaxis = plt.Line2D([xax_lim[0], xax_lim[1]], [xaxis_loc, xaxis_loc],
                      linestyle='-', color='k')
    yaxis = plt.Line2D([yaxis_loc, yaxis_loc], [yax_lim[0], yax_lim[1]],
                      linestyle='-', color='k')

    # Label axes3, 0.5, 'hello', fontsize=14)
    ax.text(xax_lim[1], xaxis_loc - 0.02 * yspan, ax.get_xlabel(),
            fontsize=14, ha='right', va='top', rotation=12)
    ax.text(yaxis_loc - 0.02 * xspan, yax_lim[1], ax.get_ylabel(),
            fontsize=14, ha='right', va='top', rotation=78)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add title
    ax.text(0.5 * (xax_lim[1] + xax_lim[0]), yax_lim[1],
            ax.get_title(),
            ha='center', va='bottom', fontsize=16)
    ax.set_title('')

    Nlines = len(ax.lines)
    lines = [xaxis, yaxis] + [ax.lines.pop(0) for i in range(Nlines)]

    for line in lines:
        x, y = line.get_data()

        x_int, y_int = xkcd_line(x, y, xlim, ylim,
                                 mag, f1, f2, f3)

        # create foreground and background line
        lw = line.get_linewidth()
        line.set_linewidth(2 * lw)
        line.set_data(x_int, y_int)

        # don't add background line for axes
        if (line is not xaxis) and (line is not yaxis):
            line_bg = plt.Line2D(x_int, y_int, color=bgcolor,
                                linewidth=8 * lw)

            ax.add_line(line_bg)
        ax.add_line(line)

    # Draw arrow-heads at the end of axes lines
    arr1 = 0.03 * np.array([-1, 0, -1])
    arr2 = 0.02 * np.array([-1, 0, 1])

    arr1[::2] += np.random.normal(0, 0.005, 2)
    arr2[::2] += np.random.normal(0, 0.005, 2)

    x, y = xaxis.get_data()
    if '+' in str(xaxis_arrow):
        ax.plot(x[-1] + arr1 * xspan * aspect,
                y[-1] + arr2 * yspan,
                color='k', lw=2)
    if '-' in str(xaxis_arrow):
        ax.plot(x[0] - arr1 * xspan * aspect,
                y[0] - arr2 * yspan,
                color='k', lw=2)

    x, y = yaxis.get_data()
    if '+' in str(yaxis_arrow):
        ax.plot(x[-1] + arr2 * xspan * aspect,
                y[-1] + arr1 * yspan,
                color='k', lw=2)
    if '-' in str(yaxis_arrow):
        ax.plot(x[0] - arr2 * xspan * aspect,
                y[0] - arr1 * yspan,
                color='k', lw=2)

    # Change all the fonts to humor-sans.
    prop = fm.FontProperties(fname='FONTS\\Humor-Sans.ttf', size=16)
    for text in ax.texts:
        text.set_fontproperties(prop)
    
    # modify legend
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        
        for child in leg.get_children():
            if isinstance(child, plt.Line2D):
                x, y = child.get_data()
                child.set_data(xkcd_line(x, y, mag=10, f1=100, f2=0.001))
                child.set_linewidth(2 * child.get_linewidth())
            if isinstance(child, plt.Text):
                child.set_fontproperties(prop)
    
    # Set the axis limits
    ax.set_xlim(xax_lim[0] - 0.1 * xspan,
                xax_lim[1] + 0.1 * xspan)
    ax.set_ylim(yax_lim[0] - 0.1 * yspan,
                yax_lim[1] + 0.1 * yspan)

    # adjust the axes
    ax.set_xticks([])
    ax.set_yticks([])      

    if expand_axes:
        ax.figure.set_facecolor(bgcolor)
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
    
    return ax
    
def rstyle(ax):
    """Styles an axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been carried out (needs to know final tick spacing)
    """
    #set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)
   
    #set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    ax.yaxis.set_minor_locator(plt.MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))
   
    #remove axis border
    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.spines.Spine):
            child.set_alpha(0)
       
    #restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
   
    #remove the minor tick lines    
    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
   
    #only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
   
   
    if ax.legend_ <> None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)
       
       
def rhist(ax, data, **keywords):
    """Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten
    """
   
    defaults = {
                'facecolor' : '0.7',
                'edgecolor' : '0.28',
                'linewidth' : '1',
                'bins' : 100
                }
   
    for k, v in defaults.items():
        if k not in keywords: keywords[k] = v
   
    return ax.hist(data, **keywords)


def rbox(ax, data, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:
   
    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """

    hasColors = 'colors' in keywords
    if hasColors:
        colors = keywords['colors']
        keywords.pop('colors')
       
    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'] )
        keywords.pop('names')
   
    bp = ax.boxplot(data, **keywords)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black', linestyle = 'solid')
    plt.setp(bp['fliers'], color='black', alpha = 0.9, marker= 'o', markersize = 3)
    plt.setp(bp['medians'], color='black')
   
    numBoxes = len(data)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
          boxX.append(box.get_xdata()[j])
          boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)
       
        if hasColors:
            boxPolygon = plt.Polygon(boxCoords, facecolor = colors[i % len(colors)])
        else:
            boxPolygon = plt.Polygon(boxCoords, facecolor = '0.95')
           
        ax.add_patch(boxPolygon)
    return bp
    
class common_densitynet_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmapflag = True
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111,polar=True)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            #if self.rflag:
            #    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            #    axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            #    ptol = (datax.max()-datax.min())*0.03                 
            #    axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111,polar=True)
            if self.lflag:
                if self.cmapflag:
                    ax.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1])
                else:
                    ax.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111,polar=True)
            ax.hexbin(datax,datay,gridsize=self.size,cmap='Blues')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111,polar=True)
            ax.hexbin(datax,datay,gridsize=self.size,cmap='Greys')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Hist2D':
            if self.hist_flag:
                """
                axScatter = subplot(111)
                axScatter.scatter(x, y)
                axScatter.set_aspect(1.)
                
                # create new axes on the right and on the top of the current axes.
                divider = make_axes_locatable(axScatter)
                axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
                axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
                
                # the scatter plot:
                # histograms
                bins = np.arange(-lim, lim + binwidth, binwidth)
                axHistx.hist(x, bins=bins)
                axHisty.hist(y, bins=bins, orientation='horizontal')
                """
                fig = plt.figure(dpi=self.dpi)
                axScatter = fig.add_subplot(111,polar=True)
                #axScatter.set_aspect(1.)
                if self.cmapflag:
                    axScatter.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    axScatter.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                tolx = (datax.max()-datax.min())*0.05
                toly = (datay.max()-datay.min())*0.05
                axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
                axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
                #if self.rflag:
                #    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                #    axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                #    ptol = (datax.max()-datax.min())*0.03                 
                #    axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
                divider = make_axes_locatable(axScatter)            
                axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
                axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
                axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
                axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
                axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
                axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
                plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
                #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
                #axHisty.axis["left"].major_ticklabels.set_visible(False)
                if self.xflag: axScatter.set_xlabel(self.xlabel)
                if self.yflag: axScatter.set_ylabel(self.ylabel)
                if self.tflag: axHistx.set_title(self.tlabel)
                if self.gflag: axScatter.grid()
                if self.lflag: axScatter.legend(loc=self.legend_loc)
                plt.show()
            else:
                fig = plt.figure(dpi=self.dpi)
                ax = fig.add_subplot(111,polar=True)
                if self.lflag:
                    if self.cmapflag:
                        ax.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    else:
                        ax.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                        plt.legend(loc=self.legend_loc)
                else:
                    if self.cmapflag:
                        ax.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1])
                    else:
                        ax.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1])
                if self.rflag:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                    ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                    ptol = (datax.max()-datax.min())*0.03                    
                    ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
                if self.xflag: plt.xlabel(self.xlabel)
                if self.yflag: plt.ylabel(self.ylabel)
                if self.tflag: plt.title(self.tlabel)
                if self.gflag: plt.grid()
                plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111,polar=True)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111,polar=True)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
    
class common_fillplot_feed():
    def __init__(self,bins,color,color2,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.color1b = (color2[0]/255,color2[1]/255,color2[2]/255)
        self.color2b = (color2[0]/255,color2[1]/255,color2[2]/255,color2[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        """
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        """
        self.marker_dictionary  = {'solid':'-','dashed':'--','dash_dot':'-.','dotted':':'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        x = np.arange(datax.shape[0])
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        x = np.arange(datax.shape[0])
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.fill_between(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.fill_between(x,datax,datay,color = self.color1,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                axScatter.fill_between(x,datax,0,color = self.color1b,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    #ax.plot(datax,datay,linewidth=self.size,c=datac,cmap=self.cmap,linestyle=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,0,color = self.color1b,alpha=self.color2b[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,datay,color = self.color1,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    #ax.plot(datax,datay,linewidth=self.size,color = self.color1,linestyle=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,0,color = self.color1b,alpha=self.color2b[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,datay,color = self.color1,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    #ax.plot(datax,datay,linewidth=self.size,c=datac,cmap=self.cmap,linestyle=self.marker,alpha=self.color2[-1])
                    ax.fill_between(x,datax,0,color = self.color1b,alpha=self.color2b[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,datay,color = self.color1,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.fill_between(x,datax,0,color = self.color1b,alpha=self.color2b[-1],label = self.xlabel+' vs '+self.ylabel)
                    ax.fill_between(x,datax,datay,color = self.color1,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    #ax.plot(datax,datay,linewidth=self.size,color = self.color1,linestyle=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
    
class common_stereonet_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111, polar=True)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            #for i in xrange(len(color)):
            #    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
            #    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            #ax.spines['right'].set_color('none')
            #ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111, polar=True)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111, polar=True)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            #ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            #ax.spines['right'].set_color('none')
            #ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
    
class common_lineplot_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        """
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        """
        self.marker_dictionary  = {'solid':'-','dashed':'--','dash_dot':'-.','dotted':':'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.plot(datax[i],datay[i],linewidth=self.size,color = color[i],linestyle=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    ax.plot(datax,datay,linewidth=self.size,c=datac,cmap=self.cmap,linestyle=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.plot(datax,datay,linewidth=self.size,color = self.color1,linestyle=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.plot(datax,datay,linewidth=self.size,c=datac,cmap=self.cmap,linestyle=self.marker,alpha=self.color2[-1])
                else:
                    ax.plot(datax,datay,linewidth=self.size,color = self.color1,linestyle=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
class common_bubblenet_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,size2,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.size2 = size2
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111, polar=True)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111, polar=True)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111, polar=True)
            #dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            #dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            #data = (data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay,datac):
        lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,s=lsize,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, polar=True)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = fig.add_subplot(111, polar=True)
            axes.scatter(datax,datay,s=lsize,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
class common_scatter4_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,size2,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.size2 = size2
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datas,datac):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            #dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            #data = (data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=lsize,c=datas,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datas,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datas,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay,datac):
        lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,s=lsize,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
class common_bubbleplot_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,size2,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.size2 = size2
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #dmin = self.object_list[selection[0]].variable[selection[1]].data.min()
            #dmax = self.object_list[selection[0]].variable[selection[1]].data.max()
            #data = (data-dmin)*(maximum-minimum)/(dmax-dmin)+minimum
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=lsize,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,s=lsize,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay,datac):
        lsize = (datac-datac.min())*(self.size2-self.size)/(datac.max()-datac.min())+self.size
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=lsize,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,s=lsize,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
class common_densityplot_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmapflag = True
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            #if self.rflag:
            #    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            #    axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            #    ptol = (datax.max()-datax.min())*0.03                 
            #    axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    ax.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.hexbin(datax,datay,gridsize=self.size,cmap=self.cmap,alpha=self.color2[-1])
                else:
                    ax.hexbin(datax,datay,gridsize=self.size,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.hexbin(datax,datay,gridsize=self.size,cmap='Blues')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.hexbin(datax,datay,gridsize=self.size,cmap='Greys')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Hist2D':
            if self.hist_flag:
                """
                axScatter = subplot(111)
                axScatter.scatter(x, y)
                axScatter.set_aspect(1.)
                
                # create new axes on the right and on the top of the current axes.
                divider = make_axes_locatable(axScatter)
                axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
                axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
                
                # the scatter plot:
                # histograms
                bins = np.arange(-lim, lim + binwidth, binwidth)
                axHistx.hist(x, bins=bins)
                axHisty.hist(y, bins=bins, orientation='horizontal')
                """
                fig = plt.figure(dpi=self.dpi)
                axScatter = fig.add_subplot(111)
                #axScatter.set_aspect(1.)
                if self.cmapflag:
                    axScatter.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    axScatter.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                tolx = (datax.max()-datax.min())*0.05
                toly = (datay.max()-datay.min())*0.05
                axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
                axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
                #if self.rflag:
                #    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                #    axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                #    ptol = (datax.max()-datax.min())*0.03                 
                #    axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
                divider = make_axes_locatable(axScatter)            
                axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
                axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
                axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
                axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
                axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
                axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
                plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
                #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
                #axHisty.axis["left"].major_ticklabels.set_visible(False)
                if self.xflag: axScatter.set_xlabel(self.xlabel)
                if self.yflag: axScatter.set_ylabel(self.ylabel)
                if self.tflag: axHistx.set_title(self.tlabel)
                if self.gflag: axScatter.grid()
                if self.lflag: axScatter.legend(loc=self.legend_loc)
                plt.show()
            else:
                fig = plt.figure(dpi=self.dpi)
                ax = fig.add_subplot(111)
                if self.lflag:
                    if self.cmapflag:
                        ax.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    else:
                        ax.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                        plt.legend(loc=self.legend_loc)
                else:
                    if self.cmapflag:
                        ax.hist2d(datax,datay,bins=self.size,cmap=self.cmap,alpha=self.color2[-1])
                    else:
                        ax.hist2d(datax,datay,bins=self.size,alpha=self.color2[-1])
                if self.rflag:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                    ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                    ptol = (datax.max()-datax.min())*0.03                    
                    ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
                if self.xflag: plt.xlabel(self.xlabel)
                if self.yflag: plt.ylabel(self.ylabel)
                if self.tflag: plt.title(self.tlabel)
                if self.gflag: plt.grid()
                plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
    
class common_scatterplot_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac=None):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
            
class common_3Dscatterplot_feed():
    def __init__(self,bins,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,
                 legend_flag,legend_location,use_style_flag,style,dpi,gflag,marker,size,hist_flag,reg_flag,cmap_flag=False,cmap='jet'):
        self.bins = bins
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.rflag = reg_flag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.marker_dictionary  = {'Caret down':7,'Caret left':4,'Caret right':5,'Caret up':6,
                                   'Circle':'o','Diamond':'D','Hexagon 1':'h','Hexagon 2':'H','Underscore':'_',
                                   'Octagon':'8','Pentagon':'p','Pixel':',','Plus':'+','Point':'.','Square':'s',
                                   'Star':'*','Thin diamond':'d','Tick down':3,'Tick left':0,'Thick right':1,
                                   'Tick up':2,'Triangle down':'v','Triangle left':'<','Triangle right':'>',
                                   'Triangle up':'^','Slash lines':'|','X':'x','Tri down':'1','Tri left':'3',
                                   'Tri right':'4','Tri up':'2'}
        self.marker = self.marker_dictionary[marker]
        self.size = size
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.hist_flag = hist_flag
        
        self.cmapflag = cmap_flag
        self.cmap = cmap
        
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
        self.dif_markers = ['o','x','h','_','^','|','1','s','8']
                            
    def get_limits(self,data):
        maximum = []
        minimum = []
        for i in xrange(len(data)):
            maximum.append(data[i].max())
            minimum.append(data[i].min())
        return max(maximum),min(minimum)
        
    def do_style_multiple_scatterplot(self,datax,datay,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=150,marker='+',color=self.blue_colors[i],label=label[i])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],marker='+',color=self.blue_colors[i])
            for i in xrange(len(color)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o',label=label[i])
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],color = self.gray_colors[i],marker='o')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag: 
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],label=label[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
                ax.legend(loc = self.legend_loc,frameon=False)
            else:
                for i in xrange(l):
                    ax.scatter(datax[i],datay[i],s=60,color = 'black',marker=self.dif_markers[i],facecolors='none')
                    if self.rflag:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                        ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color='black',linestyle='--')
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.show()
                                    
            
        
    def multiple_scatterplot(self,datax,datay,color,label):
        if self.hist_flag:
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.lflag:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])
                axScatter.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    axScatter.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            xmax,xmin = self.get_limits(datax)
            ymax,ymin = self.get_limits(datay)
            tolx = (xmax-xmin)*0.05
            toly = (ymax-ymin)*0.05
            axScatter.set_xlim(xmin-tolx,xmax+tolx)
            axScatter.set_ylim(ymin-toly,ymax+toly)
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    axScatter.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                   
                    axScatter.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = color,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(xmin-tolx,xmax+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = color,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(ymin-toly,ymax+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1],label = label[i])               
                ax.legend(loc=self.legend_loc)
            else:
                for i in xrange(len(color)):
                    ax.scatter(datax[i],datay[i],s=self.size,color = color[i],marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                for i in xrange(len(color)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(datax[i],datay[i])
                    ax.plot([datax[i].min(),datax[i].max()],[datax[i].min()*slope+intercept,datax[i].max()*slope+intercept],color=color[i])
                    ptol = (datax[i].max()-datax[i].min())*0.03                     
                    ax.text(datax[i].max()+ptol,datax[i].max()*slope+intercept, '%.3f'%r_value, fontsize=10,color=color[i])
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
        
    def common_scatterplot(self,datax,datay,datac):
        if self.hist_flag:
            """
            axScatter = subplot(111)
            axScatter.scatter(x, y)
            axScatter.set_aspect(1.)
            
            # create new axes on the right and on the top of the current axes.
            divider = make_axes_locatable(axScatter)
            axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)
            
            # the scatter plot:
            # histograms
            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')
            """
            fig = plt.figure(dpi=self.dpi)
            axScatter = fig.add_subplot(111)
            #axScatter.set_aspect(1.)
            if self.cmapflag:
                axScatter.scatter(datax,datay,s=self.size,c=datac,cmap=self.cmap,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            else:
                axScatter.scatter(datax,datay,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
            tolx = (datax.max()-datax.min())*0.05
            toly = (datay.max()-datay.min())*0.05
            axScatter.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axScatter.set_ylim(datay.min()-toly,datay.max()+toly)
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axScatter.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                 
                axScatter.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            divider = make_axes_locatable(axScatter)            
            axHistx = divider.append_axes("top", size=1.3, pad=0.15, sharex=axScatter)
            axHisty = divider.append_axes("right", size=1.3, pad=0.15, sharey=axScatter)
            axHistx.hist(datax, bins=self.bins, color = self.color1,alpha=self.color2[-1],normed=True)
            axHistx.set_xlim(datax.min()-tolx,datax.max()+tolx)
            axHisty.hist(datay, bins=self.bins, orientation='horizontal', color = self.color1,alpha=self.color2[-1],normed=True)
            axHisty.set_ylim(datay.min()-toly,datay.max()+toly)            
            plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),visible=False)            
            #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
            #axHisty.axis["left"].major_ticklabels.set_visible(False)
            if self.xflag: axScatter.set_xlabel(self.xlabel)
            if self.yflag: axScatter.set_ylabel(self.ylabel)
            if self.tflag: axHistx.set_title(self.tlabel)
            if self.gflag: axScatter.grid()
            if self.lflag: axScatter.legend(loc=self.legend_loc)
            plt.show()
        else:
            fig = plt.figure(dpi=self.dpi)
            #ax = fig.add_subplot(111)
            ax = fig.gca(projection='3d')
            if self.lflag:
                if self.cmapflag:
                    ax.scatter(datax,datay,datac,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                else:
                    ax.scatter(datax,datay,datac,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1],label = self.xlabel+' vs '+self.ylabel)
                    plt.legend(loc=self.legend_loc)
            else:
                if self.cmapflag:
                    ax.scatter(datax,datay,datac,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1])
                else:
                    ax.scatter(datax,datay,datac,s=self.size,color = self.color1,marker=self.marker,alpha=self.color2[-1])
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
                ptol = (datax.max()-datax.min())*0.03                    
                ax.text(datax.max()+ptol,datax.max()*slope+intercept, '%.3f'%r_value, fontsize=10,color='red')
            if self.xflag: plt.xlabel(self.xlabel)
            if self.yflag: plt.ylabel(self.ylabel)
            if self.tflag: plt.title(self.tlabel)
            if self.gflag: plt.grid()
            plt.show()
            
    def do_style_scatterplot(self,datax,datay):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,marker='+',color='blue')
            slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
            ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')            
            
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='red')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style == 'Basic':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,color = 'black',marker='o',facecolors='none')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'BasicX':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            ax.scatter(datax,datay,s=90,color = 'black',marker='x')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                ax.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([datax.min(), datax.max()])
            ax.set_ylim([datay.min(), datay.max()])
            plt.show()
        elif self.style == 'Sober':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            axes.scatter(datax,datay,color = 'black',marker='o')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            if self.rflag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(datax,datay)
                axes.plot([datax.min(),datax.max()],[datax.min()*slope+intercept,datax.max()*slope+intercept],color='black',linestyle='--')
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([datax.min(), datax.max()])
            axes.set_ylim([datay.min(),datay.max()])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
            
class common_boxplot_feed():
    def __init__(self,axis,per_flag,percentile_marker,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,legend_flag,legend_location,
                 use_style_flag,style,dpi,gflag,linewidth):
        if axis == 'Vertical': self.axis = 'horizontal'
        else: self.axis = 'vertical'
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.per_flag = per_flag
        self.percentile_marker = percentile_marker
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.my_dictionary_barstyle = {'Bar':'bar','Step':'step','Step filled':'stepfilled'}
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        self.linewidth = linewidth
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]
    
    def do_style_multiple_boxplot(self,data,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(dpi=self.dpi, facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            for i in xrange(l):
                minimum = data[i].min()
                per25 = np.percentile(data[i],25)
                per50 = np.percentile(data[i],50)
                mean = data[i].mean()
                per75 = np.percentile(data[i],75)
                maximum = data[i].max()
                ax.plot([i+1-0.4,i+1+0.4],[per25,per25],color='blue',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[per75,per75],color='blue',linewidth=self.linewidth)
                ax.plot([i+1+0.4,i+1+0.4],[per25,per75],color='blue',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1-0.4],[per25,per75],color='blue',linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.4],[per50,per50],color='red',linewidth=self.linewidth)
                
                ax.plot([i+1,i+1],[minimum,per25],color='blue',linewidth=self.linewidth)
                ax.plot([i+1,i+1],[per75,maximum],color='blue',linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.4],[minimum,minimum],color='blue',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[maximum,maximum],color='blue',linewidth=self.linewidth)
                if self.per_flag:
                    mark = np.percentile(data[i],self.percentile_marker)
                    ax.plot([i+1-0.2,i+1+0.2],[mark,mark],color='grey',linewidth=self.linewidth)
                ax.scatter([i+1],[mean],s=400,color=self.blue_colors[i],marker='x',label=label[i])
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            if self.lflag: ax.legend(loc = self.legend_loc,frameon=False)
            plt.show()
        elif self.style == 'SGeMS alike':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            ax = fig.add_subplot(111)
            l = len(color)
            for i in xrange(l):
                minimum = data[i].min()
                per25 = np.percentile(data[i],25)
                per50 = np.percentile(data[i],50)
                mean = data[i].mean()
                per75 = np.percentile(data[i],75)
                maximum = data[i].max()
                tol = (data[i].max()-data[i].min())*0.05
                ax.plot([i+1-0.4,i+1+0.4],[per25,per25],color='grey',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[per75,per75],color='grey',linewidth=self.linewidth)
                ax.plot([i+1+0.4,i+1+0.4],[per25,per75],color='grey',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1-0.4],[per25,per75],color='grey',linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.4],[per50,per50],color='black',linewidth=self.linewidth)
                
                ax.plot([i+1,i+1],[minimum,per25],color='grey',linewidth=self.linewidth)
                ax.plot([i+1,i+1],[per75,maximum],color='grey',linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.3],[minimum,minimum],color='grey',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.3],[maximum,maximum],color='grey',linewidth=self.linewidth)
                if self.per_flag:
                    mark = np.percentile(data,self.percentile_marker)
                    ax.plot([i+1-0.2,i+1+0.2],[mark,mark],color='grey',linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[mean,mean],color=self.gray_colors[i],linewidth=self.linewidth,linestyle='--',label=label[i])
            ax.grid()
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            if self.lflag: ax.legend(loc = self.legend_loc,frameon=False)
            plt.show()
        elif self.style=='Sober':
            # THANKS TO PAUL H in: 
            # http://stackoverflow.com/questions/14349055/making-matplotlib-graphs-look-like-r-by-default
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            l = len(color)
            for i in xrange(l):
                minimum = data[i].min()
                per25 = np.percentile(data[i],25)
                per50 = np.percentile(data[i],50)
                mean = data[i].mean()
                per75 = np.percentile(data[i],75)
                maximum = data[i].max()
                #tol = (data.max()-data.min())*0.05
                
                axes.plot([i+1-0.4,i+1+0.4],[per25,per25],color='black',linewidth=self.linewidth)
                axes.plot([i+1-0.4,i+1+0.4],[per75,per75],color='black',linewidth=self.linewidth)
                axes.plot([i+1+0.4,i+1+0.4],[per25,per75],color='black',linewidth=self.linewidth)
                axes.plot([i+1-0.4,i+1-0.4],[per25,per75],color='black',linewidth=self.linewidth)
                
                axes.plot([i+1-0.4,i+1+0.4],[per50,per50],color='black',linewidth=self.linewidth)
                
                axes.plot([i+1,i+1],[minimum,per25],color='black',linewidth=self.linewidth)
                axes.plot([i+1,i+1],[per75,maximum],color='black',linewidth=self.linewidth)
                
                axes.plot([i+1-0.2,i+1+0.2],[minimum,minimum],color='black',linewidth=self.linewidth)
                axes.plot([i+1-0.2,i+1+0.2],[maximum,maximum],color='black',linewidth=self.linewidth)
                if self.per_flag:
                    mark = np.percentile(data,self.percentile_marker)
                    axes.plot([i+1-0.1,i+1+0.1],[mark,mark],color='black',linewidth=self.linewidth)
                axes.plot([i+1-0.2,i+1+0.2],[mean,mean],color='black',linewidth=self.linewidth)

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
                            
    def do_style_boxplot(self,data):
        if self.style == 'GeoMS':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            ax = fig.add_subplot(111)
            minimum = data.min()
            per25 = np.percentile(data,25)
            per50 = np.percentile(data,50)
            mean = data.mean()
            per75 = np.percentile(data,75)
            maximum = data.max()
            tol = (data.max()-data.min())*0.05
            ax.plot([1-0.5,1+0.5],[per25,per25],color='blue',linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[per75,per75],color='blue',linewidth=self.linewidth)
            ax.plot([1+0.5,1+0.5],[per25,per75],color='blue',linewidth=self.linewidth)
            ax.plot([1-0.5,1-0.5],[per25,per75],color='blue',linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[per50,per50],color='red',linewidth=self.linewidth)
            
            ax.plot([1,1],[minimum,per25],color='blue',linewidth=self.linewidth)
            ax.plot([1,1],[per75,maximum],color='blue',linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[minimum,minimum],color='blue',linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[maximum,maximum],color='blue',linewidth=self.linewidth)
            if self.per_flag:
                mark = np.percentile(data,self.percentile_marker)
                ax.plot([1-0.3,1+0.3],[mark,mark],color='grey',linewidth=self.linewidth)
            ax.scatter([1],[mean],s=400,color=(0,1,1),marker='x',label=self.ylabel)
            ax.set_ylim([data.min()-tol, data.max()+tol])
            ax.set_xlim([0.4, 1.6])
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style == 'SGeMS alike':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            ax = fig.add_subplot(111)
            minimum = data.min()
            per25 = np.percentile(data,25)
            per50 = np.percentile(data,50)
            mean = data.mean()
            per75 = np.percentile(data,75)
            maximum = data.max()
            tol = (data.max()-data.min())*0.05
            ax.plot([1-0.5,1+0.5],[per25,per25],color='grey',linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[per75,per75],color='grey',linewidth=self.linewidth)
            ax.plot([1+0.5,1+0.5],[per25,per75],color='grey',linewidth=self.linewidth)
            ax.plot([1-0.5,1-0.5],[per25,per75],color='grey',linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[per50,per50],color='black',linewidth=self.linewidth)
            
            ax.plot([1,1],[minimum,per25],color='grey',linewidth=self.linewidth)
            ax.plot([1,1],[per75,maximum],color='grey',linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[minimum,minimum],color='grey',linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[maximum,maximum],color='grey',linewidth=self.linewidth)
            if self.per_flag:
                mark = np.percentile(data,self.percentile_marker)
                ax.plot([1-0.3,1+0.3],[mark,mark],color='grey',linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[mean,mean],color='black',linewidth=self.linewidth,linestyle='--')
            ax.set_ylim([data.min()-tol, data.max()+tol])
            ax.set_xlim([0.4, 1.6])
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style=='Sober':
            # THANKS TO PAUL H in: 
            # http://stackoverflow.com/questions/14349055/making-matplotlib-graphs-look-like-r-by-default
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            
            minimum = data.min()
            per25 = np.percentile(data,25)
            per50 = np.percentile(data,50)
            mean = data.mean()
            per75 = np.percentile(data,75)
            maximum = data.max()
            tol = (data.max()-data.min())*0.05
            
            axes.plot([1-0.5,1+0.5],[per25,per25],color='black',linewidth=self.linewidth)
            axes.plot([1-0.5,1+0.5],[per75,per75],color='black',linewidth=self.linewidth)
            axes.plot([1+0.5,1+0.5],[per25,per75],color='black',linewidth=self.linewidth)
            axes.plot([1-0.5,1-0.5],[per25,per75],color='black',linewidth=self.linewidth)
            
            axes.plot([1-0.5,1+0.5],[per50,per50],color='black',linewidth=self.linewidth)
            
            axes.plot([1,1],[minimum,per25],color='black',linewidth=self.linewidth)
            axes.plot([1,1],[per75,maximum],color='black',linewidth=self.linewidth)
            
            axes.plot([1-0.3,1+0.3],[minimum,minimum],color='black',linewidth=self.linewidth)
            axes.plot([1-0.3,1+0.3],[maximum,maximum],color='black',linewidth=self.linewidth)
            if self.per_flag:
                mark = np.percentile(data,self.percentile_marker)
                axes.plot([1-0.2,1+0.2],[mark,mark],color='black',linewidth=self.linewidth)
            axes.plot([1-0.3,1+0.3],[mean,mean],color='black',linewidth=self.linewidth)

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
            
    def multiple_boxplot(self,data,color,label):
        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)
        for i in xrange(len(color)):
            minimum = data[i].min()
            per25 = np.percentile(data[i],25)
            per50 = np.percentile(data[i],50)
            mean = data[i].mean()
            per75 = np.percentile(data[i],75)
            maximum = data[i].max()
            if self.axis == 'horizontal':
                #tol = (data.max()-data.min())*0.05
                ax.plot([per25,per25],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([per75,per75],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([per25,per75],[i+1+0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([per25,per75],[i+1-0.4,i+1-0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                
                ax.plot([per50,per50],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([mean,mean],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth,linestyle='--',label=label[i])
                
                ax.plot([minimum,per25],[i+1,i+1],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([per75,maximum],[i+1,i+1],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                
                ax.plot([minimum,minimum],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([maximum,maximum],[i+1-0.4,i+1+0.4],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                if self.per_flag:
                    mark = np.percentile(data[i],self.percentile_marker)
                    ax.plot([mark,mark],[i+1-0.3,i+1+0.3],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                #ax.set_xlim([data.min()-tol, data.max()+tol])
            else:
                #tol = (data.max()-data.min())*0.05
                ax.plot([i+1-0.4,i+1+0.4],[per25,per25],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[per75,per75],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1+0.4,i+1+0.4],[per25,per75],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1-0.4],[per25,per75],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.4],[per50,per50],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[mean,mean],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth,linestyle='--',label=label[i])
                
                ax.plot([i+1,i+1],[minimum,per25],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1,i+1],[per75,maximum],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                
                ax.plot([i+1-0.4,i+1+0.4],[minimum,minimum],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                ax.plot([i+1-0.4,i+1+0.4],[maximum,maximum],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                if self.per_flag:
                    mark = np.percentile(data[i],self.percentile_marker)
                    ax.plot([i+1-0.3,i+1+0.3],[mark,mark],color=color[i],alpha=self.color2[-1],linewidth=self.linewidth)
                #ax.set_ylim([data.min()-tol, data.max()+tol])
        
        if self.tflag: ax.set_title(self.tlabel)
        if self.xflag: ax.set_xlabel(self.xlabel)
        if self.yflag: ax.set_ylabel(self.ylabel)
        if self.gflag: ax.grid()
        if self.lflag: plt.legend(loc = self.legend_loc)
        plt.show()
            
    def common_boxplot(self,data):
        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)
        minimum = data.min()
        per25 = np.percentile(data,25)
        per50 = np.percentile(data,50)
        mean = data.mean()
        per75 = np.percentile(data,75)
        maximum = data.max()
        if self.axis == 'horizontal':
            tol = (data.max()-data.min())*0.05
            ax.plot([per25,per25],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([per75,per75],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([per25,per75],[1+0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([per25,per75],[1-0.5,1-0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            
            ax.plot([per50,per50],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([mean,mean],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth,linestyle='--',label=self.ylabel)
            
            ax.plot([minimum,per25],[1,1],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([per75,maximum],[1,1],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            
            ax.plot([minimum,minimum],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([maximum,maximum],[1-0.5,1+0.5],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            if self.per_flag:
                mark = np.percentile(data,self.percentile_marker)
                ax.plot([mark,mark],[1-0.3,1+0.3],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.set_xlim([data.min()-tol, data.max()+tol])
        else:
            tol = (data.max()-data.min())*0.05
            ax.plot([1-0.5,1+0.5],[per25,per25],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[per75,per75],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1+0.5,1+0.5],[per25,per75],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1-0.5,1-0.5],[per25,per75],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[per50,per50],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[mean,mean],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth,linestyle='--',label=self.ylabel)
            
            ax.plot([1,1],[minimum,per25],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1,1],[per75,maximum],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            
            ax.plot([1-0.5,1+0.5],[minimum,minimum],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.plot([1-0.5,1+0.5],[maximum,maximum],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            if self.per_flag:
                mark = np.percentile(data,self.percentile_marker)
                ax.plot([1-0.3,1+0.3],[mark,mark],color=self.color1,alpha=self.color2[-1],linewidth=self.linewidth)
            ax.set_ylim([data.min()-tol, data.max()+tol])
        
        if self.tflag: ax.set_title(self.tlabel)
        if self.xflag: ax.set_xlabel(self.xlabel)
        if self.yflag: ax.set_ylabel(self.ylabel)
        if self.gflag: ax.grid()
        if self.lflag: plt.legend(loc = self.legend_loc)
        plt.show()            

class common_histogram_feed():
    def __init__(self,bins,minimum,maximum,axis,cumulative,normed,color,xlabel_flag,xlabel,
                 ylabel_flag,ylabel,title_flag,title,legend_flag,legend_location,
                 bar_style,polygon_frequency_flag,use_style_flag,style,dpi,gflag):
        self.bins = bins
        self.minimum = np.float(minimum)
        self.maximum = np.float(maximum)
        if axis == 'Vertical': self.axis = 'horizontal'
        else: self.axis = 'vertical'
        self.color1 = (color[0]/255,color[1]/255,color[2]/255)
        self.color2 = (color[0]/255,color[1]/255,color[2]/255,color[3]/255)
        self.normed = normed
        self.cumulative = cumulative
        self.xflag = xlabel_flag
        self.yflag = ylabel_flag
        self.tflag = title_flag
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.tlabel = title
        self.lflag = legend_flag
        self.gflag = gflag
        self.my_dictionary_legend = {'best':0,'upper right':1,'upper left':2,'lower left':3,
                                    'lower right':4,'right':5,'center left':6,'center right':7,
                                    'lower center':8,'upper center':9,'center':10}
        self.legend_loc = self.my_dictionary_legend[legend_location]
        self.my_dictionary_barstyle = {'Bar':'bar','Step':'step','Step filled':'stepfilled'}
        self.barstyle = self.my_dictionary_barstyle[bar_style]
        self.pflag = polygon_frequency_flag
        self.sflag = use_style_flag
        self.style = style
        self.dpi = dpi
        # BLUE, DODGER BLUE,ROYAL BLUE,MEDIUM BLUE,STEEL BLUE, LIGHT STEEL BLUE,
        # POWDER BLUE, DARK TURQUOISE, CORNFLOWER BLUE
        self.blue_colors = [(0,0,1),(0.11,0.44,1),(0.23,0.41,0.88),(0,0,0.81),
                            (0.27,0.50,0.70),(0.69,0.76,0.87),(0.69,0.87,0.9)
                            ,(0,0.8,0.81),(0.39,0.58,0.92)]
        # GRAY, LIGHT GRAY, IVORY 2, IVORY 3, IVORY 4, SEASHELL 2, SEASHELL 3,
        # SEASHELL 4, LAVENDER
        self.gray_colors = [(0.74,0.74,0.74),(0.82,0.82,0.82),(0.93,0.93,0.87),
                            (0.80,0.80,0.75),(0.54,0.54,0.51),(0.93,0.89,0.87)
                            ,(0.80,0.77,0.74),(0.54,0.52,0.50),(0.90,0.90,0.98)]        
        
    def multiple_histogram(self,data,color,label):
        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)
        if self.lflag:
            h = plt.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,
                     cumulative=self.cumulative,orientation=self.axis,
                     color=color,label=label)
            plt.legend(loc=self.legend_loc)
        else:
            h = plt.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,
                     cumulative=self.cumulative,orientation=self.axis,
                     color=color)
        if self.xflag: plt.xlabel(self.xlabel)
        if self.yflag: plt.ylabel(self.ylabel)
        if self.tflag: plt.title(self.tlabel)
        if self.gflag: plt.grid()
        plt.show()
        
    def do_style_multiple_histogram(self,data,color,label):
        if self.style == 'GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag:
                h = ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=self.blue_colors[:l],label=label)
                leg = ax.legend(loc=self.legend_loc,frameon=False)
            else:
                h = ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=self.blue_colors[:l])
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([h[1].min(), h[1].max()])
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style == 'SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag:
                h = ax.hist(data,color = self.gray_colors[:l],bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,cumulative=self.cumulative,label=label)
                leg = ax.legend(loc=self.legend_loc,frameon=False)
            else:
                h = ax.hist(data,color = self.gray_colors[:l],bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,cumulative=self.cumulative)           
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([h[1].min(), h[1].max()])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style=='Axes3D':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            for zs in xrange(len(color)):
                h = np.histogram(data[zs],bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed)
                half = (h[1][1]-h[1][0])/2
                ax.bar(h[1][:-1]+half, h[0], zs=zs,width=half*1.8, zdir='y', color=color[zs], alpha=self.color2[-1])
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style == 'Overlapped':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            l = len(color)
            if self.lflag:
                for le in xrange(l):
                    ax.hist(data[le],bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=color[le],label=label[le],alpha=self.color2[-1])
                leg = ax.legend(loc=self.legend_loc,frameon=False)
            else:
                for le in xrange(l):
                    ax.hist(data[le],bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=color[le],alpha=self.color2[-1])
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style == 'Stacked':
            # CURRENTLY NOT WORKING
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=color,stacked=True,fill=True,label=label,alpha=self.color2[-1])
                ax.legend(loc=self.legend_loc,frameon=False)
            else:
                ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=color,stacked=True,fill=True,label=label,alpha=self.color2[-1])
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
            
    def common_histogram(self,data):
        fig = plt.figure(dpi=self.dpi)
        ax = fig.add_subplot(111)
        if self.lflag:
            h = plt.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,
                     cumulative=self.cumulative,histtype=self.barstyle,orientation=self.axis,
                     color=self.color1,alpha=self.color2[3],label=self.xlabel)
            plt.legend(loc=self.legend_loc)
        else:
            h = plt.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,
                     cumulative=self.cumulative,histtype=self.barstyle,orientation=self.axis,
                     color=self.color1,alpha=self.color2[3])
        if self.xflag: plt.xlabel(self.xlabel)
        if self.yflag: plt.ylabel(self.ylabel)
        if self.tflag: plt.title(self.tlabel)
        if self.gflag: plt.grid()
        if self.pflag:
            if self.axis == 'vertical': 
                half = (h[1][1]-h[1][0])/2
                plt.scatter(h[1][:-1]+half,h[0],color='red',s=60)
                plt.plot(h[1][:-1]+half,h[0],color='red')
                plt.ylim(ymin=0,ymax=h[0].max()+h[0].max()*0.1)
                plt.xlim(h[1].min(),h[1].max())
            else:
                half = (h[1][1]-h[1][0])/2
                plt.scatter(h[0],h[1][:-1]+half,color='red',s=60)
                plt.plot(h[0],h[1][:-1]+half,color='red')
                plt.xlim(xmin=0,xmax=h[0].max()+h[0].max()*0.1)
                plt.ylim(h[1].min(),h[1].max())
        else:
            if self.axis == 'vertical': 
                plt.ylim(ymin=0,ymax=h[0].max()+h[0].max()*0.1)
                plt.xlim(h[1].min(),h[1].max())
            else:
                plt.xlim(xmin=0,xmax=h[0].max()+h[0].max()*0.1)
                plt.ylim(h[1].min(),h[1].max())
        plt.show()
        
    def do_style_histogram(self,data):
        if self.style=='Sober':
            # THANKS TO PAUL H in: 
            # http://stackoverflow.com/questions/14349055/making-matplotlib-graphs-look-like-r-by-default
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            heights, positions, patches = axes.hist(data,bins=self.bins,range=(self.minimum,self.maximum),
                                                    normed=self.normed,cumulative=self.cumulative, color='white')

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([positions.min(), positions.max()])
            axes.set_ylim([0,heights.max()+heights.max()*0.1])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
        elif self.style=='Entrepreneur':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            if self.lflag:
                rhist(ax, data, bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,cumulative=self.cumulative,label=self.xlabel)
                ax.legend(loc = self.legend_loc)
            else:
                rhist(ax, data, bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,cumulative=self.cumulative)
            rstyle(ax)
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style=='GeoMS':
            fig = plt.figure(facecolor='white')
            ax = fig.add_subplot(111)
            h = ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color='blue')
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax.yaxis.set_major_locator(plt.MaxNLocator(2))
            ax.set_xlim([h[1].min(), h[1].max()])
            ax.set_ylim([0,h[0].max()+h[0].max()*0.1])
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.show()
        elif self.style=='SGeMS':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            h = ax.hist(data,color = 'lightgrey',bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,cumulative=self.cumulative)
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([h[1].min(), h[1].max()])
            ax.set_ylim([0,h[0].max()+h[0].max()*0.1])
            ax.grid()
            ax.tick_params(direction='out', length=6, width=2, colors='black')
            ax.tick_params(axis='x',which='minor',bottom='off')
            plt.show()
        elif self.style=='Silhouette':
            fig = plt.figure(dpi=self.dpi, facecolor="white")
            axes = plt.subplot(111)
            #heights, positions, patches = axes.hist(data,bins=self.bins,range=(self.minimum,self.maximum),
            #                                        normed=self.normed,cumulative=self.cumulative,histtype='stepfilled', color='black')
            #f = scipy.interpolate.interp1d(x, y)
            h = np.histogram(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed)
            f = scipy.interpolate.interp1d(h[1][:-1], h[0], kind='cubic')
            xnew = np.linspace(h[1][0],h[1][-2],1000)
            axes.fill_between(xnew,f(xnew),color='black')
            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            
            # was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
            axes.spines['bottom'].set_position(('axes', -0.05))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('axes', -0.05))
            
            axes.set_xlim([h[1].min(), h[1][-2]])
            axes.set_ylim([0,h[0].max()+h[0].max()*0.1])
            axes.xaxis.grid(False)
            axes.yaxis.grid(False)
            fig.tight_layout()
            plt.show()
        elif self.style=='Tidy':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            h = np.histogram(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed)
            half = (h[1][1]-h[1][0])/2
            ax.bar(h[1][:-1]+half, h[0],width=half*1.5, color=self.color1, alpha=self.color2[-1])
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()
        elif self.style=='Wand':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            h = np.histogram(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed)
            half = (h[1][1]-h[1][0])/2
            for i in xrange(h[0].shape[0]):
                ax.plot([h[1][i]+half,h[1][i]+half],[0,h[0][i]],'--',color='black')
                ax.scatter([h[1][i]+half],[h[0][i]],s=120,color=self.color1)
            if self.tflag: ax.set_title(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([h[1].min(), h[1][-2]])
            ax.set_ylim([0,h[0].max()+h[0].max()*0.1])
            plt.show()
        elif self.style=='Gaussian cousin':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            h = ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=self.color1,alpha=self.color2[-1])
            x = np.linspace(h[1][0],h[1][-1],1000)
            y = mlab.normpdf(x,data.mean(),data.std())
            ax.plot(x,y*h[0].max()/y.max(),'--',color='red')
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            ax.set_xlim([h[1].min(), h[1][-2]])
            ax.set_ylim([0,h[0].max()+h[0].max()*0.1])
            plt.show()
        elif self.style=='Informative':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            h = ax.hist(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed,color=self.color1,alpha=self.color2[-1])
            q1 = np.percentile(data,25)
            q2 = np.percentile(data,50)
            q3 = np.percentile(data,75)
            m  = data.mean()
            ax.plot([q1,q1],[0,h[0].max()+h[0].max()*0.1],'--',color='purple',linewidth=3,label='Percentile 25')
            ax.plot([q2,q2],[0,h[0].max()+h[0].max()*0.1],'--',color='red',linewidth=3,label='Percentile 50')
            ax.plot([q3,q3],[0,h[0].max()+h[0].max()*0.1],'--',color='purple',linewidth=3,label='Percentile 75')
            ax.plot([m,m],[0,h[0].max()+h[0].max()*0.1],'--',color='black',linewidth=3,label='Mean')
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            if self.lflag: ax.legend(loc=self.legend_loc)
            ax.set_xlim([h[1].min(), h[1][-2]])
            ax.set_ylim([0,h[0].max()+h[0].max()*0.1])
            plt.show()
        elif self.style=='Axes3D':
            fig = plt.figure(dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            h = np.histogram(data,bins=self.bins,range=(self.minimum,self.maximum),normed=self.normed)
            half = (h[1][1]-h[1][0])/2
            ax.bar(h[1][:-1]+half, h[0], zs=0,width=half*1.8, zdir='y', color=self.color1, alpha=self.color2[-1])
            if self.tflag: ax.set_zlabel(self.tlabel)
            if self.xflag: ax.set_xlabel(self.xlabel)
            if self.yflag: ax.set_ylabel(self.ylabel)
            plt.show()

class LineBuilder2:
    def __init__(self, line,ax,color):
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}

    def __call__(self, event):
        #print 'click', event
        if event.inaxes!=self.line.axes: return
        if self.counter == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        if np.abs(event.xdata-self.xs[0])<=2 and np.abs(event.ydata-self.ys[0])<=2 and self.counter != 0:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.scatter(self.xs[0],self.ys[0],s=80,color='red')
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.shape[self.shape_counter] = [self.xs,self.ys]
            self.shape_counter = self.shape_counter + 1
            self.xs = []
            self.ys = []
            self.counter = 0
        else:
            if self.counter != 0:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
            #self.line.set_data(self.xs, self.ys)
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1
            
def change_shapes(shapes):
    new_shapes = {}
    for i in xrange(len(shapes)):
        l = len(shapes[i][1])
        new_shapes[i] = np.zeros((l,2),dtype='int')
        for j in xrange(l):
            new_shapes[i][j,0] = shapes[i][0][j]
            new_shapes[i][j,1] = shapes[i][1][j]
    return new_shapes
    
def change_shapes2(shapes):
    new_shapes = {}
    for i in xrange(len(shapes)):
        l = len(shapes[i][1])
        new_shapes[i] = np.zeros((l,2),dtype='float32')
        for j in xrange(l):
            new_shapes[i][j,0] = shapes[i][0][j]
            new_shapes[i][j,1] = shapes[i][1][j]
    return new_shapes
            
def create_shape_on_surface(data,cmap='jet'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to include shape markers (2 block precision to close the shape)')
    line = ax.imshow(data[:,:,0].T,origin='lower',interpolation='nearest',cmap=cmap)
    ax.set_xlim(0,data[:,:,0].shape[0])
    ax.set_ylim(0,data[:,:,0].shape[1])
    linebuilder = LineBuilder2(line,ax,'white')
    plt.show()
    blocks = data.shape
    new_shapes = change_shapes(linebuilder.shape)
    return new_shapes
    
def points_inside_polygon(coords,shape):
    #return pip.points_inside_poly(coords,shape)
	p = path.Path(shape)
	return p.contains_points(coords)
    
class SurfBuilder:
    def __init__(self, line,ax,color,blocks):
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}
        self.init_flag = True
        self.finit_flag = True
        self.blocks = blocks

    def __call__(self, event):
        #print 'click', event
        if self.finit_flag:
            if event.inaxes!=self.line.axes: return
            if self.counter == 0:
                self.xs.append(0)
                self.ys.append(event.ydata)
            if self.counter != 0:
                #print self.finit_flag
                if self.xs[self.counter-1]-event.xdata<0:
                    self.xs.append(event.xdata)
                    self.ys.append(event.ydata)
            #self.line.set_data(self.xs, self.ys)
            
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            if self.xs[-1]>= self.blocks[1]-3:
                self.finit_flag = False
                self.xs[-1] = self.blocks[1]-1
                self.ax.scatter(self.xs[-1],self.ys[-1],s=80,color='red')
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1
            
        
class SurfBuilder2:
    def __init__(self, line,ax,color,last_line):
        self.last_line = last_line
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}
        self.finit_flag = True

    def __call__(self, event):
        #print 'click', event
        if self.finit_flag:
            if event.inaxes!=self.line.axes: return
            if self.counter == 0:
                #self.xs.append(0)
                self.ys.append(event.ydata)
            if self.counter != 0:
                if self.last_line[self.counter-1]-event.xdata<0:
                    if self.counter < len(self.last_line):
                        #self.xs.append(event.xdata)
                        self.ys.append(event.ydata)
            #self.line.set_data(self.xs, self.ys)
            #print self.xs[:len(self.ys)],self.ys
            self.ax.scatter(self.last_line[:len(self.ys)],self.ys,s=120,color=self.color)
            if self.counter == len(self.last_line):
                self.ax.scatter(self.last_line[:len(self.ys)][-1],self.ys[-1],s=80,color='red')
                self.finit_flag=False
            self.ax.plot(self.last_line[:len(self.ys)],self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1
            
def create_surface(data,axis='X',step=10,cmap='jet'):
    if axis=='X':
        xs = None
        ys = None
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('click to include surface markers, change page on red, X = '+str(0))
        line = ax.imshow(data[0,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
        ax.plot([data.shape[1]-3,data.shape[1]-3],[0,data.shape[2]],linestyle='--',color='white')
        ax.set_xlim(0,data[0,:,:].shape[0])
        ax.set_ylim(0,data[0,:,:].shape[1])
        linebuilder = SurfBuilder(line,ax,'white',data.shape)
        plt.show()
        blocks = data.shape
        #new_shapes = change_shapes(linebuilder.shape)
        xs = linebuilder.xs
        final_ys = xs
        final_xs = []
        for j in xs:
            final_xs.append(0) 
        final_zs = linebuilder.ys
        for i in xrange(step,data.shape[0],step):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('click to include surface markers, change page on red, X = '+str(i))
            line = ax.imshow(data[i,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            for j in xs:
                final_xs.append(i) 
                ax.plot([j,j],[0,data.shape[2]],linestyle='--',color='white')
            ax.set_xlim(0,data[i,:,:].shape[0])
            ax.set_ylim(0,data[i,:,:].shape[1])
            linebuilder = SurfBuilder2(line,ax,'white',xs)
            plt.show()
            blocks = data.shape
            final_ys = final_ys+xs
            final_zs = final_zs+linebuilder.ys
            if len(linebuilder.ys)!=len(xs): return False
        if i!=data.shape[0]-1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('click to include surface markers, X = '+str(data.shape[0]))
            line = ax.imshow(data[-1,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            for j in xs:
                final_xs.append(data.shape[0]-1) 
                ax.plot([j,j],[0,data.shape[2]],linestyle='--',color='white')
            ax.set_xlim(0,data[-1,:,:].shape[0])
            ax.set_ylim(0,data[-1,:,:].shape[1])
            linebuilder = SurfBuilder2(line,ax,'white',xs)
            plt.show()
            blocks = data.shape
            final_ys = final_ys+xs
            final_zs = final_zs+linebuilder.ys
            if len(linebuilder.ys)!=len(xs): return False
        grid_x, grid_y = np.mgrid[0:data.shape[0]:1, 0:data.shape[1]:1]
        points = np.zeros((len(final_xs),2))
        points[:,0] = final_xs[:]
        points[:,1] = final_ys[:]
        from scipy.interpolate import griddata
        surf = griddata(points, final_zs, (grid_x, grid_y), method='cubic')
    elif axis=='Y':
        xs = None
        ys = None
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('click to include surface markers, change page on red, X = '+str(0))
        line = ax.imshow(data[0,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
        ax.plot([data.shape[1]-3,data.shape[1]-3],[0,data.shape[2]],linestyle='--',color='white')
        ax.set_xlim(0,data[0,:,:].shape[0])
        ax.set_ylim(0,data[0,:,:].shape[1])
        linebuilder = SurfBuilder(line,ax,'white',data.shape)
        plt.show()
        blocks = data.shape
        #new_shapes = change_shapes(linebuilder.shape)
        xs = linebuilder.xs
        final_xs = xs
        final_ys = []
        for j in xs:
            final_ys.append(0) 
        final_zs = linebuilder.ys
        for i in xrange(step,data.shape[0],step):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('click to include surface markers, change page on red, X = '+str(i))
            line = ax.imshow(data[i,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            for j in xs:
                final_ys.append(i) 
                ax.plot([j,j],[0,data.shape[2]],linestyle='--',color='white')
            ax.set_xlim(0,data[i,:,:].shape[0])
            ax.set_ylim(0,data[i,:,:].shape[1])
            linebuilder = SurfBuilder2(line,ax,'white',xs)
            plt.show()
            blocks = data.shape
            final_xs = final_xs+xs
            final_zs = final_zs+linebuilder.ys
            if len(linebuilder.ys)!=len(xs): return False
        if i!=data.shape[0]-1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('click to include surface markers, X = '+str(data.shape[0]))
            line = ax.imshow(data[-1,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            for j in xs:
                final_ys.append(data.shape[0]-1) 
                ax.plot([j,j],[0,data.shape[2]],linestyle='--',color='white')
            ax.set_xlim(0,data[-1,:,:].shape[0])
            ax.set_ylim(0,data[-1,:,:].shape[1])
            linebuilder = SurfBuilder2(line,ax,'white',xs)
            plt.show()
            blocks = data.shape
            final_xs = final_xs+xs
            final_zs = final_zs+linebuilder.ys
            if len(linebuilder.ys)!=len(xs): return False
        grid_x, grid_y = np.mgrid[0:data.shape[0]:1, 0:data.shape[1]:1]
        points = np.zeros((len(final_ys),2))
        points[:,0] = final_ys[:]
        points[:,1] = final_xs[:]
        from scipy.interpolate import griddata
        surf = griddata(points, final_zs, (grid_x, grid_y), method='cubic')
    return surf.reshape((surf.shape[0],surf.shape[1],1))
    
class BidistributionBinBuilder:
    def __init__(self, line,ax,color,xran,yran):
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.xran=xran
        self.yran=yran
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}

    def __call__(self, event):
        #print 'click', event
        if event.inaxes!=self.line.axes: return
        if self.counter == 0:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        #dist = np.sqrt((event.xdata-self.xs[0])**2+(event.ydata-self.ys[0])**2)
        #tol = np.sqrt(self.xran**2+self.yran**2)
        if np.abs((event.xdata-self.xs[0]))<=self.xran and np.abs((event.ydata-self.ys[0]))<=self.yran and self.counter != 0:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.scatter(self.xs[0],self.ys[0],s=80,color='red')
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.shape[self.shape_counter] = [self.xs,self.ys]
            self.shape_counter = self.shape_counter + 1
            self.xs = []
            self.ys = []
            self.counter = 0
        else:
            if self.counter != 0:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
            #self.line.set_data(self.xs, self.ys)
            self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
            self.ax.plot(self.xs,self.ys,color=self.color)
            self.line.figure.canvas.draw()
            self.counter = self.counter + 1
    
def create_bidistribution_class(x,y,ba,cmap='jet'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to include markers, (1 percent precision to close the shape)')
    #line = ax.imshow(data[0,:,:].T,origin='lower',interpolation='nearest',cmap=cmap)
    line = ax.scatter(x,y,c=ba,cmap=cmap)
    #ax.plot([data.shape[1]-3,data.shape[1]-3],[0,data.shape[2]],linestyle='--',color='white')
    xran = (x.max()-x.min())*0.05
    yran = (y.max()-y.min())*0.05
    ax.set_xlim(x.min()-xran,x.max()+xran)
    ax.set_ylim(y.min()-yran,y.max()+yran)
    xran = (x.max()-x.min())*0.01
    yran = (y.max()-y.min())*0.01
    linebuilder = BidistributionBinBuilder(line,ax,'black',xran,yran)
    plt.show()
    new_shapes = change_shapes2(linebuilder.shape)
    return new_shapes #points_inside_polygon(coords,shape)
    
class PointBuilder2:
    def __init__(self, line,ax,color):
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}

    def __call__(self, event):
        #print 'click', event
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
        self.ax.scatter(self.xs,self.ys,marker='+',s=90,color='black')
        #self.ax.plot(self.xs,self.ys,color=self.color)
        self.line.figure.canvas.draw()
        self.counter = self.counter + 1
            
def create_samples_on_mesh(data,cmap='jet',sp=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print sp
    ax.set_title('click to include sample markers')
    line = ax.imshow(data[:,:,sp].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
    ax.set_xlim(0,data[:,:,0].shape[0])
    ax.set_ylim(0,data[:,:,0].shape[1])
    linebuilder = PointBuilder2(line,ax,'white')
    plt.show()
    new_shapes = np.zeros((len(linebuilder.xs),2))
    new_shapes[:,0] = linebuilder.xs[:]
    new_shapes[:,1] = linebuilder.ys[:]
    return new_shapes
    
class SectionBuilder2:
    def __init__(self, line,ax,color):
        self.line = line
        #self.xs = list(line.get_xdata())
        #self.ys = list(line.get_ydata())
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.counter = 0
        self.shape_counter = 0
        self.shape = {}

    def __call__(self, event):
        #print 'click', event
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.ax.scatter(self.xs,self.ys,s=120,color=self.color)
        self.ax.scatter(self.xs,self.ys,marker='+',s=90,color='black')
        self.ax.plot(self.xs,self.ys,color=self.color,linestyle='--')
        self.line.figure.canvas.draw()
        self.counter = self.counter + 1
            
def create_section_on_mesh(data,cmap='jet',sp=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to include sample markers')
    line = ax.imshow(data[:,:,sp].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
    ax.set_xlim(0,data[:,:,0].shape[0])
    ax.set_ylim(0,data[:,:,0].shape[1])
    linebuilder = PointBuilder2(line,ax,'white')
    plt.show()
    new_shapes = np.zeros((len(linebuilder.xs),2))
    new_shapes[:,0] = linebuilder.xs[:]
    new_shapes[:,1] = linebuilder.ys[:]
    #print new_shapes
    if len(new_shapes)>=2:
        lines = []
        x = [new_shapes[0,0],new_shapes[0+1,0]]
        y = [new_shapes[0,1],new_shapes[0+1,1]]
        xr = x[1]-x[0]
        yr = y[1]-y[0]
        if np.abs(xr)<1 and np.abs(yr)<1:
            return False
        if np.abs(xr)>=np.abs(yr):
            if xr<0 and yr>0: 
                xo = np.arange(x[1],x[0]).astype('int32')
                m = yr/xr
                b = y[0]-m*x[0]
                yo = np.int_(xo*m+b)
                new = data[xo,yo,:]
                lines.append(new.shape[0])
            else: 
                xo = np.arange(x[0],x[1]).astype('int32')
                m = yr/xr
                b = y[0]-m*x[0]
                yo = np.int_(xo*m+b)
                new = data[xo[::-1],yo,:]
                lines.append(new.shape[0])
        else:
            if yr<0: 
                yo = np.arange(y[1],y[0]).astype('int32')
                m = xr/yr
                b = x[0]-m*y[0]
                xo = np.int_(yo*m+b)
                new = data[xo,yo,:]
                lines.append(new.shape[0])
            else: 
                yo = np.arange(y[0],y[1]).astype('int32')
                m = xr/yr
                b = x[0]-m*y[0]
                xo = np.int_(yo*m+b)
                new = data[xo[::-1],yo,:]
                lines.append(new.shape[0])
        for i in xrange(1,new_shapes.shape[0]-1):
            x = [new_shapes[i,0],new_shapes[i+1,0]]
            y = [new_shapes[i,1],new_shapes[i+1,1]]
            xr = x[1]-x[0]
            yr = y[1]-y[0]
            if np.abs(xr)<1 and np.abs(yr)<1:
                return False
            else:
                if np.abs(xr)>=np.abs(yr):
                    if xr<0 and yr>0: 
                        xo = np.arange(x[1],x[0]).astype('int32')
                        m = yr/xr
                        b = y[0]-m*x[0]
                        yo = np.int_(xo*m+b)
                        new = np.vstack((new,data[xo,yo,:]))
                        lines.append(new.shape[0])
                    else: 
                        xo = np.arange(x[0],x[1]).astype('int32')
                        m = yr/xr
                        b = y[0]-m*x[0]
                        yo = np.int_(xo*m+b)
                        new = np.vstack((new,data[xo[::-1],yo,:]))
                        lines.append(new.shape[0])
                else:
                    if yr<0: 
                        yo = np.arange(y[1],y[0]).astype('int32')
                        m = xr/yr
                        b = x[0]-m*y[0]
                        xo = np.int_(yo*m+b)
                        new = np.vstack((new,data[xo,yo,:]))
                        lines.append(new.shape[0])
                    else: 
                        yo = np.arange(y[0],y[1]).astype('int32')
                        m = xr/yr
                        b = x[0]-m*y[0]
                        xo = np.int_(yo*m+b)
                        new = np.vstack((new,data[xo[::-1],yo,:]))
                        lines.append(new.shape[0])
        plt.imshow(new.T,origin='lower',interpolation='nearest',cmap=cmap)
        for i in lines:
            plt.plot([i,i],[0,new.shape[1]],linestyle='--',color='white')
        plt.xlim(0,new.shape[0])
        plt.ylim(0,new.shape[1])
        plt.show()

def generate_mesh_image_set(odir,axis,step,data,cmap='jet',colorbar=False):
    if axis == 'X':
        for i in xrange(0,data.shape[0],step):
            plt.imshow(data[i,:,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            if colorbar: plt.colorbar()
            plt.savefig(odir+'\\X'+str(i)+'.png')
            plt.close()
    elif axis == 'Y':
        for i in xrange(0,data.shape[1],step):
            plt.imshow(data[:,i,:].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            if colorbar: plt.colorbar()
            plt.savefig(odir+'\\Y'+str(i)+'.png')
            plt.close()
    elif axis == 'Z':
        for i in xrange(0,data.shape[2],step):
            plt.imshow(data[:,:,i].T,origin='lower',interpolation='nearest',cmap=cmap,vmin=data.min(),vmax=data.max())
            if colorbar: plt.colorbar()
            plt.savefig(odir+'\\Z'+str(i)+'.png')
            plt.close()
    return True
    
def give_model_arrays(limits,nugget,sills,model_check,models,model_ranges):
    msize = 1000
    per = 3
    if model_check[0]:
        per = 3
        model_line = np.zeros((msize,2))
        model_line[:,0] = np.linspace(0,limits[0]+per*limits[0],msize)
        if models[0]=='Gaussian':
            sill = sills[0]
            amp = model_ranges[0]
            model_line[:,1] =  (nugget+(sill * (1 - np.e**(-3 * (model_line[:,0]/amp)**2))))
        elif models[0]=='Exponential':
            sill = sills[0]
            amp = model_ranges[0]
            model_line[:,1] = (nugget+(sill*(1-np.e**(-3*model_line[:,0]/amp))))
        elif models[0]=='Spherical':
            sill = sills[0]
            amp = model_ranges[0]
            for i in xrange(model_line.shape[0]):
                if model_line[i,0] < amp: model_line[i,1] = (nugget+(sill * (1.5*model_line[i,0]/amp-0.5*(model_line[i,0]/amp)**3)))
                else: model_line[i,1] = sill
    if model_check[1]:
        if models[1]=='Gaussian':
            sill = sills[1]
            amp = model_ranges[1]
            model_line[:,1] =  (nugget+model_line[:,1]+(sill * (1 - np.e**(-3 * (model_line[:,0]/amp)**2))))
        elif models[1]=='Exponential':
            sill = sills[1]
            amp = model_ranges[1]
            model_line[:,1] = (nugget+model_line[:,1]+(sill*(1-np.e**(-3*model_line[:,0]/amp))))
        elif models[1]=='Spherical':
            sill = sills[1]
            amp = model_ranges[1]
            for i in xrange(model_line.shape[0]):
                if model_line[i,0] < amp: model_line[i,1] = (nugget+model_line[i,1]+(sill * (1.5*model_line[i,0]/amp-0.5*(model_line[i,0]/amp)**3)))
                else: model_line[i,1] = sum([sills[0],sills[1]])
    if model_check[2]:
        if models[2]=='Gaussian':
            sill = sills[2]
            amp = model_ranges[2]
            model_line[:,1] =  (nugget+model_line[:,1]+(sill * (1 - np.e**(-3 * (model_line[:,0]/amp)**2))))
        elif models[2]=='Exponential':
            sill = sills[2]
            amp = model_ranges[2]
            model_line[:,1] = (nugget+model_line[:,1]+(sill*(1-np.e**(-3*model_line[:,0]/amp))))
        elif models[2]=='Spherical':
            sill = sills[2]
            amp = model_ranges[2]
            for i in xrange(model_line.shape[0]):
                if model_line[i,0] < amp: model_line[i,1] = (nugget+model_line[i,1]+(sill * (1.5*model_line[i,0]/amp-0.5*(model_line[i,0]/amp)**3)))
                else: model_line[i,1] = sum(sills)
    return model_line
    
def plot_geoms2_variogram(dist,points,sill,nugget,mflag,model_line):
    #sill = sum(sills)
    plt.plot([0,dist.max()+0.1*dist.max()],[sill,sill],linewidth=3,color='#FF3333')
    plt.scatter(dist,points,s=190,color='green',marker='o')
    plt.scatter(dist,points,s=150,color='#99FF33',marker='o')
    if mflag:
        plt.plot(model_line[:,0],model_line[:,1],linewidth=2,color='#66B2FF')
    xmin = 0
    xmax = dist.max()+0.1*dist.max()
    plt.xlim(xmin,xmax)
    yrange = points.max()-points.min()
    ymin = np.clip(points.min()-yrange*0.05,0,100000000000)
    ymax = max([points.max(),sill])+yrange*0.05
    plt.ylim(0,ymax)
    """    
    plt.annotate(
        '%.2f'%sill, 
        xy = (xmax/2, sill), xytext = (1*xmax/2, -0.05*sill),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    """
    plt.show()

def plot_geoms_variogram(dist,points,sill,nugget,mflag,model_line):
    #sill = sum(sills)
    plt.plot([0,dist.max()+0.1*dist.max()],[sill,sill],linewidth=2,color='red')
    plt.scatter(dist,points,s=90,color='#80FF00',marker='o')
    if mflag:
        plt.plot(model_line[:,0],model_line[:,1],linewidth=2,color='blue')
    plt.xlim(0,dist.max()+0.1*dist.max())
    yrange = points.max()-points.min()
    ymin = np.clip(points.min()-yrange*0.05,0,100000000000)
    ymax = max([points.max(),sill])+yrange*0.05
    plt.ylim(0,ymax)
    plt.show()

def plot_sgems_variogram(dist,points,sill,nugget,mflag,model_line):
    #sill = sum(sills)
    #plt.plot([0,dist.max()+0.1*dist.max()],[sill,sill],linewidth=2,color='red')
    plt.scatter(dist,points,s=90,color='red',marker='*')
    if mflag:
        plt.plot(model_line[:,0],model_line[:,1],linewidth=1,color='black')
    plt.xlim(0,dist.max()+0.1*dist.max())
    yrange = points.max()-points.min()
    ymin = np.clip(points.min()-yrange*0.05,0,100000000000)
    ymax = max([points.max(),sill])+yrange*0.05
    plt.ylim(0,ymax)
    plt.grid()
    plt.show()
    
def plot_basic_variogram(dist,points,sill,nugget,mflag,model_line):
    #sill = sum(sills)
    plt.plot([0,dist.max()+0.1*dist.max()],[sill,sill],linewidth=2,linestyle='--',color='black')
    plt.scatter(dist,points,s=90,edgecolors='black',facecolors='none',marker='o')
    if mflag:
        plt.plot(model_line[:,0],model_line[:,1],linewidth=2,color='black')
    plt.xlim(0,dist.max()+0.1*dist.max())
    yrange = points.max()-points.min()
    ymin = np.clip(points.min()-yrange*0.05,0,100000000000)
    ymax = max([points.max(),sill])+yrange*0.05
    plt.ylim(0,ymax)
    plt.show()
    
def plot_basicx_variogram(dist,points,sill,nugget,mflag,model_line):
    #sill = sum(sills)
    plt.plot([0,dist.max()+0.1*dist.max()],[sill,sill],linewidth=2,linestyle='--',color='black')
    plt.scatter(dist,points,s=90,color='black',marker='x')
    if mflag:
        plt.plot(model_line[:,0],model_line[:,1],linewidth=2,color='black')
    plt.xlim(0,dist.max()+0.1*dist.max())
    yrange = points.max()-points.min()
    ymin = np.clip(points.min()-yrange*0.05,0,100000000000)
    ymax = max([points.max(),sill])+yrange*0.05
    plt.ylim(0,ymax)
    plt.show()

def variogram_plot(dist,points,limits,sill,sills,nugget,model_check,models,model_ranges,style='GeoMS'):
    #sills = np.array(sills)
    #if True in model_check:
    #    sills[~model_check]=0
    #else:
    #    sills = [sills[0],0,0]
    styles = ['GEOMS2','GeoMS','SGeMS','Basic','BasicX','GIS']
    model_line = None
    mflag = False
    if model_check[0]==True:
        model_line = give_model_arrays(limits,nugget,sills,model_check,models,model_ranges)
        mflag = True
    if style=='GEOMS2':
        plot_geoms2_variogram(dist,points,sill,nugget,mflag,model_line)
    elif style=='GeoMS':
        plot_geoms_variogram(dist,points,sill,nugget,mflag,model_line)
    elif style=='SGeMS':
        plot_sgems_variogram(dist,points,sill,nugget,mflag,model_line)
    elif style=='Basic':
        plot_basic_variogram(dist,points,sill,nugget,mflag,model_line)
    elif style=='BasicX':
        plot_basicx_variogram(dist,points,sill,nugget,mflag,model_line)
        
def plot_geoms2_nunesplot(azi,dip,dist,d0,d1,d2):
    ax = plt.subplot(111, polar=True) # ESTOU A CRIAR UM JANELA PARA PLOTS POLARES.
    #plt.scatter(dist,points,s=190,color='green',marker='o')
    #plt.scatter(dist,points,s=150,color='#99FF33',marker='o')
    ax.scatter(azi,dip,s=190,c=dist,cmap='Blues')
    ax.scatter(azi,dip,s=150,c=dist,cmap='Greens')
    ax.scatter(d0[0],d0[1],s=320,marker='o',color='#FF3333')
    ax.scatter(d1[0],d1[1],s=320,marker='+',color='#FF3333')
    ax.scatter(d2[0],d2[1],s=320,marker='+',color='#FF3333')
    #ax.grid()
    plt.show()
    
def plot_geoms_nunesplot(azi,dip,dist,d0,d1,d2):
    ax = plt.subplot(111, polar=True) # ESTOU A CRIAR UM JANELA PARA PLOTS POLARES.
    ax.scatter(azi,dip,s=90,color='blue',marker='o')
    ax.scatter(d0[0],d0[1],s=320,marker='+',color='red')
    ax.scatter(d1[0],d1[1],s=320,marker='x',color='red')
    ax.scatter(d2[0],d2[1],s=320,marker='x',color='red')
    plt.show()
    
def plot_sgems_nunesplot(azi,dip,dist,d0,d1,d2):
    ax = plt.subplot(111, polar=True) # ESTOU A CRIAR UM JANELA PARA PLOTS POLARES.
    ax.scatter(azi,dip,s=90,color='red',marker='*')
    ax.scatter(d0[0],d0[1],s=320,marker='+',color='#8C8C8C')
    ax.scatter(d1[0],d1[1],s=320,marker='x',color='#8C8C8C')
    ax.scatter(d2[0],d2[1],s=320,marker='x',color='#8C8C8C')
    #ax.grid()
    plt.show()
    
def plot_basic_nunesplot(azi,dip,dist,d0,d1,d2):
    ax = plt.subplot(111, polar=True) # ESTOU A CRIAR UM JANELA PARA PLOTS POLARES.
    ax.scatter(azi,dip,s=90,color='black',facecolors='none',marker='o')
    #ax.scatter(d0[0],d0[1],s=320,marker='+',color='#8C8C8C')
    #ax.scatter(d1[0],d1[1],s=320,marker='x',color='#8C8C8C')
    #ax.scatter(d2[0],d2[1],s=320,marker='x',color='#8C8C8C')
    #ax.grid()
    plt.show()
    
def plot_basicx_nunesplot(azi,dip,dist,d0,d1,d2):
    ax = plt.subplot(111, polar=True) # ESTOU A CRIAR UM JANELA PARA PLOTS POLARES.
    ax.scatter(azi,dip,s=90,color='black',marker='x')
    #ax.scatter(d0[0],d0[1],s=320,marker='+',color='#8C8C8C')
    #ax.scatter(d1[0],d1[1],s=320,marker='x',color='#8C8C8C')
    #ax.scatter(d2[0],d2[1],s=320,marker='x',color='#8C8C8C')
    #ax.grid()
    plt.show()
        
def nunesplot(azi,dip,dist,var,d0,d1,d2,style='GEOMS2'):
    # nstyles = ['GEOMS2','GeoMS','SGeMS','Basic','BasicX']
    if style=='GEOMS2':
        plot_geoms2_nunesplot(azi,dip,dist,d0,d1,d2)
    elif style=='GeoMS':
        plot_geoms_nunesplot(azi,dip,dist,d0,d1,d2)
    elif style=='SGeMS':
        plot_sgems_nunesplot(azi,dip,dist,d0,d1,d2)
    elif style=='Basic':
        plot_basic_nunesplot(azi,dip,dist,d0,d1,d2)
    elif style=='BasicX':
        plot_basicx_nunesplot(azi,dip,dist,d0,d1,d2)
        
def simple_histogram(data,bins):
    plt.hist(data,bins=bins)
    plt.show()
        