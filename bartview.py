#!/usr/bin/python
#
# Copyright 2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2015 Frank Ong <frankong@berkeley.edu>


from __future__ import division
import operator
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from functools import partial
import time
import threading
import os.path

class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        self.previous_val = kwargs['valinit']
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = round(val)
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if self.previous_val!=discrete_val:
            self.previous_val = discrete_val
            if not self.eventson: 
                return
            for cid, func in self.observers.iteritems():
                func(discrete_val)


class BartView(object):

    def __init__(self, cflname):
        
        matplotlib.rcParams['toolbar'] = 'None'
        #matplotlib.rcParams['font.size'] = 6

        # Read data
        self.cflname = sys.argv[1]
        self.im = self.readcfl(self.cflname)
        self.im_unsqueeze_shape = np.where( np.array(self.im.shape) > 1 )[0]
        self.im = self.im.squeeze()
        t1 = time.clock()

        # Reorder image
        self.Ndims = len( self.im.shape )
        self.order = np.r_[:self.Ndims]
        self.im_ordered = self.im
        self.order3 = np.array([0,1,1])
        
        # Slice image
        self.slice_num = np.zeros( self.Ndims, dtype = 'int' );
        self.im_shape = self.im_ordered.shape
        self.im_slice = self.im_ordered[ (slice(None), slice(None)) + tuple(self.slice_num[2:]) ]
        
        # Create figure
        self.fig = plt.figure(facecolor='black', figsize=(9,6))
        #self.fig = plt.figure(facecolor='black', figsize=(6,4))
        self.fig.subplots_adjust( left=0.0 , bottom=0.0 , right=1.0 , top=1 - 0.25)
        self.fig.canvas.set_window_title(self.cflname)
        
        # Show image
        self.immax = np.max(abs(self.im))
        self.l = plt.imshow( abs(self.im_slice) , cmap = "gray", vmin=0, vmax=self.immax)
        self.ax = plt.gca()
        self.asp = self.im_ordered.shape[1] / self.im_ordered.shape[0]
        self.aspect = 1
        self.ax.set_aspect( 1 )
        plt.axis('off')
        
        radios = []
        buttons = []
        sliders = []
        

        # Create Radio Buttons for X Y dimensions
        dims = self.im_unsqueeze_shape[ self.order ].astype(str)
        for i in xrange(0,len(dims)):
            dims[i] = "Dim " + dims[i]
        oboxx_ax = plt.axes( [0, 1 - 0.03, 0.1, 0.03], axisbg = "gainsboro" )
        oboxx_ax.set_xticks([]);
        oboxx_ax.set_yticks([]);
        orderx_ax = plt.axes( [0, 1 - 0.18, 0.1, 0.15], axisbg = 'gainsboro' )
        orderx_radio = RadioButtons( orderx_ax, dims, activecolor = 'SteelBlue', active = 0 )
        orderx_ax.text(0.5,1.05,  'Up/Down', horizontalalignment = 'center')
        radios.append( orderx_radio )
        orderx_radio.on_clicked( self.update_orderx )

        oboxy_ax = plt.axes( [0.1, 1 - 0.03, 0.1, 0.03], axisbg = "gainsboro" )
        oboxy_ax.set_xticks([]);
        oboxy_ax.set_yticks([]);
        ordery_ax = plt.axes( [0.1, 1 - 0.18, 0.1, 0.15], axisbg = 'gainsboro' )
        ordery_radio = RadioButtons( ordery_ax, dims, activecolor = 'SteelBlue', active = 1 )
        ordery_ax.text(0.5,1.05,  'Left/Right', horizontalalignment = 'center')
        radios.append( ordery_radio )
        ordery_radio.on_clicked( self.update_ordery )
    

        # Create Radio buttons for mosaic
        self.mosaic_valid = False
        mbox_ax = plt.axes( [0.2, 1 - 0.03, 0.1, 0.03], axisbg = "gainsboro" )
        mbox_ax.set_xticks([]);
        mbox_ax.set_yticks([]);
        mosaic_ax = plt.axes( [0.2, 1 - 0.18, 0.1, 0.15], axisbg = 'gainsboro' )
        mosaic_radio = RadioButtons( mosaic_ax, dims, activecolor = 'SteelBlue', active = 1 )
        mosaic_ax.text(0.5,1.05,  'Mosaic', horizontalalignment = 'center')
        radios.append( mosaic_radio )
        mosaic_radio.on_clicked( self.update_mosaic )
    

            
        # Create flip buttons
        self.flipx = 1;
        flipx_ax = plt.axes( [0.3, 1 - 0.09, 0.1, 0.09] )
        flipx_button = Button( flipx_ax, 'Flip\nUp/Down', color='gainsboro' )
        flipx_button.on_clicked(self.update_flipx);

        self.flipy = 1;
        flipy_ax = plt.axes( [0.3, 1 - 0.18, 0.1, 0.09] )
        flipy_button = Button( flipy_ax, 'Flip\nLeft/Right', color='gainsboro' )
        flipy_button.on_clicked(self.update_flipy);

        
        # Create Refresh buttons
        refresh_ax = plt.axes( [0.4, 1 - 0.09, 0.1, 0.09] )
        refresh_button = Button( refresh_ax, 'Refresh', color='gainsboro' )
        refresh_button.on_clicked(self.update_refresh);

        # Create Save button
        
        save_ax = plt.axes( [0.4, 1 - 0.18, 0.1, 0.09] )
        save_button = Button( save_ax, 'Export to\nPNG', color='gainsboro' )
        save_button.on_clicked(self.save);


        # Create dynamic refresh radio button
        #self.drefresh = threading.Event()
        #drefresh_ax = plt.axes( [0.4, 1 - 0.18, 0.1, 0.09] )
        #drefresh_button = Button( drefresh_ax, 'Dynamic\nRefresh', color='gainsboro' )
        #drefresh_button.on_clicked(self.update_drefresh);


        # Create Magnitude/phase radio button
        self.magnitude = True
        mag_ax = plt.axes( [0.5, 1 - 0.18, 0.1, 0.18], axisbg = 'gainsboro' )
        mag_radio = RadioButtons( mag_ax, ('Mag','Phase') , activecolor = 'SteelBlue', active = 0 )
        radios.append( mag_radio )
        mag_radio.on_clicked( self.update_magnitude )

        
        sbox_ax = plt.axes( [0.6, 1 - 0.18, 0.5, 0.18], axisbg='gainsboro')
        sbox_ax.set_xticks([])
        sbox_ax.set_yticks([])
        
        # Create aspect sliders
        aspect_ax = plt.axes( [0.65, 1 - 0.09 + 0.02, 0.1, 0.02], axisbg = 'white' )
        aspect_slider = Slider( aspect_ax, "", 0.25, 4, valinit=1, color='SteelBlue')
        aspect_ax.text( 4 / 2,1.5,  'Aspect Ratio', horizontalalignment = 'center')
        sliders.append( aspect_slider )
        aspect_slider.on_changed( self.update_aspect )
        
        # Create contrast sliders
        self.vmin = 0
        vmin_ax = plt.axes( [0.83, 1 - 0.09 + 0.02, 0.1, 0.02], axisbg = 'white' )
        vmin_slider = Slider( vmin_ax, "", 0, 1, valinit=0, color='SteelBlue')
        vmin_ax.text(0.5,1.5,  'Contrast Min', horizontalalignment = 'center')
        sliders.append( vmin_slider )
        vmin_slider.on_changed( self.update_vmin )

        self.vmax = 1
        vmax_ax = plt.axes( [0.83, 1 - 0.18 + 0.02, 0.1, 0.02], axisbg = 'white' )
        vmax_slider = Slider( vmax_ax, "", 0, 1, valinit=1, color='SteelBlue')
        vmax_ax.text(0.5,1.5,  'Contrast Max', horizontalalignment = 'center')
        sliders.append( vmax_slider )
        vmax_slider.on_changed( self.update_vmax )

        
        # Create sliders for choosing slices
        box_ax = plt.axes( [0, 1 - 0.25, 1, 0.07], axisbg='gainsboro')
        box_ax.set_xticks([])
        box_ax.set_yticks([])

        slider_thick = 0.02
        slider_start = 0.1
        ax = []
        for d in np.r_[:self.Ndims]:
            slice_ax = plt.axes( [0.01 + 1 / self.Ndims * d, 1 - 0.24, 0.8 / self.Ndims, slider_thick] , axisbg='white')
            slice_slider = DiscreteSlider( slice_ax, "", 0, self.im_shape[d]-1, valinit=self.slice_num[d],valfmt='%i', color='SteelBlue')
            slice_ax.text( (self.im_shape[d]-1)/2,1.5,  'Dim %d Slice' % self.im_unsqueeze_shape[d], horizontalalignment = 'center' )
            sliders.append(slice_slider);
            slice_slider.on_changed( partial( self.update_slice, d ) )
            

        plt.show()

    def readcfl(self, name):
            h = open(name + ".hdr", "r")
            h.readline() # skip
            l = h.readline()
            dims = [int(i) for i in l.split( )]
            n = reduce(operator.mul, dims, 1)
            h.close()
            return np.memmap( name + ".cfl", dtype = np.complex64, mode='r', shape=tuple(dims), order='F' )

    def save( self, event ):
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        num = 0
        fname = self.cflname + '_' + str(num) + '.png'
        while( os.path.isfile(fname)  ):
            num += 1
            fname = self.cflname + '_' + str(num) + '.png'
            
        self.fig.savefig(fname, bbox_inches=extent)

    def update_flipx( self, event ):
        self.flipx *= -1
        self.update_image()

        
    def update_flipy( self, event ):
        self.flipy *= -1
        self.update_image()

    def update_refresh( self, event ):
        self.update_image()

    def dynamic_refresh( self ):
        while( self.drefresh.is_set() ):
            self.update_image()

    def update_drefresh( self, event ):
        if ( self.drefresh.is_set() ):
            self.drefresh.clear()
        else:
            self.drefresh.set()
            th = threading.Thread( target = self.dynamic_refresh )
            th.start()
            
    def update_aspect( self, aspect ):
        self.aspect = aspect
        self.ax.set_aspect( self.asp * self.im_ordered.shape[0] / self.im_ordered.shape[1] * aspect )

    def update_vmax( self, vmax ):
        self.vmax = max(self.vmin, vmax)
        self.l.set_clim( vmax = self.vmax * self.immax );

    def update_vmin( self, vmin ):
        self.vmin = min(self.vmax,vmin)
        self.l.set_clim( vmin = self.vmin * self.immax );

    def update_magnitude( self, l ):
        self.magnitude =  ( l == 'Mag' )
        if (self.magnitude):
            self.l.set_cmap('gray')
        else:
            self.l.set_cmap('hsv')
        self.update_image()
            
    def update_orderx( self, l ):
        l = int(l[4:])
        self.order3[0] = np.where( self.im_unsqueeze_shape == l )[0]
        self.update_ordered_image()

    def update_ordery( self, l ):
        l = int(l[4:])
        self.order3[1] = np.where( self.im_unsqueeze_shape == l )[0]
        self.update_ordered_image()

        
    def update_ordered_image(self):
        self.mosaic_valid = len( self.order3[:3] ) == len( set( self.order3[:3] ) )
        self.order_valid = len( self.order3[:2] ) == len( set( self.order3[:2] ) );
        
        if ( self.mosaic_valid ):
            self.order[:3] = self.order3[:3]
            order_remain = np.r_[:self.Ndims]
            for t in np.r_[:3]:
                order_remain = order_remain[  (order_remain != self.order[t] ) ]
            self.order[3:] = order_remain
            self.im_ordered = np.transpose( self.im, self.order )
            self.ax.set_aspect( self.asp * self.im_ordered.shape[0] / self.im_ordered.shape[1] * self.aspect )
            self.update_image()
        elif ( self.order_valid ):
            self.order[:2] = self.order3[:2]
            order_remain = np.r_[:self.Ndims]
            for t in np.r_[:2]:
                order_remain = order_remain[  (order_remain != self.order[t] ) ]
            self.order[2:] = order_remain
            self.im_ordered = np.transpose( self.im, self.order )
            self.ax.set_aspect( self.asp * self.im_ordered.shape[0] / self.im_ordered.shape[1] * self.aspect )
            self.update_image()


    def update_image( self ):
        self.immax = np.max(abs(self.im))
        self.l.set_clim( vmin = self.vmin * self.immax ,  vmax = self.vmax * self.immax );
        if ( self.mosaic_valid ):
            im_slice = self.im_ordered[ (slice(None,None,self.flipx), slice(None,None,self.flipy), slice(None)) + tuple(self.slice_num[self.order[3:]])]
            im_slice = self.mosaic( im_slice )
        else:
            im_slice = self.im_ordered[ (slice(None,None,self.flipx), slice(None,None,self.flipy)) + tuple(self.slice_num[self.order[2:]]) ]
        
        if self.magnitude:
            self.l.set_data( abs(im_slice) )
        else:
            self.l.set_data( (np.angle(im_slice) + np.pi) / (2 * np.pi) )


        
        self.fig.canvas.draw()

    def update_slice( self, d, s ):
        self.slice_num[d] = int(round(s))
        self.update_image()

    def mosaic( self, im ):
        im = im.squeeze()
        (x, y, z) = im.shape
        z2 = int( np.ceil( z ** 0.5 ) )
        z = int( z2 ** 2 )
        im = np.pad( im, [(0,0), (0,0), (0, z - im.shape[2] )], mode='constant')
        im = im.reshape(  (x, y * z, 1), order = 'F' )

        im = im.transpose( (1, 2, 0) )
        im = im.reshape( (y * z2 , z2, x), order = 'F' )
        im = im.transpose( (2, 1, 0) )
        im = im.reshape( (x * z2, y * z2), order = 'F' )

        return im

    def update_mosaic( self, l ):
        l = int(l[4:])
        self.order3[2] = np.where( self.im_unsqueeze_shape == l )[0]
        self.update_ordered_image()

if __name__ == "__main__":

    # Error if more than 1 argument
    if (len(sys.argv) != 2):
        print "BartView: multidimensional image viewer for cfl"
        print "Usage: bview cflname"
        exit()

    BartView( sys.argv[1] )

    
