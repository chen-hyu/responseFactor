""" A Python implementation for calculating and plotting response factor by Hong-Yu Chen 2024"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import RegularGridInterpolator

pi = np.pi

# use in geo plot
def geoplot_xticks(longtitude):
    longtitude_plot = np.copy(longtitude)
    longtitude_plot[np.where(longtitude >= pi)] -= 2*pi
    return geoplot_result(longtitude_plot)
def geoplot_result(longtitude):
    longtitude_plot = np.zeros_like(longtitude)
    N_len = len(longtitude)
    if (N_len % 2 == 1):
        longtitude_plot[0:N_len//2+1] = np.copy(longtitude[-N_len//2::])
        longtitude_plot[-N_len//2+1::] = np.copy(longtitude[0:N_len//2])
    elif (N_len % 2 == 0):
        longtitude_plot[0:N_len//2] = np.copy(longtitude[-N_len//2::])
        longtitude_plot[-N_len//2::] = np.copy(longtitude[0:N_len//2])
    else:
        raise ValueError
    return longtitude_plot

class ResFac:
    def __init__(self):
        N = 1000
        # parameter lists
        self.lamb_list = np.linspace(0,2*pi,N)
        self.beta_list = np.arcsin(np.linspace(-1,1,N))
        self.psi_list  = np.linspace(0,pi,N)
        self.iota_list = np.arccos(np.linspace(-1,1,N))

        # reading the result
        self.sky_result = np.loadtxt('RF_sky_result.txt')
        self.pol_result = np.loadtxt('RF_polar_result.txt')

        # TianQin orbit
        self.thetas = -4.7
        self.phis = 120.5
        orbit_tq = np.loadtxt('TQ_orbit.txt').T
        lamb_tq = geoplot_xticks(orbit_tq[0]/180*pi)
        beta_tq = geoplot_result(orbit_tq[1]/180*pi)
        idx_max = np.where(lamb_tq==np.max(lamb_tq))[0][0]
        idx_min = np.where(lamb_tq==np.min(lamb_tq))[0][0]
        # print(idx_max, idx_min)
        self.lamb_tq = lamb_tq[idx_max:idx_min+1]*180/pi
        self.beta_tq = beta_tq[idx_max:idx_min+1]*180/pi

        self.factor = 180/pi
        self.geoplot_lamb_list = geoplot_xticks(self.lamb_list)
        self.geoplot_sky_result = geoplot_result(self.sky_result).T
    
    def plot_sky(self):
        levels = np.arange(0.4,1.6,0.2)

        plt.figure()
        bmap = Basemap(projection='moll'
                        ,lon_0=0
                        ,lat_0=0
                        ,resolution='i' # c (crude), l (low), i (intermediate), h (high), f (full)
                        )
        Lamb_list, Beta_list = bmap(*np.meshgrid(self.geoplot_lamb_list*self.factor, self.beta_list*self.factor))

        bmap.drawmeridians(np.arange(0,360,30),color='grey')
        bmap.drawparallels(np.arange(-90,90,30),color='grey')

        bmap.contourf(Lamb_list,Beta_list,self.geoplot_sky_result
                        ,levels=levels
                        ,cmap='Blues'
                        ,zorder=1)

        bmap.colorbar(ticks=levels)
        CS = bmap.contour(Lamb_list,Beta_list,self.geoplot_sky_result,levels=[1.0],colors='black',zorder=2)
        manual_locations = [bmap(-120,0), bmap(60, 0)]
        clabels = plt.clabel(CS, fontsize=12, inline=True, manual=manual_locations, fmt='%1.1f', zorder=3)
        for label in clabels:
            label.set_rotation(0)

        # convert to map projection coords. 
        # Note that lon, lat can be scalars, lists or numpy arrays.
        lamb_tq_pt,beta_tq_pt = bmap(self.lamb_tq,self.beta_tq)
        phis_pt,thetas_pt = bmap(self.phis,self.thetas)

        bmap.plot(lamb_tq_pt,beta_tq_pt,'r--',zorder=4, rasterized=True)
        bmap.scatter(phis_pt,thetas_pt,c='r',marker='*',zorder=4, rasterized=True)

        plt.xlabel(r'Ecliptic Longitude $\lambda$',fontsize=20)
        plt.ylabel(r'Ecliptic Latitude $\beta$',fontsize=20)
        plt.savefig('TQ_sky.png',bbox_inches='tight')
        plt.savefig('TQ_sky.pdf',bbox_inches='tight')
        plt.show()
    
    def plot_polar(self):
        levels = np.arange(0.4,1.6,0.2)

        plt.figure()
        bmap = Basemap(projection='moll'
                        ,lon_0=0
                        ,lat_0=0
                        ,resolution='i' # c (crude), l (low), i (intermediate), h (high), f (full)
                        )
        Psi_list, Iota_list = bmap(*np.meshgrid(self.psi_list*self.factor, self.iota_list*self.factor-90))
        Psi_list2, Iota_list2 = bmap(*np.meshgrid(self.psi_list*self.factor*-1, self.iota_list*self.factor-90))

        bmap.drawmeridians(np.arange(0,360,30),color='grey')
        bmap.drawparallels(np.arange(-90,90,30),color='grey')

        bmap.contourf(Psi_list,Iota_list,self.pol_result.T
                        ,levels=levels
                        ,cmap='Blues'
                        )
        bmap.contourf(Psi_list2,Iota_list2,self.pol_result.T
                        ,levels=levels
                        ,cmap='Blues'
                        )
        bmap.colorbar(ticks=levels)
        CS = plt.contour(Psi_list,Iota_list,self.pol_result.T,levels=[1.0],colors='black')
        CS2 = plt.contour(Psi_list2,Iota_list2,self.pol_result.T,levels=[1.0],colors='black')

        clabels = plt.clabel(CS, fontsize=12, inline=True, manual=[bmap(30, 40)], fmt='%1.1f')
        clabels2 = plt.clabel(CS2, fontsize=12, inline=True, manual=[bmap(-30, -40)], fmt='%1.1f')
        for label in clabels:
            label.set_rotation(0)
        for label in clabels2:
            label.set_rotation(0)

        plt.xlabel(r'Polarization $\psi$',fontsize=20)
        plt.ylabel(r'Inclination $\iota$',fontsize=20)
        plt.savefig('TQ_polar.png',bbox_inches='tight')
        plt.savefig('TQ_polar.pdf',bbox_inches='tight')
        plt.show()

    def calculate(self,lamb,beta,psi,iota):
        sky_interp = RegularGridInterpolator((self.lamb_list, self.beta_list), self.sky_result)
        pol_interp = RegularGridInterpolator((self.psi_list,  self.iota_list), self.pol_result)

        RF = sky_interp([lamb, beta])[0]*pol_interp([psi, iota])[0]
        return RF