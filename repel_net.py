#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:04:16 2019

@author: virati, Vineet Tiruvadi
Two-pop diffusion network with repulsion
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class diff_net:
    def __init__(self,nodes=100,repuls_strength=1):
        self.N = nodes
        self.set_L()
        self.state = np.zeros((self.N,2))
        self.state[0:50] = np.random.normal(5,2,(int(self.N/2),2))
        self.state[50:100] = np.random.normal(-5,2,(int(self.N/2),2))
        self.state_raster = []
        self.repuls_strength = repuls_strength
        
    def set_L(self):
        self._L = np.zeros((self.N,self.N))
        
        for ii in range(0,50):
            for jj in range(50,100):
                self._L[ii,jj] = 1
                self._L[jj,ii] = 1
        
        #can be simplified into
        # self._L = np.array([[1 for jj in range(ii,50)] for ii in range(50)])
        # do a symmetric copy
        
    def dynamics(self,state):
        update = -np.dot(self._L,state)
        update = np.tanh(update)
        #pdb.set_trace()
        return self.repuls_strength*update
    

    def integrator(self):
        k1 = self.dynamics(self.state) * self.dt
        k2 = self.dynamics(self.state + .5*k1)*self.dt
        k3 = self.dynamics(self.state + .5*k2)*self.dt
        k4 = self.dynamics(self.state + k3)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        new_state += np.random.normal(0,10,new_state.shape) * self.dt
        
        return new_state
    
    def run_sim(self,tend=10,dt=0.1):
        self.dt = dt
        for tt in np.arange(0,tend,dt):
            print(tt)
            self.state_raster.append(self.state)
            self.state = self.integrator()
            
    def map_atTime(self,tstep=0):
        plt.scatter(self.state_raster[tstep][0:50,0],self.state_raster[tstep][0:50,1],color='red')
        plt.scatter(self.state_raster[tstep][50:100,0],self.state_raster[tstep][50:100,1],color='blue')
        plt.ylim((-40,40))
        plt.xlim((-40,40))
    
    def map_gif(self,tend=10):
        pass
    
    
class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, state_raster, label=''):
        #state_raster = network.state_raster[tstep]
        plt_im1 = plt.scatter(state_raster[0:50,0],state_raster[0:50,1],color='red')
        plt_im2 = plt.scatter(state_raster[50:100,0],state_raster[50:100,1],color='blue')
        plt.ylim((-40,40))
        plt.xlim((-40,40))
        
        self.images.append([plt_im1, plt_im2])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=5)


#%%
# Setup and run the diffusion
net = diff_net(repuls_strength=1)
net.run_sim()

#%%
#Animation part
gif = AnimatedGif()
gif.add(net.state_raster[0],label='0')
for tt in range(100):
    gif.add(net.state_raster[tt])
    
gif.save('/tmp/repeldiff_repuls.gif')


#%%
## Crappy way to animate
### Animation time, but this code sucks, need to use proper animation next
#plt.figure()
#for tt in range(100):
#    plt.clf()
#    net.map_atTime(tstep=tt)
#    plt.pause(0.05)