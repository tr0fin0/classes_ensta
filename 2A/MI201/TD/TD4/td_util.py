#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:50:26 2017

@author: stephane
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from matplotlib.colors import ListedColormap

seaborn.set()

def plot_svc_decision_regions(model, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    
    Z = model.predict(xy)
   # Z = classifier.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
    ny = max(Z)
    Z = Z.reshape(Y.shape)
    plt.contourf(X, Y, Z,  alpha=0.2, cmap= ListedColormap(seaborn.color_palette()[:ny+1]))
    #plot.contourf(X, Y, Z,  alpha=0.3)
   
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_svc_decision_function(model, ax=None, plot_support=False):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    Z = 1*(P>0) -1*(P<=0)
    plt.contourf(X, Y, Z,  alpha=0.2)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1], color='k',
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_decision_regions(X, y, classifier,  sv_show = False, resolution=0.02):

    # setup marker generator and color map
    markers = ('x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #cmap = ListedColormap(colors[:len(np.unique(y))])
    #cmap = seaborn.color_palette(:len(np.unique(y)))
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   # Z = classifier.decision_function(np.c_[xx1.ravel(), xx2.ravel()])

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3)
#    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#    plt.contour(xx1, xx2,  Z, colors=['k', 'k', 'k'],
#                linestyles=['--', '-', '--'], levels=[-.1, 0, .1])
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    '''
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
        if (sv_show):
            try:
                plt.scatter(classifier.support_vectors_[:,0],
                    classifier.support_vectors_[:,1],
                    marker='o', edgecolor='black', s=40, c='g')       
            except AttributeError:
                pass
    '''

def generate_data(test_number=0, noise = 0, nsample=50, seed=20):
    """
        Data generator.
        

        Args:
            test_number: Number of data family (0 -> 4)
            noise: Ratio of noisy data (between 0 to 1.0)
            nsample: number of samples 
        
        Returns:
            trainX, trainY, testX, testY
    """
    np.random.seed(seed)
    if test_number == 0:
        print('Gaussian two class isovariance samples')
        mu = [2,3]
        sigma = [[1,1.5],[1.5,3]]
        
        r1 = np.random.multivariate_normal(mu,sigma,nsample)
        r1t = np.random.multivariate_normal(mu,sigma,2*nsample)
        r2 = np.random.multivariate_normal(mu+np.array([1.5,0]),sigma,nsample)       
        r2t = np.random.multivariate_normal(mu+np.array([1.5,0]),sigma,2*nsample)
        
    elif test_number == 1:
        print('Gaussian two class heterogeneous variance samples')
        
        mu = [2,3]
        sigma1 = [[1,1.5],[1.5,3]]
        r1 = np.random.multivariate_normal(mu,sigma1,nsample)
        r1t = np.random.multivariate_normal(mu,sigma1,2*nsample)
        
        sigma2 = [[0.3,0.1],[0.1,0.2]]
        r2 = np.random.multivariate_normal(mu+np.array([1.5,0]),sigma2,nsample)
        r2t = np.random.multivariate_normal(mu+np.array([1.5,0]),sigma2,2*nsample)
        
    elif test_number == 2:
        print('Intricated data');
             
        mu = [0,0]
        sigma1 = [[0.5,0],[0,0.5]]
        r1 = np.random.multivariate_normal(mu,sigma1,nsample)
        r1t = np.random.multivariate_normal(mu,sigma1,2*nsample)
        
        rho = 2+np.random.beta(1,1,size=nsample)
        t = 0.4*np.random.randn(nsample)*np.pi
        r2 = np.vstack((rho * np.sin(t), rho * np.cos(t))).transpose()

        rho = 2+np.random.beta(1,1,size=2*nsample)
        t = 0.4*np.random.randn(2*nsample)*np.pi
        r2t = np.vstack((rho * np.sin(t), rho * np.cos(t))).transpose()
                                               
    elif test_number == 3:
        print('XOR like distribution')
        
        r1 = np.vstack((np.random.multivariate_normal([0,0],0.3*np.eye(2),2*nsample),
                     np.random.multivariate_normal([2,2],0.3*np.eye(2),2*nsample)))
        r1t = np.vstack((np.random.multivariate_normal([0,0],0.3*np.eye(2),2*nsample),
                     np.random.multivariate_normal([2,2],0.3*np.eye(2),2*nsample)))
              
        r2 = np.vstack((np.random.multivariate_normal([2,0],0.3*np.eye(2),2*nsample),
                     np.random.multivariate_normal([0,2],0.3*np.eye(2),2*nsample)))
        r2t = np.vstack((np.random.multivariate_normal([2,0],0.3*np.eye(2),2*nsample),
                     np.random.multivariate_normal([0,2],0.3*np.eye(2),2*nsample)))
             
        
    elif test_number ==  4:
        print('Three classes')
        mu = [2,3];
        sigma1 = [[1,1.5],[1.5,3]];
        r1 = np.random.multivariate_normal(mu,sigma1,nsample)
        r1t = np.random.multivariate_normal(mu,sigma1,2*nsample)

        sigma1 = [[0.3,0.1],[0.1,1]];
        r2 = np.random.multivariate_normal(mu+np.array([2.5,0]),sigma1,nsample)
        r2t = np.random.multivariate_normal(mu+np.array([2.5,0]),sigma1,2*nsample)

        r2 = np.vstack((r2,np.random.multivariate_normal(mu+np.array([-2.5,0]),sigma1,nsample)))
        r2t = np.vstack((r2t,np.random.multivariate_normal(mu+np.array([-2.5,0]),sigma1,2*nsample)))

        sigma1 = [[5,0.1],[0.1,1]];
        r3 = np.random.multivariate_normal(mu+np.array([0,-3]),sigma1,nsample)
        r3t = np.random.multivariate_normal(mu+np.array([0,-3]),sigma1,2*nsample)

    else:
        return

    try:
        trainX = np.vstack((r1,r2,r3))
        trainY = np.array([0]*r1.shape[0] + [1]*r2.shape[0] + [2]*r3.shape[0])
    
        testX = np.vstack((r1t,r2t,r3t))
        testY = np.array([0]*r1t.shape[0] + [1]*r2t.shape[0] + [2]*r3t.shape[0])
        
    except NameError:
        trainX = np.vstack((r1,r2))
        trainY = np.array([0]*r1.shape[0] + [1]*r2.shape[0])
    
        testX = np.vstack((r1t,r2t))
        testY = np.array([0]*r1t.shape[0] + [1]*r2t.shape[0])

    if (noise > 0 and noise <= 1.0):
        nout =  int(noise * trainX.shape[0])
        idnoise=np.random.permutation(trainX.shape[0])
        trainY[idnoise[1:nout]] = 1-trainY[idnoise[1:nout]]
    
        idswitch=np.random.permutation(trainX.shape[0])

        trainX = trainX[idswitch,:]
        trainY = trainY[idswitch]

        nout =  int(noise * testX.shape[0])
        idnoise=np.random.permutation(testX.shape[0])
        testY[idnoise[1:nout]] = 1-testY[idnoise[1:nout]]
    
        idswitch=np.random.permutation(testX.shape[0])

        testX = testX[idswitch,:]
        testY = testY[idswitch]

    return (trainX, trainY, testX, testY)

def show_data_2D(X,Y):
    np.unique(Y)
    fig, ax = plt.subplots(figsize=(8, 6))
    for id in np.unique(Y):
        idpositive=np.nonzero(Y == id)[0]
        ax.scatter(X[idpositive,0], X[idpositive,1], s=50)


