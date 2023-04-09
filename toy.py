#import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#colors
rd = (230/255, 45/255,  40/255,  .7)
gr = (45/255,  170/255, 95/255,  .7)
bl = (45/255,  140/255, 200/255, .7)

def plot_train(ax,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max):
    """Function for plotting the training data
    Inputs
    ----------
    ax: plt.axes
        The axis of a fig upon which the data will be plotted
    x_train: np.array
        Training data predictors
    y_train: np.array
        Training data target
    classes: np.array
        Unique class names
    colors: list
        List of colors of classes
    mk_size: int
        Plot marker size
    x_axis_min: int or float
        Minimum value of plot's x axis
    x_axis_max: int or float
        Maximum value of plot's x axis
    y_axis_min: int or float
        Minimum value of plot's y axis
    y_axis_max: int or float
        Maximum value of plot's y axis
    Returns
    ----------
    None (Visualization only)
    """
    for i,(yi,c) in enumerate(zip(classes,colors)):
            class_ids = y_train==yi
            ax.scatter(x_train[class_ids,0], x_train[class_ids,1], marker = 'o', s = mk_size, color = c, alpha = 1,
                        label = 'Class={} (Training Data)'.format(yi))
    ax.legend(loc = 'upper left')
    ax.set_title('Training Data')
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)

def plot_test(ax,x_train,y_train,x_test,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max):
    """Function for plotting the testing data with the training data
    Inputs
    ----------
    ax: plt.axes
        The axis of a fig upon which the data will be plotted
    x_train: np.array
        Training data predictors
    y_train: np.array
        Training data target
    x_test: np.array
        Testing data predictors
    classes: np.array
        Unique class names
    colors: list
        List of colors of classes
    mk_size: int
        Plot marker size
    x_axis_min: int or float
        Minimum value of plot's x axis
    x_axis_max: int or float
        Maximum value of plot's x axis
    y_axis_min: int or float
        Minimum value of plot's y axis
    y_axis_max: int or float
        Maximum value of plot's y axis
    Returns
    ----------
    None (Visualization only)
    """
    ax.scatter(x_test[:,0], x_test[:,1], s = mk_size, edgecolors = 'black', marker = '^', facecolors = 'white', 
                label = 'Predicted Class=? (Testing Data)')
    ax.legend(loc = 'upper left')
    ax.set_title('Training and Testing Data')
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)

def plot_knn_classif(ax,x_train,y_train,x_test,pred_class,pred_class_color,kindices,
                     classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max):
    """Function for plotting how KNN classifies
    Inputs
    ----------
    ax: plt.axes
        The axis of a fig upon which the data will be plotted
    x_train: np.array
        Training data predictors
    y_train: np.array
        Training data target
    x_test: np.array
        Testing data predictors
    pred_class: int or string
        Predicted class of testing data point
    pred_class_color: tuple
        RGBA values representing color of testing data's predicted class
    kindices: np.array
        Index values in the training data of nearest neighbors to testing data point
    classes: np.array
        Unique class names
    mk_size: int
        Plot marker size
    x_axis_min: int or float
        Minimum value of plot's x axis
    x_axis_max: int or float
        Maximum value of plot's x axis
    y_axis_min: int or float
        Minimum value of plot's y axis
    y_axis_max: int or float
        Maximum value of plot's y axis
    Returns
    ----------
    None (Visualization only)
    """
    #plot classified testing data
    ax.scatter(x_test[:,0], x_test[:,1], s = mk_size,  edgecolors = 'black', marker = '^', 
                facecolors = pred_class_color,
                label = 'Predicted Class={} (Testing Data)'.format(pred_class), alpha = 1)
    #mark lines of k closest points
    for ki in kindices:
        ax.plot([x_train[ki,0], x_test[0,0]], [x_train[ki,1], x_test[0,1]], color = 'gray', 
                 linestyle = '--', zorder = 0)
    ax.legend(loc = 'upper left', framealpha = 1)
    ax.set_title('Identify K Closest Points, Classify Testing Data')
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)


def plot_tree_classif(ax,x_train,y_train,x_test,pred_class,pred_class_color,
                     classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max,
                     split_features,split_thresholds,dt,forest = False):
    """Function for plotting how KNN classifies
    Inputs
    ----------
    ax: plt.axes
        The axis of a fig upon which the data will be plotted
    x_train: np.array
        Training data predictors
    y_train: np.array
        Training data target
    x_test: np.array
        Testing data predictors
    pred_class: int or string
        Predicted class of testing data point
    pred_class_color: tuple
        RGBA values representing color of testing data's predicted class
    kindices: np.array
        Index values in the training data of nearest neighbors to testing data point
    classes: np.array
        Unique class names
    colors: list
        List of colors corresponding to represent each class
    mk_size: int
        Plot marker size
    x_axis_min: int or float
        Minimum value of plot's x axis
    x_axis_max: int or float
        Maximum value of plot's x axis
    y_axis_min: int or float
        Minimum value of plot's y axis
    y_axis_max: int or float
        Maximum value of plot's y axis
    split_features: np.array
        sklearn DecisionTreeClassifier's tree_.feature attribute value
    split_thresholds: np.array
        sklearn DecisionTreeClassifier's tree_.threshold attribute value
    dt: sklearn DecisionTreeClassifier instance
    forest: bool
        Indicates whether tree is part of a random forest
    Returns
    ----------
    None (Visualization only)
    """
    
    #helper function to convert predicted class from number to class name for trees and random forests
    def pred_to_classname(pred, classes, forest = False):
        if forest:
            return classes[int(pred)]
        else:
            return pred
    
    #plot classified testing data
    ax.scatter(x_test[:,0], x_test[:,1], s = mk_size, edgecolors = 'black', marker = '^', 
                facecolors = pred_class_color,
                label = 'Predicted Class={} (Testing Data)'.format(pred_class), alpha = 1) 
    
    #plot tree splits
    prev_split = (None,None)
    for i,(f,t) in enumerate(zip(split_features,split_thresholds)):
        #handle horizontal split
        if f == 1:
            #handle as first split
            if prev_split[0] == None:
                ax.hlines(y = t, xmin = x_axis_min, xmax = x_axis_max, color = 'black')
                #handle where left children go to leaf node
                if split_features[i+1] == -2:
                    pred_class_left = pred_to_classname(dt.predict(np.array([[0,t-1]])), classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = t, y2 = y_axis_min, 
                                     color = pred_class_color_left, alpha = 0.3)
                #handle where right children go to leaf node
                if split_features[i+2] == -2:
                    pred_class_right = pred_to_classname(dt.predict(np.array([[0,t+1]])), classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = y_axis_max, y2 = t, 
                                     color = pred_class_color_right, alpha = 0.3)
            #handle as second split if previous split was vertical
            if prev_split[0] == 0:
                #and this split is of left children of previous split
                if prev_split[2] != -2:
                    #mark split
                    ax.hlines(y = t, xmin = x_axis_min, xmax = prev_split[1], color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[prev_split[1]-1,t-1]])), 
                                                        classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,prev_split[1]], y1 = t, y2 = y_axis_min, 
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[prev_split[1]-1,t+1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [x_axis_min,prev_split[1]], y1 = y_axis_max, y2 = t, 
                                     color = pred_class_color_right, alpha = 0.3)
                #and this split is of right children of previous split 
                if prev_split[2] == -2:
                    #mark split
                    ax.hlines(y = t, xmin = prev_split[1], xmax = x_axis_max, color = 'black')   
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[prev_split[1]+1,t-1]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [prev_split[1],x_axis_max], y1 = t, y2 = y_axis_min,     
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[prev_split[1]+1,t+1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [prev_split[1],x_axis_max], y1 = y_axis_max, y2 = t, 
                                     color = pred_class_color_right, alpha = 0.3) 
            #handle as second split if previous split was also horizontal
            if prev_split[0] == 1:
                #and this split is of left children of previous split
                if prev_split[2] != -2:
                    #mark split
                    ax.hlines(y = t, xmin = x_axis_min, xmax = x_axis_max, color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[0,t-1]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = t, y2 = y_axis_min, 
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[0,np.mean([t,prev_split[1]])]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = prev_split[1], y2 = t, 
                                     color = pred_class_color_right, alpha = 0.3)
                #and this split is of right children of previous split
                if prev_split[2] == -2:
                    #mark split
                    ax.hlines(y = t, xmin = x_axis_min, xmax = x_axis_max, color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[0,np.mean([t,prev_split[1]])]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = t, y2 = prev_split[1], 
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[0,t+1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [x_axis_min,x_axis_max], y1 = y_axis_max, y2 = t, 
                                     color = pred_class_color_right, alpha = 0.3)
            prev_split = (f,t,split_features[i+1],split_features[i+2])
        #handle vertical split
        elif f == 0:
            #handle as first split
            if prev_split[0]==None:
                #mark split
                ax.vlines(x = t, ymin = y_axis_min, ymax = y_axis_max, color = 'black')
                #handle where left children go to leaf node
                if split_features[i+1] == -2:
                    pred_class_left = pred_to_classname(dt.predict(np.array([[t-1,0]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,t], y1 = y_axis_max, y2 = y_axis_min, 
                                     color = pred_class_color_left, alpha = 0.3)
                #handle where right children go to leaf node
                if split_features[i+2] == -2:
                    pred_class_right = pred_to_classname(dt.predict(np.array([[t+1,0]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [t,x_axis_max], y1 = y_axis_max, y2 = y_axis_min, 
                                     color = pred_class_color_right, alpha = 0.3)
            #handle second split if prev split is horizontal...
            if prev_split[0] == 1:
                #and this split is of left children of prev split
                if prev_split[2] != -2:
                    #mark split
                    ax.vlines(x = t, ymin = y_axis_min, ymax = prev_split[1], color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[t-1,prev_split[1]-1]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,t], y1 = prev_split[1], y2 = y_axis_min,
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[t+1,prev_split[1]-1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [t, x_axis_max], y1 = prev_split[1], y2 = y_axis_min,
                                     color = pred_class_color_right, alpha = 0.3)
                #and this split is of right children of prev split
                if prev_split[2] == -2:
                    #mark split 
                    ax.vlines(x = t, ymin = prev_split[1], ymax = y_axis_max, color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[t-1,prev_split[1]+1]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,t], y1 = y_axis_max, y2 = prev_split[1], 
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[t+1,prev_split[1]+1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [t,x_axis_max], y1 = y_axis_max, y2 = prev_split[1], 
                                     color = pred_class_color_right, alpha = 0.3)
            #handle second split if prev split is also vertical...
            if prev_split[0] == 0:    
                #and this split is of left children of prev split
                if prev_split[2] != -2:
                    #mark split
                    ax.vlines(x = t, ymin = y_axis_min, ymax = y_axis_max, color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[t-1,prev_split[1]-1]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [x_axis_min,t], y1 = y_axis_max, y2 = y_axis_min,
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[t+1,prev_split[1]-1]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [t, prev_split[1]], y1 = y_axis_max, y2 = y_axis_min,
                                     color = pred_class_color_right, alpha = 0.3)
                #and this split is of right children of prev split
                if prev_split[2] == -2:
                    #mark split 
                    ax.vlines(x = t, ymin = y_axis_min, ymax = y_axis_max, color = 'black')
                    #fill left child region
                    pred_class_left = pred_to_classname(dt.predict(np.array([[np.mean([t,prev_split[1]]),0]])), 
                                                         classes, forest)
                    pred_class_color_left = colors[np.where(classes==pred_class_left)[0][0]]
                    ax.fill_between(x = [prev_split[1],t], y1 = y_axis_max, y2 = y_axis_min, 
                                     color = pred_class_color_left, alpha = 0.3)
                    #fill right child region
                    pred_class_right = pred_to_classname(dt.predict(np.array([[t+1,0]])), 
                                                         classes, forest)
                    pred_class_color_right = colors[np.where(classes==pred_class_right)[0][0]]
                    ax.fill_between(x = [t,x_axis_max], y1 = y_axis_max, y2 = y_axis_min, 
                                     color = pred_class_color_right, alpha = 0.3)
            prev_split = (f,t,split_features[i+1],split_features[i+2])
    ax.legend(loc = 'upper left', framealpha = 1)    
    
def plot_knn_toy(x_train, y_train, x_test, k = 3):
    """Function for plotting KNN classification in 2 dimensions with the toy data
    Inputs
    ----------
    x_train: np.array
        Training data predictor values
    y_train: np.array
        Training data target values
    x_test: np.array
        Testing data predictor values
    k: int
        The value for k in the KNN algorithm
    Returns
    ----------
    None (Visualization only)
    """
    #check that x_train and x_test are 2d
    assert x_train.shape[1]==2 and x_test.shape[1]==2, 'Predictors must be 2-dimensional.'
    
    #store all classes
    classes = np.unique(y_train)
    
    #get data from knn model
    knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', metric = 'minkowski', p = 2)
    knn.fit(x_train,y_train)
    kdistances = knn.kneighbors(x_test)[0][0]
    kindices = knn.kneighbors(x_test)[1][0]
    pred_class = knn.predict(x_test)[0]
    
    #set styling variables
    ax_buffer_factor = 0.4
    x1_max = np.max(list(x_train[:,0]) + list(x_test[:,0]))
    x1_min = np.min(list(x_train[:,0]) + list(x_test[:,0]))
    x1_scale = np.abs(x1_max - x1_min)
    x_axis_min = x1_min - (x1_scale * ax_buffer_factor)
    x_axis_max = x1_max + (x1_scale * ax_buffer_factor)
    x2_max = np.max(list(x_train[:,1]) + list(x_test[:,1]))
    x2_min = np.min(list(x_train[:,1]) + list(x_test[:,1]))
    x2_scale = np.abs(x2_max - x2_min)
    y_axis_min = x2_min - (x2_scale * ax_buffer_factor)
    y_axis_max = x2_max + (x2_scale * ax_buffer_factor)
    mk_size = 250
    colors = [bl,rd]
    pred_class_color = colors[np.where(classes==pred_class)[0][0]]
    
    #plot
    fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,5), sharey = True)
    
    #plot training data
    plot_train(ax1,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    ax1.set_ylabel('Predictor 2')
    
    #add testing data
    plot_train(ax2,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    plot_test(ax2,x_train,y_train,x_test,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    ax2.set_xlabel('Predictor 1')

    #plot classification
    plot_train(ax3,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    plot_knn_classif(ax3,x_train,y_train,x_test,pred_class,pred_class_color,kindices,
                     classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    
    #fig formatting
    fig.suptitle('KNN Example', size = 20)
    fig.tight_layout()
    plt.show()

def plot_dt_toy(x_train,y_train,x_test,seed = 0):
    """Function for plotting decision tree boundaries in 2 dimensions with the toy data
    Note: tree will only make a maximum of 3 leaf nodes and input predictors must be 2d
    Inputs
    ----------
    x_train: np.array
        Training data predictor values
    y_train: np.array
        Training data target values
    x_test: np.array
        Testing data predictor values
    seed: int
        Value to set random_state in sklearn's DecisionTreeClassifier
    Returns
    ----------
    None (Visualization only)
    """
    #check that x_train and x_test are 2d
    assert x_train.shape[1]==2 and x_test.shape[1]==2, 'Predictors must be 2-dimensional.'

    #store all classes
    classes = np.unique(y_train)

    #get data from dt model
    dt = DecisionTreeClassifier(random_state = seed, max_leaf_nodes = 3, criterion = 'gini').fit(x_train,y_train)
    split_features = dt.tree_.feature
    split_thresholds = dt.tree_.threshold
    pred_class = dt.predict(x_test)[0]

    #set styling variables
    ax_buffer_factor = 0.4
    x1_max = np.max(list(x_train[:,0]) + list(x_test[:,0]))
    x1_min = np.min(list(x_train[:,0]) + list(x_test[:,0]))
    x1_scale = np.abs(x1_max - x1_min)
    x_axis_min = x1_min - (x1_scale * ax_buffer_factor)
    x_axis_max = x1_max + (x1_scale * ax_buffer_factor)
    x2_max = np.max(list(x_train[:,1]) + list(x_test[:,1]))
    x2_min = np.min(list(x_train[:,1]) + list(x_test[:,1]))
    x2_scale = np.abs(x2_max - x2_min)
    y_axis_min = x2_min - (x2_scale * ax_buffer_factor)
    y_axis_max = x2_max + (x2_scale * ax_buffer_factor)
    mk_size = 250
    colors = [bl,rd]
    pred_class_color = colors[np.where(classes==pred_class)[0][0]]
    
    #plot
    fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,5), sharey = True)

    #plot training data
    plot_train(ax1,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    ax1.set_ylabel('Predictor 2')
    
    #add testing data
    plot_train(ax2,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    plot_test(ax2,x_train,y_train,x_test,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    ax2.set_xlabel('Predictor 1')

    #plot classification
    plot_train(ax3,x_train,y_train,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
    plot_tree_classif(ax3,x_train,y_train,x_test,pred_class,pred_class_color,
                     classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max,
                     split_features,split_thresholds,dt, forest = False)
    ax3.set_title('Generate Splits, Classify Testing Data')
    
    #fig formatting
    fig.suptitle('Decision Tree Example', size = 20)
    fig.tight_layout()
    plt.show()
    
def plot_rf_toy(x_train,y_train,x_test,seed = 0):
    """Function for plotting decision tree boundaries in 2 dimensions for three trees in a forest with the toy data
    Note: each tree will only make a maximum of 3 leaf nodes and input predictors must be 2d
    and since this is just for a small toy dataset with only 2 predictors, each tree is fit using all predictors as candidates
    for splitting.
    Inputs
    ----------
    x_train: np.array
        Training data predictor values
    y_train: np.array
        Training data target values
    x_test: np.array
        Testing data predictor values
    seed: int
        Value to set random_state in sklearn's DecisionTreeClassifier
        and to set np's random seed for selecting samples of training data for each tree
    Returns
    ----------
    None (Visualization only)
    """
    
    #check that x_train and x_test are 2d
    assert x_train.shape[1]==2 and x_test.shape[1]==2, 'Predictors must be 2-dimensional.'

    #store all classes
    classes = np.unique(y_train)
    
    #set styling variables
    ax_buffer_factor = 0.4
    x1_max = np.max(list(x_train[:,0]) + list(x_test[:,0]))
    x1_min = np.min(list(x_train[:,0]) + list(x_test[:,0]))
    x1_scale = np.abs(x1_max - x1_min)
    x_axis_min = x1_min - (x1_scale * ax_buffer_factor)
    x_axis_max = x1_max + (x1_scale * ax_buffer_factor)
    x2_max = np.max(list(x_train[:,1]) + list(x_test[:,1]))
    x2_min = np.min(list(x_train[:,1]) + list(x_test[:,1]))
    x2_scale = np.abs(x2_max - x2_min)
    y_axis_min = x2_min - (x2_scale * ax_buffer_factor)
    y_axis_max = x2_max + (x2_scale * ax_buffer_factor)
    mk_size = 250
    colors = [bl,rd]
    
    #set seed
    np.random.seed(seed)
    
    #plot
    fig, ax = plt.subplots(1,3, figsize = (15,5), sharey = True)
    
    #subsample data points and build tree for each sample
    for i,ax_i in enumerate(ax):
        train_idx_rand = np.random.choice([0,1,2,3,4,5,6], size = 5, replace = False)
        x_train_samp = x_train[train_idx_rand,:].copy()
        y_train_samp = y_train_toy[train_idx_rand].copy()
        
        #get data from dt model
        dt = DecisionTreeClassifier(random_state = seed, max_leaf_nodes = 3, 
                                    criterion = 'gini').fit(x_train_samp,y_train_samp)
        split_features = dt.tree_.feature
        split_thresholds = dt.tree_.threshold
        pred_class = dt.predict(x_test)[0]
        pred_class_color = colors[np.where(classes==pred_class)[0][0]]
        #plot training data
        plot_train(ax_i,x_train_samp,y_train_samp,classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max)
        if i==0:
            ax_i.set_ylabel('Predictor 2')
        if i==1:
            ax_i.set_xlabel('Predictor 1')
        #plot classification
        plot_tree_classif(ax_i,x_train,y_train,x_test,pred_class,pred_class_color,
                         classes,colors,mk_size,x_axis_min,x_axis_max,y_axis_min,y_axis_max,
                         split_features,split_thresholds,dt, forest = False)
        ax_i.set_title('Tree {}'.format(i+1))
    #fig formatting
    fig.suptitle('Random Forest Example', size = 20)
    fig.tight_layout()
    plt.show()
    
#toy data
x_train_toy = np.array([[1.8,2.5], 
                        [2.2,3.0],
                        [1.5,3.0],
                        [2.1,1.1], 
                        [3.0,1.5],
                        [3.3,3.3],
                        [0.5,2.5]]) 
y_train_toy = np.array(['Class 1','Class 1','Class 1','Class 2','Class 2','Class 2','Class 2'])
x_test_toy = np.array([[2.3,2.2]])
train_df_toy = pd.DataFrame({'Predictor 1':x_train_toy[:,0],
                              'Predictor 2':x_train_toy[:,1],
                              'Class':y_train_toy})


########
########
#toy data and functions for examples illustrating utility of PCA rotation for tree classifier
def plot_train_tree_pca_ex(train_df, ax, pca = False):
    if pca:
        sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Class', hue_order = ['A','B'], s = 70, alpha = 0.7,
                        palette = 'BuPu', data = train_df, ax = ax)
    else:   
        sns.scatterplot(x = 'X1', y = 'X2', hue = 'Class', hue_order = ['A','B'], s = 70, alpha = 0.7,
                    palette = 'BuPu', data = train_df, ax = ax)
        
def plot_test_tree_pca_ex(test_df, y_pred, ax, pca = False):
    df = test_df.copy()
    y_test = df['Class']
    misclass = (y_test != y_pred)
    df['Misclassified\n(Testing Only)'] = ['yes' if i else 'no' for i in misclass]
    if pca:
        sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Class', hue_order = ['A','B'], 
                        style = 'Misclassified\n(Testing Only)', style_order = ['no','yes'], s = 70, alpha = 0.7,
                        palette = 'BuPu', data = df, ax = ax)
    else:
        sns.scatterplot(x = 'X1', y = 'X2', hue = 'Class', hue_order = ['A','B'], 
                        style = 'Misclassified\n(Testing Only)', style_order = ['no','yes'], s = 70, alpha = 0.7,
                        palette = 'BuPu', data = df, ax = ax)
        
def get_tree_pca_toy_data():
    np.random.seed(0)
    n = 150
    X1A = np.linspace(-10,10,n) 
    X2A = X1A + np.random.normal(loc = 0, scale = 1, size = n)
    X1B = np.linspace(-10,10,n) + 3
    X2B = X1B + np.random.normal(loc = 0, scale = 1, size = n) - 6
    dataA = pd.DataFrame(np.concatenate((np.vstack(X1A),np.vstack(X2A)), axis = 1), columns = ['X1','X2'])
    dataB = pd.DataFrame(np.concatenate((np.vstack(X1B),np.vstack(X2B)), axis = 1), columns = ['X1','X2'])
    data = pd.concat((dataA, dataB)).reset_index(drop = True)
    data['Class'] = 'B'
    data.loc[:149,'Class'] = 'A'
    return data

def plot_non_pca_model_ex(gridsearch, data, train, test, X_test, y_test):
    #plot model built on untransformed data
    features = gridsearch.best_estimator_.tree_.feature
    thresholds = gridsearch.best_estimator_.tree_.threshold
    y_pred = gridsearch.predict(X_test)
    test_acc = round(100*np.mean(y_pred == y_test))

    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,4), sharey = True)
    xmin,xmax = data['X1'].min()-1,data['X1'].max()+1
    ymin,ymax = data['X2'].min()-1,data['X2'].max()+1

    #plot model fitted to training data with decision boundary
    plot_train_tree_pca_ex(train, ax = ax1, pca = False)
    ax1.fill_between(x = [xmin,thresholds[1]], y1 = ymin, y2 = thresholds[0], color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[1],xmax], y1 = ymin, y2 = thresholds[0], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [xmin,thresholds[1]], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[4],xmax], y1 = thresholds[0], y2 = thresholds[7], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [thresholds[4],thresholds[1]], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[4],thresholds[9]], y1 = thresholds[7], y2 = thresholds[6], color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[9],xmax], y1 = thresholds[7], y2 = thresholds[6], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [thresholds[4],thresholds[14]], y1 = thresholds[6], y2 = thresholds[13], color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[14],xmax], y1 = thresholds[6], y2 = thresholds[13], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [thresholds[4],thresholds[12]], y1 = thresholds[13], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[12],thresholds[19]], y1 = thresholds[13], y2 = thresholds[20], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [thresholds[12],thresholds[19]], y1 = thresholds[20], y2 = thresholds[18], color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[19],xmax], y1 = thresholds[13], y2 = thresholds[18], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [thresholds[12],thresholds[24]], y1 = thresholds[18], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax1.fill_between(x = [thresholds[24],xmax], y1 = thresholds[18], y2 = ymax, color = 'purple', alpha = 0.2)
    ax1.set_title('Tree Fitted to Training Data')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    ax1.legend().set_visible(False)

    #plot test data with decision boundary
    plot_test_tree_pca_ex(test, y_pred = y_pred, ax = ax2, pca = False)
    ax2.fill_between(x = [xmin,thresholds[1]], y1 = ymin, y2 = thresholds[0], color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[1],xmax], y1 = ymin, y2 = thresholds[0], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [xmin,thresholds[1]], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[4],xmax], y1 = thresholds[0], y2 = thresholds[7], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [thresholds[4],thresholds[1]], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[4],thresholds[9]], y1 = thresholds[7], y2 = thresholds[6], color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[9],xmax], y1 = thresholds[7], y2 = thresholds[6], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [thresholds[4],thresholds[14]], y1 = thresholds[6], y2 = thresholds[13], color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[14],xmax], y1 = thresholds[6], y2 = thresholds[13], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [thresholds[4],thresholds[12]], y1 = thresholds[13], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[12],thresholds[19]], y1 = thresholds[13], y2 = thresholds[20], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [thresholds[12],thresholds[19]], y1 = thresholds[20], y2 = thresholds[18], color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[19],xmax], y1 = thresholds[13], y2 = thresholds[18], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [thresholds[12],thresholds[24]], y1 = thresholds[18], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax2.fill_between(x = [thresholds[24],xmax], y1 = thresholds[18], y2 = ymax, color = 'purple', alpha = 0.2)
    
    ax2.set_title('Tree Applied to Testing Data (Accuracy: {}%)'.format(test_acc))
    ax2.legend(loc = (1.02,0.525), ncol = 1)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    fig.tight_layout()
    
def plot_pca_transformation_tree_ex(gridsearch_pca, train, test, X_train, y_train, X_test, y_test):
    #get scaled and pca-transcormed data
    X_train_scaled = gridsearch_pca.best_estimator_.named_steps['scale'].transform(X_train)
    X_train_pca = gridsearch_pca.best_estimator_.named_steps['PCA'].transform(X_train_scaled)
    train_pca = pd.DataFrame(X_train_pca, columns = ['PC1','PC2']).reset_index(drop = True)
    train_pca['Class'] = np.array(y_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = ['X1','X2'])
    X_train_scaled_df['Class'] = np.array(y_train)
    X_test_scaled = gridsearch_pca.best_estimator_.named_steps['scale'].transform(X_test)
    X_test_pca = gridsearch_pca.best_estimator_.named_steps['PCA'].transform(X_test_scaled)
    test_pca = pd.DataFrame(X_test_pca, columns = ['PC1','PC2']).reset_index(drop = True)
    test_pca['Class'] = np.array(y_test)
    data_pca = pd.concat((train_pca,test_pca)).reset_index(drop = True)
    components = gridsearch_pca.best_estimator_.named_steps['PCA'].components_
    evrs = gridsearch_pca.best_estimator_.named_steps['PCA'].explained_variance_ratio_

    #plot transformed data
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (12,4))
    sns.scatterplot(x = 'X1', y = 'X2', hue = 'Class', hue_order = ['A','B'], s = 100,
                    data = train, palette = 'BuPu', alpha = 0.7, ax = ax1)
    ax1.set_title('Original Traning Data')
    ax1.legend().set_visible(False)
    sns.scatterplot(x = 'X1', y = 'X2', hue = 'Class', data = X_train_scaled_df, 
                        s = 100, alpha = 0.2, palette = 'BuPu', ax = ax2)
    ax2.arrow(0,0,components[0][0],components[0][1], color = 'black', alpha = 0.9,
              linewidth = 3, head_width = 0.1)
    ax2.text(components[0][0]+0.5,components[0][1]+0.30, size = 14,
             s = 'PC1 ({}%)'.format(round(100*evrs[0],1),1), ha = 'center')
    ax2.arrow(0,0,components[1][0],components[1][1], color = 'black', alpha = 0.9,
              linewidth = 3, head_width = 0.1)
    ax2.text(components[1][0]-0.5,components[1][1]+0.3, size = 14,
             s = 'PC2 ({}%)'.format(round(100*evrs[1],1)), ha = 'center')
    ax2.set_xlim(np.min(X_train_scaled)*1.2,np.max(X_train_scaled)*1.2)
    ax2.set_ylim(np.min(X_train_scaled)*1.2,np.max(X_train_scaled)*1.2)
    ax2.legend(loc = 'upper left')
    ax2.set_title('Scaled Data with PCA Coordinate System\nand Explained Variance Ratios')
    ax2.legend().set_visible(False)
    sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Class', hue_order = ['A','B'], s = 100,
                    data = train_pca, palette = 'BuPu', alpha = 0.7, ax = ax3)
    ax3.set_xlim(-3,3)
    ax3.set_ylim(-3,3)
    ax3.set_title('PCA Transformation of Traning Data')
    ax3.legend(loc = (1.02,0.77)).set_title('Class')
    fig.tight_layout()
    plt.show()
    
def plot_pca_model_ex(gridsearch_pca, X_train, y_train, X_test, y_test):
    #get scaled and pca-transcormed data
    X_train_scaled = gridsearch_pca.best_estimator_.named_steps['scale'].transform(X_train)
    X_train_pca = gridsearch_pca.best_estimator_.named_steps['PCA'].transform(X_train_scaled)
    train_pca = pd.DataFrame(X_train_pca, columns = ['PC1','PC2']).reset_index(drop = True)
    train_pca['Class'] = np.array(y_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = ['X1','X2'])
    X_train_scaled_df['Class'] = np.array(y_train)
    X_test_scaled = gridsearch_pca.best_estimator_.named_steps['scale'].transform(X_test)
    X_test_pca = gridsearch_pca.best_estimator_.named_steps['PCA'].transform(X_test_scaled)
    test_pca = pd.DataFrame(X_test_pca, columns = ['PC1','PC2']).reset_index(drop = True)
    test_pca['Class'] = np.array(y_test)
    data_pca = pd.concat((train_pca,test_pca)).reset_index(drop = True)
    components = gridsearch_pca.best_estimator_.named_steps['PCA'].components_
    evrs = gridsearch_pca.best_estimator_.named_steps['PCA'].explained_variance_ratio_
    #plot model with PCA
    features = gridsearch_pca.best_estimator_.named_steps['model'].tree_.feature
    thresholds = gridsearch_pca.best_estimator_.named_steps['model'].tree_.threshold
    y_pred = gridsearch_pca.predict(X_test)
    test_acc = round(100*np.mean(y_pred == y_test))
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,4), sharey = True)
    xmin,xmax = data_pca['PC1'].min()-1,data_pca['PC1'].max()+1
    ymin,ymax = data_pca['PC2'].min()-1,data_pca['PC2'].max()+1
    #plot model fitted to training data with boundary
    plot_train_tree_pca_ex(train_pca, ax = ax1, pca = True)
    ax1.fill_between(x = [xmin,xmax], y1 = ymin, y2 = thresholds[0], color = 'purple', alpha = 0.2)
    ax1.fill_between(x = [xmin,xmax], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax1.set_title('Tree Fitted to Training Data')
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    ax1.legend().set_visible(False)
    #plot model applied to testing data with boundary
    plot_test_tree_pca_ex(test_pca, y_pred = y_pred, ax = ax2, pca = True)
    ax2.fill_between(x = [xmin,xmax], y1 = ymin, y2 = thresholds[0], color = 'purple', alpha = 0.2)
    ax2.fill_between(x = [xmin,xmax], y1 = thresholds[0], y2 = ymax, color = 'steelblue', alpha = 0.2)
    ax2.set_title('Tree Applied to Testing Data (Accuracy: {}%)'.format(test_acc))
    ax2.legend(loc = (1.02,0.525), ncol = 1)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    fig.tight_layout()