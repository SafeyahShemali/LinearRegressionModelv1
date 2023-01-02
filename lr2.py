import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D


def main():
    '''get the data and results files'''
    data_file = str(sys.argv[1])
    result_file = str(sys.argv[2])
    file1 = open(result_file, "w")  # append mode

    ''' Preprocessing Input data'''
    '''get the training data X'''
    X = pd.read_csv(data_file, sep=',', names=['age','weight','height'])
    X = pd.DataFrame(X, columns=['age','weight','height'])
    X.insert(0 ,'ones', np.ones_like(X['age']))

    # add a vector column for the intercept at the front of your matrix
    # normalize the data
    age_mean = X['age'].mean()
    age_std = X['age'].std()

    weight_mean = X['weight'].mean()
    weight_std = X['weight'].std()

    X['age'] = (X['age'] - age_mean) / age_std
    X['weight'] = (X['weight'] - weight_mean) / weight_std
    
    X_copy = X
    X = X.to_numpy()

    # Set alpha's and # of iteration from Y
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10 , 0.7]
    iterations = [100,100,100,100,100,100,100,100,100,10]
    #iterations = 100

    ''' Building and training the model  '''
    for ii in range(len(alphas)):
        alpha = alphas[ii]
        b_1 = 0.0
        b_2 = 0.0
        b_0 = 0.0
        n = X.shape[0]  # Number of elements in X

        for itr in range(iterations[ii]):
            R = 0.0  # error
            diff_sequare = 0.0
            diff_sum_b_1 = 0.0
            diff_sum_b_2 = 0.0
            diff_sum_b_0 = 0.0

            for i in range(n):
                # Prediction of training data X using the linear regression equation
                fx = (b_0 * X[i,0]) + (b_1 * X[i,1]) + (b_2 * X[i,2])

                diff_sum_b_0 = diff_sum_b_0 + (fx-X[i,3])*X[i,0]
                diff_sum_b_1 = diff_sum_b_1 + (fx-X[i,3])*X[i,1]
                diff_sum_b_2 = diff_sum_b_2 + (fx-X[i,3])*X[i,2]
                diff_sequare = diff_sequare + ((fx-X[i,3])**2)

            # update the parameters
            b_0 = b_0 - (alpha/n)*diff_sum_b_0
            b_1 = b_1 - (alpha/n)*diff_sum_b_1
            b_2 = b_2 - (alpha/n)*diff_sum_b_2

            # Calculate the loss R
            R = (1 / (2 * n)) * diff_sequare


        visualize_3d(X_copy, lin_reg_weights=[b_0,b_1,b_2], feat1='age', feat2='weight', labels='height',
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=alpha, xlabel='age', ylabel='weight', zlabel='height',
                 title='')

        file1.write("{},{},{},{},{}\n".format(alpha,iterations[ii],b_0,b_1,b_2))
        print(alpha, R)

def visualize_3d(df, lin_reg_weights=[1,1,1], feat1=0, feat2=1, labels=2,
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
   
    '''Credit to Kelsey D'Souza'''
    # Setup 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection ='3d')


    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array( [lin_reg_weights[0] +
                       lin_reg_weights[1]*f1 +
                       lin_reg_weights[2]*f2  # height
                       for f1, f2 in zip(axes1, axes2)] )
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":
    main()