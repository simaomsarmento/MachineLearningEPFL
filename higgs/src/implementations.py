'''File which contains all function implementations from table 1
of step 2 as well as auxiliar functions used in the project'''

#Necessary imports:
import numpy as np
import itertools
from helpers import batch_iter
from proj1_helpers import predict_labels


#Cost functions
def compute_loss_MSE(y, tx, w):
    e = y - tx.dot(w) ;
    return (1/(2*len(y)))*e.dot(e)

def loss_accuracy(y, tx, w):
    y_predict = predict_labels(w, tx)
    loss = len(np.where(y_predict != y)[0])/len(y)
    return loss


#GRADIENT DESCENT
def compute_gradient(y, tx, w):
    """Compute the gradient of mse function."""
    e = y - tx.dot(w) ;
    gradient = -(1/len(y))*tx.T.dot(e)
    
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):

        # compute gradient
        gradient = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma*gradient
        
    return w, compute_loss_MSE(y, tx, w)


#STOCHASTIC GRADIENT DESCENT
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w    
    for n_iter in range(max_iters):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):         
       
            # compute gradient and loss
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)

            # update w by gradient
            w = w - gamma*gradient

    return w, compute_loss_MSE(y, tx, w)

#LEAST SQUARES
def least_squares(y, tx):

    # calculate w
    A = np.dot(tx.T, tx) #Gram Matrix
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)
    
    # calculate loss
    loss = compute_loss_MSE(y, tx, w)
    
    return w, loss

#RIDGE REGRESSION
def ridge_regression(y, tx, lambda_):

    # calculate w
    lambda_aux = lambda_ * (2*len(y)) 
    A = np.dot(tx.T, tx) + lambda_aux*np.eye(tx.shape[1])    
    b = np.dot(tx.T, y) 
    w = np.linalg.solve(A, b)
    
    # calculate loss
    loss = compute_loss_MSE(y, tx, w)
   
    return w, loss

#necessary functions for MULTINOMIAL EXPANSION
def multinomial_partitions(n, k):
    """returns an array of length k sequences of integer partitions of n"""
    nparts = itertools.combinations(range(1, n+k), k-1)
    tmp = [(0,) + p + (n+k,) for p  in nparts]
    sequences =  np.diff(tmp) - 1
    return sequences[::-1] # reverse the order

def build_multinomial_crossterms(tx, degree) :
    '''Make multinomial feature matrix'''
    
    order= np.arange(degree)+1
    Xtmp = np.ones_like(tx[:,0])
    for ord in order :
        if ord==1 :
            fstmp = tx
        else :
            pwrs = multinomial_partitions(ord,tx.shape[1])
            fstmp = np.column_stack( ( np.prod(tx**pwrs[i,:], axis=1) for i in range(pwrs.shape[0]) ))

        Xtmp = np.column_stack((Xtmp,fstmp))

    return Xtmp


def build_poly(x, degree):
    # polynomial basis function: 
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    
    phi = np.vander(x, N = degree+1, increasing = True)

    return phi

def build_multinomial(tx, degree, important_features, other_features):
    #build usual polinomial
    poly_other = []
    for feature in other_features:
        poly_other.append(build_poly(tx[:, feature], degree)[:,1:])   
    poly_other = np.concatenate(poly_other, axis = 1)  
      
    #build polinomial with cross terms as well for important features
    poly_important = build_multinomial_crossterms(tx[:,important_features], degree)
    
    poly = np.column_stack((poly_important, poly_other)) 
    
    return poly


#Cross Validation:

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    #Vector with indices vectors
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree, multinomial):

    # get k'th subgroup in test, others in train: 
    y_test = y[k_indices[k]]
    y_train = np.delete(y, k_indices[k])
    
    # form data with polynomial degree: 
    phi_test = multinomial[k_indices[k],:]
    phi_train = np.delete(multinomial, k_indices[k], 0)

    w_rg, _ = ridge_regression(y_train, phi_train, lambda_)
    loss_te = loss_accuracy(y_test, phi_test, w_rg)

    return w_rg, loss_te


#LOGISTIC REGRESSION USING GD

#auxiliar functions
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    # compute the cost: 
    loss = calculate_loss(y, tx, w)
    
    # compute the gradient:
    gradient = calculate_gradient(y, tx, w)

    # update w: 
    w = w - gamma*gradient
  
    return loss, w

def sigmoid(t):
    """apply sigmoid function on t."""
    num = np.exp(t)
    den = 1 + np.exp(t)
    
    return num/den

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    t  = tx.dot(w)
    loss = (np.sum(np.log(1+np.exp(t))) - y.T.dot(t))[0][0]
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    
    gradient = (tx.T).dot(sigmoid(tx.dot(w))-y)
    return gradient

#applies logistic regressin
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss



#REGULARIZED LOGISTIC REGRESSION USING GD

#auxiliar functions
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient: 
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)

    # update w: 
    w = w - gamma*gradient
    
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
   
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w) + lambda_*w
    
    return loss, gradient

#applies regularized logistic regression
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    #set tresh hold
    threshold = 1e-8
    losses = []

    w = initial_w
    
    # start the penalized logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss  