import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helpers import *

##################### Update User Features ####################

def update_column_user(user, user_nz, nnz_items_per_user, M, lambda_I, user_features_new):
    """
    Auxiliar function to update one column of user matrix
    Update user feature matrix, for fixed item matrix
    ***INPUT***
    :param user: user to update
    :param user_nz: train values for user
    :param nnz_items_per_user: non zero items per each user
    :param M: item matrix
    :param lambda_I: regularization parameter
    :param user_features_new: matrix to update
    ***OUTPUT***
    :return user_features_new
    """
    g = user_nz  #- bias_item_nz - bias_user[user]
    b = M @ g
    A = M @ M.T + nnz_items_per_user[user] * lambda_I 
    user_features_new[:, user] = np.linalg.solve(A, b) 
    
    return user_features_new


def update_user_bias(user, user_nz, bias_item_nz, user_features_new, nnz_items_per_user,
                     M, bias_user_new, lambda_bias_user):
    """
    Updates bias for a fixed user.
    ***INPUT***
    :param user: user to update
    :param user_nz: train values for user
    :param bias_item_nz: train values for bias
    :param user_features_new: user matrix
    :param nnz_items_per_user: non zero items per each user
    :param M: item matrix
    :param bias_user_new: bias vector to update
    :param lambda_bias_user: regularization parameter
     ***OUTPUT***
    :return bias_user_new
    """
    y = user_nz - bias_item_nz
    aux = np.sum(y - M.T @ user_features_new[:, user]) 
    bias_user_new[user] = aux / (nnz_items_per_user[user] + lambda_bias_user) 
    
    return bias_user_new


##################### Update Item Features ####################

def update_column_item(item, item_nz, nnz_users_per_item, M,lambda_I,
                       item_features_new):
    """
    Auxiliar function to update one column of user matrix
    Update item feature matrix, for fixed item matrix
    ***INPUT***
    :param item: item to update
    :param item_nz: train values for item
    :param nnz_users_per_item: non zero users per each item
    :param M: users matrix
    :param lambda_I: regularization parameter
    :param item_features_new: matrix to update
    ***OUTPUT***
    :return item_features_new
    """
    g = item_nz #- bias_user_nz - bias_item[item]
    b = M @ g
    A = M @ M.T + nnz_users_per_item[item] * lambda_I
    item_features_new[:, item] = np.linalg.solve(A, b)
    
    return item_features_new


def update_item_bias(item, item_nz, bias_user_nz, item_features_new,
                     nnz_users_per_item, M, bias_item_new, lambda_bias_item):
    """
    Updates bias for item
    ***INPUT***
    :param item: item to update
    :param item_nz: train values for item
    :param bias_user_nz: train values for bias
    :param item_features_new: item's matrix
    :param nnz_users_per_item: non zero users per each item
    :param M: user's matrix
    :param bias_item_new: bias vector to update
    :param lambda_bias_item: regularization parameter
    ***OUTPUT***
    :return bias_item_new
    """
    y = item_nz - bias_user_nz
    aux = np.sum(y - M.T @ item_features_new[:, item])
    bias_item_new[item] =  aux / (nnz_users_per_item[item] + lambda_bias_item)
    
    return bias_item_new


##################### Auxiliar ALS functions ###################

def initialize_parameters(train, num_features, seed):
    """
    Initializes parameters tu use in ALS
    ***INPUT***
    :param train: data for train
    :param num_features: number of features for Matrix factorization
    :param seed: seed for random generator
    ***OUTPUT***
    :return user_features: user's matrix
    :return item_features: item's matrix
    :return bias_user: user's bias vector
    :return bias_item: item's bias vector
    """
    # set seed
    np.random.seed(seed)  #np.random.seed(84)
    
    # initialize biases
    num_items, num_users = train.shape
    bias_item = np.zeros(num_items) #np.random.rand(num_items) - 0.5
    bias_user = np.zeros(num_users) #np.random.rand(num_users) - 0.5

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    return user_features, item_features, bias_item, bias_user


def get_non_zero_values(train):
    """
    Get non zero values, as well as indices
    ***INPUT**
    :param train: data for train
    :return nz_row: non zero rows on train data
    ***OUTPUT***
    :return nz_col: non zero cols on train data
    :return nz_train: tuple of non zero (row, col)
    :return nnz_users_per_item: # of non zero users per item
    :return nz_item_userindices: indices of non zero items per user
    :return nnz_items_per_user: # of non zero items per user
    :return nz_user_itemindices: indices of non zero users per item
    """
    # find the non-zero ratings indices for the train
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    # self.rows is a sorted list of the user indices corresponding to 
    # non-zero ratings, for each item (row) 
    nnz_users_per_item = [len(row) for row in train.rows]
    nz_item_userindices = train.rows

    # self.T.rows is a sorted list of the item indices corresponding 
    # to non-zero ratings, for each column (user)
    nnz_items_per_user = [len(col) for col in train.transpose().rows]
    nz_user_itemindices = train.transpose().rows
    
    return nz_row, nz_col, nz_train,nnz_users_per_item, nz_item_userindices, nnz_items_per_user, nz_user_itemindices

def compute_error_ALS(data, user_features, item_features, bias_user, bias_item, nz):
    """
    Compute the loss (RMSE) of the prediction of nonzero elements.
    Each entry of the prediction vector has the inner product of the corresponding 
    item and user vectors for each nonzero entry data + biases of the
    corresponding item and user. In ALS the biases represent the mean scores for each
    each item and each user and no the deviation from the mean
    ***INPUT***
    :param data : original data
    :param user_features : user's matrix
    :param item_features : item's matrix
    :param bias_user : bias for users
    :param bias_item : bias for items
    :param nz : non zero elements on data
    ***OUTPUT***
    :return rmse: returns root mean square
    """
    nz_row, nz_col = zip(*nz)
    
    prediction = np.sum(item_features[:, nz_row] * user_features[:, nz_col], axis=0) + \
                 bias_item[list(nz_row)] + bias_user[list(nz_col)] 
    
    error = prediction - data[nz_row, nz_col].toarray()[0]
    rmse = np.sqrt(np.mean(np.square(error)))
    
    return rmse

def evaluate(test, user_features, item_features, bias_user, bias_item):
    """
    Evaluate ALS performance on test data
    ***INPUT***
    :param test: data to perform evaluation on
    :param user_features: user's matrix
    :param item_features: item's matrix
    :param bias_user: vector of user's bias
    :param bias_item: vector of item's bias
    ***OUTPUT***
    :return rmse: calculated rmse error
    """
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    rmse = compute_error_ALS(test, user_features, item_features, 
                             bias_user, bias_item, nz_test)
    print("test RMSE after running ALS: {v}.".format(v=rmse))
    
    return rmse


##################### Others ###################

def grid_search(train, test):
    """
    Run a grid search on the parameters specifie below
    Returns the minimium error and corresponding parameters
    ***INPUT***
    :param train: train data
    :param test: test data
    ***OUTPUT***
    :return user_features_min:
    :return item_features_min:
    :return bias_user_min:
    :return bias_item_min:
    :return min_parameters: list that contains the parameters corresponding
                            to minimium rmse: [0]- rmse
                                              [1]- # of features
                                              [2]- lambda_ (1- user, 2- item)
                                              [3]- lambda_bias (1-user, 2-item)
                                        
    """
    
    # define parameters to grid search:
    num_features = [20] #[20, 25, 30, 35, 40, 45]
    lambda_ = [[0.06,0.14]]#, [0.06, 0.42], [0.015, 0.015], [0.14,0.06]]
    lambda_bias = [[0.1, 0.1]]#[0.02, 0.02]]
    stop_criterion = 1e-2
    seed = 132
    min_error = 100 #initialization with high value
    
    for n_features in num_features:
        for lambda_i in lambda_:
            for lambda_bias_i in lambda_bias:
                print("Num_features: ", n_features)
                print("Lambda_user: ", lambda_i[0], "\nLambda_item  ", lambda_i[1] )
                print("Bias User: ", lambda_bias_i[0],'\nBias Item:', lambda_bias_i[1])
                
                # ALS
                user_features, item_features, bias_user, bias_item, rmse = ALS_bias(
                    train, test, n_features, lambda_i[0], lambda_i[1], lambda_bias_i[0],
                    lambda_bias_i[1], stop_criterion, seed)
                
                print("ERROR: ", rmse, '\n')
                
                # Update min
                if rmse < min_error:
                    #save parameters                    
                    min_parameters=[]
                    min_parameters.extend((rmse, n_features,lambda_i,lambda_bias_i))

                    #save matrix
                    user_features_min = user_features
                    item_features_min = item_features
                    bias_user_min =  bias_user
                    bias_item_min = bias_item
    
    return user_features_min, item_features_min, bias_user_min, bias_item_min, min_parameters

