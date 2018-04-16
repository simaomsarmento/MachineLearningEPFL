# Script which "produces exactly the same .csv predictions which you used in
# your best submission to the competition on Kaggle".

import numpy as np
from implementations import *
from helpers import *
from proj1_helpers import *
import pickle

# Getting back the objects:
with open('objs.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    w_rg, degree_rg = pickle.load(f)

with open('top_features.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    top_features, other_features = pickle.load(f)

with open('catNaNcol.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    cat0_NaNcol, cat1_NaNcol, cat2_NaNcol, cat3_NaNcol = pickle.load(f)

_, tx_predict, ids_predict = load_csv_data(
    '../data/test.csv', sub_sample=False)

### Data Analysis

# According to what was done previoulsy, we analyse test data and check it has the same format

### - Category analysis:

# From test data, define row indices for each category:

#22 is the column that indicates the category
cat_index = 22

cat0_row_indices = np.where(tx_predict[:, cat_index] == 0)[0]
cat1_row_indices = np.where(tx_predict[:, cat_index] == 1)[0]
cat2_row_indices = np.where(tx_predict[:, cat_index] == 2)[0]
cat3_row_indices = np.where(tx_predict[:, cat_index] == 3)[0]

# We verified, on background, that the type of NaN values distribution is fairly the same as for the training data.
# So, similarly, we "clean" the NaN values of the first column using the mean of the cleaned lines

# get cleaned lines and average them
cleaned_lines_index = np.where(tx_predict[:, 0] != -999)[0]
avg = np.mean(tx_predict[cleaned_lines_index, 0])

# replace with average value calculated above
NaN_lines_index = np.where(tx_predict[:, 0] == -999)[0]
tx_predict[NaN_lines_index, 0] = avg * np.ones(len(NaN_lines_index))

### Set Creation by category

# In this section we separate the data into categories and clean -999 columns except first column (only 15% NaN values)

tx_cat0 = tx_predict[cat0_row_indices, :]
tx_cat1 = tx_predict[cat1_row_indices, :]
tx_cat2 = tx_predict[cat2_row_indices, :]
tx_cat3 = tx_predict[cat3_row_indices, :]

# As done before, we merge category 2 and 3 due to their similarity.

tx_cat2 = tx_predict[np.where(tx_predict[:, cat_index] >= 2)[0]]

# **Define columns to delete**

# Now, we will also add the category column to the NaNcol set so that it is
# deleted next (not relevant) We verify that for category 0 the last column is
# also 0 always, so we eliminate it

cat0_toDelete = np.hstack((cat0_NaNcol, cat_index, np.array([29])))
cat1_toDelete = np.hstack((cat1_NaNcol, cat_index))
cat2_toDelete = np.array([cat_index])

### Data cleaning

# Delete NaN columns for each tx set:

tx_cat0 = np.delete(tx_cat0, cat0_toDelete, 1)
tx_cat1 = np.delete(tx_cat1, cat1_toDelete, 1)
tx_cat2 = np.delete(tx_cat2, cat2_toDelete, 1)

tx_cat = [tx_cat0, tx_cat1, tx_cat2]

## Prediction

# Standardize Data
for i in range(len(tx_cat)):
    tx_cat[i], _, _ = standardize(tx_cat[i])

### Multinomial Expansion for each Category

# According to ridge regression, build multinomial for degree corresponding to minimal loss
phi_predict = []
for i in range(len(tx_cat)):
    phi_predict.append(
        build_multinomial(tx_cat[i], degree_rg[i], top_features[i],
                          other_features[i]))
print('Phi predict', np.shape(phi_predict[1]))

### Create predictions

y_predict = np.zeros(tx_predict.shape[0])

category_col = tx_predict[:, cat_index]
# Merge categories 2 and 3 into one single category (2)
cat3_idx = np.where(category_col == 3)
category_col[cat3_idx] = 2 * np.ones(len(cat3_idx))

for i in range(len(tx_cat)):
    y_predict[np.where(category_col == i)[0]] = predict_labels(
        w_rg[i], phi_predict[i])

### Export CSV file

create_csv_submission(ids_predict, y_predict, 'submission.csv')
