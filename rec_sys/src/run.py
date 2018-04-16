import pickle
from helpers import samples_csv_submission, create_csv_submission


with open('model.pickle', 'rb') as f:
    item_features, user_features, bias_item, bias_user = pickle.load(f)

SUBMISSION_SAMPLES_PATH = "./Data/sample_submission.csv"
samples_submission      = samples_csv_submission(SUBMISSION_SAMPLES_PATH)

create_csv_submission(samples_submission,
                      item_features,
                      user_features,
                      bias_item,
                      bias_user,
                      'submission_run.csv')
