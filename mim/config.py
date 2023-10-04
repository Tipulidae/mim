import os

ROOT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
PATH_TO_DATA = os.path.join(ROOT_PATH, 'data')
PATH_TO_TEST_RESULTS = os.path.join(PATH_TO_DATA, 'test_results')
PATH_TO_TF_LOGS = os.path.join(PATH_TO_DATA, 'tf_logs')
PATH_TO_TF_CHECKPOINTS = os.path.join(PATH_TO_DATA, 'tf_checkpoints')
# PATH_TO_CACHE = os.path.join(PATH_TO_DATA, 'cache')
PATH_TO_CACHE = '/projects/air-crypt/axel/cache'

GLUCOSE_ROOT = "/home/sapfo/andersb/PycharmProjects/" \
               "Expect/json_data/pontus_glukos"
