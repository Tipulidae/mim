from os.path import normpath, join, dirname

ROOT_PATH = normpath(join(dirname(__file__), '..'))
PATH_TO_DATA = join(ROOT_PATH, 'data')
PATH_TO_TEST_RESULTS = join(PATH_TO_DATA, 'test_results')
PATH_TO_TF_LOGS = join(PATH_TO_DATA, 'tf_logs')
PATH_TO_TF_CHECKPOINTS = join(PATH_TO_DATA, 'tf_checkpoints')
PATH_TO_CACHE = join(PATH_TO_DATA, 'cache')

GLUCOSE_ROOT = "/home/sapfo/andersb/PycharmProjects/" \
               "Expect/json_data/pontus_glukos"
