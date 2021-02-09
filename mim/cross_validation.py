from mim.util.logs import get_logger

log = get_logger('Cross Validation')


# class CrossValidationWrapper:
#     def __init__(self, cv_type, **cv_args):
#         pass
#
#     def split(self, data: Container):
#         x = data.index
#         y = data['y'].as_numpy
#         groups = data.groups
#         for train, val in self.cv.split(x, y=y, groups=groups):
#             yield data.split(train, val)


# class DataProvider:
#
#     def train_val_split(self):
#         raise NotImplementedError
#

# class ChronologicalSplit:
#     def __init__(self, test_size=0.5):
#         self.test_size = test_size
#
#     def split(self, x, y=None, groups=None):
#         n = len(x)
#         k = int((1 - self.test_size) * n)
#         train = list(range(k))
#         test = list(range(k, n))
#         yield train, test
