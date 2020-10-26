import numpy as np
import tensorflow as tf

from mim.extractors.extractor import Data, Container, infer_shape


class TestData:
    def test_lazy_slice(self):
        data = Data([1, 2, 3, 4])

        assert list(data) == [1, 2, 3, 4]
        assert data.index == range(0, 4)
        assert list(data.lazy_slice([3, 2, 1])) == [4, 3, 2]
        assert data.index == range(0, 4)

    def test_lazy_slice_works_on_array(self):
        data = Data([[1, 2, 3], [2, 3, 4]], index=[0, 1, 2])
        assert list(data.lazy_slice([0])) == [[1, 2, 3]]
        assert data[1] == [2, 3, 4]

    def test_split(self):
        data = Data([1, 2, 3, 4])

        x, y = data.split([1, 2], [3, 0])
        assert list(x) == [2, 3]
        assert list(y) == [4, 1]

    def test_as_numpy(self):
        data = Data([1, 2, 3, 4])
        assert np.array_equal(data.as_numpy, np.array([1, 2, 3, 4]))

    def test_can_make_data_container_from_dict(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]
        data = Container.from_dict({'x': x, 'y': y}, index=range(3))

        assert isinstance(data, Container)
        assert isinstance(data, Data)
        assert isinstance(data['x'], Data)
        assert isinstance(data['y'], Data)
        assert np.array_equal(data['x'].as_numpy, np.array(x))

    def test_can_make_data_container_from_dict_of_data(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]

        data = Container.from_dict(
            {
                'x': Data(x, index=range(3), shape=[2]),
                'y': Data(y, index=range(3), shape=[0]),
            },
            index=range(3)
        )

        assert isinstance(data, Container)
        assert isinstance(data, Data)
        assert isinstance(data['x'], Data)
        assert isinstance(data['y'], Data)
        assert not isinstance(data['x'].data, Data)
        assert np.array_equal(data['x'].as_numpy, np.array(x))

    def test_data_shape(self):
        data = Data([1, 2, 3, 4])
        assert data.shape == tf.TensorShape([])

        data = Data([[1, 2], [3, 4]])
        assert data.shape == tf.TensorShape([2])

    def test_container_shape(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]
        data = Container.from_dict({'x': x, 'y': y}, index=range(3))

        assert data.shape == {
            'x': tf.TensorShape([2]),
            'y': tf.TensorShape([])
        }


class TestInferShape:
    def test_shape_ignores_first_dimension(self):
        data = [[1, 2, 3], [2, 3, 4]]
        assert infer_shape(data) == [3]

        data = [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                [[2, 3, 4], [3, 4, 5], [5, 6, 7]]]
        assert infer_shape(data) == [3, 3]

    def test_shape_of_list_is_empty_list(self):
        assert infer_shape([1, 2, 3]) == []

    def test_shape_of_scalar_is_none(self):
        assert infer_shape(3) is None
