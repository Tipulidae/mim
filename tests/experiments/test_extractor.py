import pytest

import numpy as np
import pandas as pd
import tensorflow as tf

from mim.experiments.extractor import Data, Container, infer_shape, DataWrapper


class TestData:
    def test_lazy_slice(self):
        data = Data([1, 2, 3, 4])

        assert list(data) == [1, 2, 3, 4]
        assert list(data.lazy_slice([3, 2, 1])) == [4, 3, 2]
        assert list(data) == [1, 2, 3, 4]

    def test_slice_of_slice(self):
        data = Data([0, 1, 2, 3])
        assert list(data) == [0, 1, 2, 3]

        sliced = data.lazy_slice([3, 2, 1, 1, 1, 0])
        assert list(sliced) == [3, 2, 1, 1, 1, 0]

        sliced_again = sliced.lazy_slice([4, 3, 2])
        assert list(sliced_again) == [1, 1, 1]

    def test_lazy_slice_works_on_array(self):
        data = Data([[1, 2, 3], [2, 3, 4]], index=[0, 1])
        assert list(data.lazy_slice([0])) == [[1, 2, 3]]
        assert data[1] == [2, 3, 4]

    def test_split(self):
        data = Data([1, 2, 3, 4])

        x, y = data.split([1, 2], [3, 0])
        assert list(x) == [2, 3]
        assert list(y) == [4, 1]

    def test_as_numpy(self):
        data = Data([1, 2, 3, 4])
        assert np.array_equal(data.as_numpy(), np.array([1, 2, 3, 4]))

    def test_can_make_data_container_from_dict(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]
        data = Container.from_dict({'x': x, 'y': y})

        assert isinstance(data, Container)
        assert isinstance(data, Data)
        assert isinstance(data['x'], Data)
        assert isinstance(data['y'], Data)
        assert np.array_equal(data['x'].as_numpy(), np.array(x))

    def test_can_make_data_container_from_dict_of_data(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]

        data = Container.from_dict(
            {
                'x': Data(x, index=range(3)),
                'y': Data(y, index=range(3)),
            }
        )

        assert isinstance(data, Container)
        assert isinstance(data, Data)
        assert isinstance(data['x'], Data)
        assert isinstance(data['y'], Data)
        assert not isinstance(data['x'].data, Data)
        assert np.array_equal(data['x'].as_numpy(), np.array(x))

    def test_data_shape(self):
        data = Data([1, 2, 3, 4])
        assert data.shape == tf.TensorShape([])

        data = Data([[1, 2], [3, 4]])
        assert data.shape == tf.TensorShape([2])

    def test_container_shape(self):
        x = [[1, 2], [2, 3], [3, 4]]
        y = [0, 1, 0]
        data = Container.from_dict({'x': x, 'y': y})

        assert data.shape == {
            'x': tf.TensorShape([2]),
            'y': tf.TensorShape([])
        }

    def test_slice_container(self):
        large_data = list(range(500))
        large_index = [0, 100, 200, 300, 400]
        x = Data(large_data, index=large_index)
        y = Data([1, 1, 1, 0, 0])
        data = Container({'x': x, 'y': y}, index=[0, 1, 2, 3, 4])

        assert list(data) == [
            {'x': 0, 'y': 1},
            {'x': 100, 'y': 1},
            {'x': 200, 'y': 1},
            {'x': 300, 'y': 0},
            {'x': 400, 'y': 0},
        ]
        assert data[0] == {'x': 0, 'y': 1}
        assert data['x'][0] == 0
        assert data['y'][0] == 1

        sliced = data.lazy_slice([3, 1, 1])
        assert list(sliced) == [{'x': 300, 'y': 0},
                                {'x': 100, 'y': 1},
                                {'x': 100, 'y': 1}]
        assert list(sliced['x']) == [300, 100, 100]
        assert sliced[0] == {'x': 300, 'y': 0}
        assert sliced[1] == {'x': 100, 'y': 1}
        assert sliced[2] == {'x': 100, 'y': 1}
        assert sliced['x'][0] == 300

    def test_dataset_has_correct_type(self):
        pass

    def test_predefined_folds(self):
        x = Data([1, 2, 3, 4, 5], predefined_splits=[0, 0, 1, 1, 1])

        assert x.predefined_splits == [0, 0, 1, 1, 1]
        assert x.lazy_slice([3, 2, 1, 0]).predefined_splits == [1, 1, 0, 0]

    def test_flat_numpy_1d(self):
        x = Data([1, 2, 3, 4, 5])
        assert np.array_equal(
            x.as_flat_numpy(),
            np.array([[1], [2], [3], [4], [5]])
        )

    def test_flat_numpy_2d(self):
        x = Data([[1, 2, 3],
                  [2, 3, 4]])
        assert np.array_equal(
            x.as_flat_numpy(),
            np.array([[1, 2, 3], [2, 3, 4]])
        )

    def test_flat_numpy_3d(self):
        x = Data([[[1, 2, 3],
                   [2, 3, 4]],

                  [[3, 4, 5],
                   [4, 5, 6]]])
        assert np.array_equal(
            x.as_flat_numpy(),
            np.array([[1, 2, 3, 2, 3, 4], [3, 4, 5, 4, 5, 6]])
        )

    def test_flat_numpy_4d(self):
        x = Data([[[[1, 2, 3, 4],
                    [2, 3, 4, 5]],

                   [[3, 4, 5, 6],
                    [4, 5, 6, 7]]],

                  [[[4, 5, 6, 7],
                    [5, 6, 7, 8]],

                   [[6, 7, 8, 9],
                    [7, 8, 9, 10]]]]
                 )
        assert np.array_equal(
            x.as_flat_numpy(),
            np.array([[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7],
                      [4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10]])
        )

    def test_shuffled_dataset(self):
        unshuffled = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        shuffled_once = [1, 5, 2, 7, 4, 3, 10, 6, 8, 9]
        shuffled_twice = [6, 1, 3, 2, 8, 5, 10, 4, 9, 7]

        x = Data(unshuffled, fits_in_memory=False)
        unshuffled_dataset = x.as_dataset()
        shuffled_dataset = x.as_dataset(shuffle=True)

        assert list(x) == unshuffled
        assert list(unshuffled_dataset.as_numpy_iterator()) == unshuffled

        assert list(shuffled_dataset.as_numpy_iterator()) == shuffled_once
        assert list(shuffled_dataset.as_numpy_iterator()) == shuffled_twice

        assert list(x) == unshuffled
        assert list(unshuffled_dataset.as_numpy_iterator()) == unshuffled

        assert list(x.as_dataset(shuffle=True).as_numpy_iterator()) == \
               shuffled_once
        assert list(x.as_dataset(shuffle=True).as_numpy_iterator()) == \
               shuffled_once

    def test_dataset_shuffle_does_nothing_if_data_fits_in_memory(self):
        unshuffled = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        x = Data(unshuffled, fits_in_memory=True)
        not_actually_shuffled = x.as_dataset(shuffle=True)

        assert list(x) == unshuffled
        assert list(not_actually_shuffled.as_numpy_iterator()) == unshuffled


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


class TestContainer:
    def test_container_index_has_correct_length(self):
        data_dict = {
            'x': Data([1, 2, 3]),
        }
        container = Container(data_dict)
        assert len(container) == 3
        assert list(container.index) == [0, 1, 2]

    def test_making_container_with_different_lengths_raises_error(self):
        data_dict = {
            'x': Data([1, 2, 3]),
            'y': Data([1, 2, 3, 4])
        }
        with pytest.raises(ValueError):
            Container(data_dict)

    def test_container_without_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            Container([0])
        with pytest.raises(TypeError):
            Container('asdf')

    def test_attributes_are_not_reset_after_lazy_slice(self):
        data_dict = {
            'x': Data([1, 2, 3]),
            'y': Data([2, 3, 4])
        }
        container = Container(data_dict, fits_in_memory=False)

        assert container.fits_in_memory is False
        assert container.lazy_slice([1, 0, 0, 0]).fits_in_memory is False

        container = Container(data_dict, fits_in_memory=True)
        assert container.fits_in_memory is True
        assert container.lazy_slice([1, 0, 0, 0]).fits_in_memory is True

        container = Container(data_dict, predefined_splits=[-1, -1, 0, 0])
        assert container.predefined_splits == [-1, -1, 0, 0]
        assert container.lazy_slice([1, 2, 0]).predefined_splits == [-1, 0, -1]

    def test_flatten_single_item(self):
        data_dict = {
            'x': Data([1, 2, 3, 4]),
        }
        container = Container(data_dict)
        assert np.array_equal(
            container.as_flat_numpy(),
            np.array([[1], [2], [3], [4]])
        )

    def test_flatten_1d(self):
        data_dict = {
            'a': Data([1, 2, 3, 4]),
            'b': Data([2, 3, 4, 5])
        }
        container = Container(data_dict)
        assert np.array_equal(
            container.as_flat_numpy(),
            np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        )

    def test_flatten_2d(self):
        data_dict = {
            'a': Data([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
            'b': Data([[2, 3], [3, 4], [4, 5]]),
            'c': Data([1, 2, 3])
        }
        container = Container(data_dict)
        assert np.array_equal(
            container.as_flat_numpy(),
            np.array([[1, 2, 3, 4, 2, 3, 1],
                      [2, 3, 4, 5, 3, 4, 2],
                      [3, 4, 5, 6, 4, 5, 3]])
        )


class TestDataWrapper:
    def test_label_dataframe(self):
        features = Container(
            {
                'foo': Data([[1, 2, 3], [2, 3, 4]], columns=['a', 'b', 'c']),
                'bar': Data([3, 4], columns=['d'])
            },
        )
        labels = Data([0, 1], columns=['some_outcome'])
        index = Data(['auhwef', 'jio213'], columns=['patient_id'])
        data = DataWrapper(
            features=features,
            labels=labels,
            index=index,
        )

        expected = pd.DataFrame(
            [0, 1],
            columns=['some_outcome'],
            index=pd.Index(['auhwef', 'jio213'], name='patient_id')
        )
        assert data.y.equals(expected)
