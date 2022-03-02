import mim.experiments.hyper_parameter as hp


class FakeRandom:
    def randint(self, a, b, step=1):
        return a

    def randrange(self, a, b, step=1):
        return a

    def uniform(self, a, b):
        return a

    def choice(self, values):
        return values[0]

    def choices(self, values, k=1):
        n = len(values)
        return [values[i % n] for i in range(k)]
        # return [values[0] for _ in range(k)]


def test_choice():
    choice = hp.Choice([1, 2, 14])
    assert choice.pick(FakeRandom()) == 1


def test_choices():
    choices = hp.Choices(
        [3, 1, 2, 4, 0, 5],
        k=3,
    )
    assert choices.pick(FakeRandom()) == [3, 1, 2]


def test_sorted_choices():
    vals = [3, 1, 2, 4, 0, 5]
    sorted_choices = hp.SortedChoices(vals, k=3, ascending=True)
    assert sorted_choices.pick(FakeRandom()) == [1, 2, 3]

    sorted_choices = hp.SortedChoices(vals, k=5, ascending=False)
    assert sorted_choices.pick(FakeRandom()) == [4, 3, 2, 1, 0]


def test_pick():
    search_space = {
        'num_layers': hp.Choice([2, 3, 4]),
        'dropout': hp.Float(0.1, 0.8),
        'something_static': 32,
        'hidden': hp.Choices(
            [
                {'foo': hp.Float(3.4, 4.4), 'bar': 99},
                {'foo': hp.Float(5.5, 4.3), 'bar': hp.Int(3, 10)}
            ],
            k=hp.Int(3, 10)
        ),
    }

    expected = {
        'num_layers': 2,
        'dropout': 0.1,
        'something_static': 32,
        'hidden': [
            {'foo': 3.4, 'bar': 99},
            {'foo': 5.5, 'bar': 3},
            {'foo': 3.4, 'bar': 99}
        ]
    }

    choices = hp.pick(search_space, FakeRandom())
    assert choices == expected


def test_flatten_dict():
    choices = {
        'num_layers': 2,
        'dropout': 0.1,
        'something_static': 32,
        'hidden': [
            {'foo': 3.4, 'bar': 99},
            {'foo': 3.4, 'bar': 99},
            {'foo': 3.4, 'bar': 99}
        ]
    }

    flattened = {
        'num_layers': 2,
        'dropout': 0.1,
        'something_static': 32,
        'hidden_0_foo': 3.4,
        'hidden_0_bar': 99,
        'hidden_1_foo': 3.4,
        'hidden_1_bar': 99,
        'hidden_2_foo': 3.4,
        'hidden_2_bar': 99
    }

    assert hp.flatten(choices) == flattened


def test_flatten_complicated_nesting():
    choices = {
        'a': [1, 2, 3],
        'b': [{'c': 1, 'd': 2, 'e': [1, 2, 3], 'f': {'g': 4, 'h': 5}},
              {'x': {'y': {'z': [1, 2, 3]}, 'w': 32}},
              [4, 5, 6],
              [{'foo': 0, 'bar': -1}]
              ],
        'c': 34
    }

    flattened = {
        'a_0': 1,
        'a_1': 2,
        'a_2': 3,
        'b_0_c': 1,
        'b_0_d': 2,
        'b_0_e_0': 1,
        'b_0_e_1': 2,
        'b_0_e_2': 3,
        'b_0_f_g': 4,
        'b_0_f_h': 5,
        'b_1_x_y_z_0': 1,
        'b_1_x_y_z_1': 2,
        'b_1_x_y_z_2': 3,
        'b_1_x_w': 32,
        'b_2_0': 4,
        'b_2_1': 5,
        'b_2_2': 6,
        'b_3_0_foo': 0,
        'b_3_0_bar': -1,
        'c': 34
    }

    assert hp.flatten(choices) == flattened
