import mim.experiments.hyper_parameter as hp


class FakeRandom:
    def randint(self, a, b):
        return a

    def uniform(self, a, b):
        return a

    def choice(self, values):
        return values[0]

    def choices(self, values, k=1):
        return [values[0] for _ in range(k)]


def test_choice():
    choice = hp.Choice([1, 2, 14])
    assert choice.pick(FakeRandom()) == 1


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
            {'foo': 3.4, 'bar': 99},
            {'foo': 3.4, 'bar': 99}
        ]
    }

    choices = hp.pick(search_space, FakeRandom())
    assert choices == expected

# def test_param():
#     # params = {'foo': Choice(['a', 'b'])}
#     r = None
#     param = Choice(['a', 'b'])
#     assert param.pick(r) in ['a', 'b']
#
#     param = Choice([
#         Choice([1, 2, 3]),
#         Choice([2, 3, 4])
#     ])
#     assert param.pick(r) == 1
#
#     param = Choice([
#         Choice([1, 2, 3]),
#         4,
#         5,
#     ])
#     assert param.pick(r) == 1
#
#     pick(param)


# from copy import deepcopy
#
# from mim.experiments.random_search import randomize
#
#
# def test_randomize_base_case():
#     assert randomize("foo") == "foo"
#     assert randomize(3) == 3
#     assert randomize([]) is None
#
#
# def test_randomize_list():
#     assert randomize([1, 2, 3]) in [1, 2, 3]
#     assert randomize([1]) == 1
#
#
# def test_randomize_dict():
#     assert randomize({'foo': [1, 2, 3]}) == {'foo': 1}
#
#     inp = {'foo': 32, 'bar': {'baz': 33, 'foobar': ['a', 'b']}}
#     inp_copy = deepcopy(inp)
#     assert randomize(inp) == {'foo': 32, 'bar': {'baz': 33, 'foobar': 'a'}}
#     assert inp == inp_copy
#
#
# # def test_foo():
# #     template = {
# #         'algorithm': 'foo',
# #         'params': {
# #             'model': 'bar',
# #             'num_conv_layers': [2, 3, 4],
# #             'input_shape': {'ecg': (1200, 8)}
# #         }
# #     }
# #     # randomize = {
# #     #     'params': {
# #     #         'num_conv_layers': [2, 3, 4]
# #     #     }
# #     # }
# #
# #     randomize = {
# #         'algorithm': ['foo', 'bar']
# #     }
# #
# #     searcher = RandomSearch(template, random_state=42)
# #     xp = next(searcher)
# #     assert isinstance(xp, Experiment)
# #     assert xp.algorithm == 'foo'
# #     assert xp.params['model'] == 'bar'
# #     assert xp.params['num_conv_layers'] in [2, 3, 4]
