from mim.util.util import ranksort


class TestRanksort:
    def test_small(self):
        assert ranksort([1, 2, 3, 4]) == [0, 1, 2, 3]
        assert ranksort([2, 1, 3, 4]) == [1, 0, 2, 3]
        assert ranksort([5, 1, 0, 3]) == [3, 1, 0, 2]

    def test_descending(self):
        assert ranksort([1, 2, 3, 4], ascending=False) == [3, 2, 1, 0]
        assert ranksort([2, 1, 3, 4], ascending=False) == [2, 3, 1, 0]
        assert ranksort([5, 1, 0, 3], ascending=False) == [0, 2, 3, 1]
