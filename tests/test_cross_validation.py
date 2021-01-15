from mim.cross_validation import ChronologicalSplit


class TestChronologicalSplit:
    def test_split_5050(self):
        cs = ChronologicalSplit()
        x = list(range(10))

        train, test = next(cs.split(x))
        assert train == [0, 1, 2, 3, 4]
        assert test == [5, 6, 7, 8, 9]

    def test_split_8020(self):
        cs = ChronologicalSplit(test_size=0.20)
        x = list(range(10))

        train, test = next(cs.split(x))
        assert train == [0, 1, 2, 3, 4, 5, 6, 7]
        assert test == [8, 9]

    def test_split_all_train(self):
        cs = ChronologicalSplit(test_size=0)
        x = list(range(10))

        train, test = next(cs.split(x))
        assert train == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert test == []
