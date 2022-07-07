from mim.massage.sk1718 import last_in_clusters


class TestLastInClusters:
    def test_simple(self):
        inp = [1, 2, 3, 61, 62, 63]
        assert last_in_clusters(inp) == [3, 63]

    def test_empty(self):
        assert last_in_clusters([]) == []

    def test_single_item_in_list(self):
        assert last_in_clusters([1]) == [1]

    def test_unsorted_simple(self):
        inp = [3, 2, 1, 63, 61, 62]
        assert last_in_clusters(inp) == [3, 63]

    def test_no_gaps(self):
        inp = [1, 2, 3, 4, 5]
        assert last_in_clusters(inp) == [5]

    def test_real_example1(self):
        inp = [78., 8., 8., 8., 8., 8., 88., 8., 8., 88., 88., 8., 88.,
               8., 8., 88., 8., 88., 88., 8., 88., 8., 88., 88., 88., 88.,
               8., 88., 8., 8., 8., 88., 8., 88., 88., 8., 88., 8.]
        assert last_in_clusters(inp) == [8.0, 88.0]

    def test_real_example2(self):
        inp = [24., 24., 84., 94., 24., 84., 24., 84., 84., 84., 24.,
               24., 24., 84., 24., 24., 24., 84., 24., 84., 84., 24.,
               84., 24., 194., 84., 24., 94., 84., 84., 84., 24., 84.,
               84., 84.]
        assert last_in_clusters(inp) == [24.0, 94.0, 194.0]
