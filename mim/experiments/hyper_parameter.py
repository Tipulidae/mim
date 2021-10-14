class Param:
    def pick(self, generator):
        raise NotImplementedError


class Int(Param):
    def __init__(self, min_value, max_value, step=1):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def pick(self, generator):
        return generator.randrange(
            pick(self.min_value, generator),
            pick(self.max_value, generator) + 1,
            step=pick(self.step, generator)
        )

    def __str__(self):
        return f"Int({self.min_value}, {self.max_value})"

    def __repr__(self):
        return f"Int({self.min_value}, {self.max_value})"


class Float(Param):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def pick(self, generator):
        return generator.uniform(
            pick(self.min_value, generator),
            pick(self.max_value, generator)
        )

    def __str__(self):
        return f"Float({self.min_value}, {self.max_value})"

    def __repr__(self):
        return f"Float({self.min_value}, {self.max_value})"


class Choice(Param):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def pick(self, generator):
        return pick(generator.choice(self.data), generator)

    def __str__(self):
        return f"Choice({self.data})"

    def __repr__(self):
        return f"Choice({self.data})"


class Choices(Param):
    def __init__(self, data, k=1):
        super().__init__()
        self.data = data
        self.k = k

    def pick(self, generator):
        k = pick(self.k, generator)
        return pick(generator.choices(self.data, k=k), generator)

    def __str__(self):
        return f"Choices({self.data}, k={self.k})"

    def __repr__(self):
        return f"Choices({self.data}, k={self.k})"


class SortedChoices(Param):
    def __init__(self, data, k=1, ascending=True):
        super().__init__()
        self.data = data
        self.k = k
        self.ascending = ascending

    def pick(self, generator):
        k = pick(self.k, generator)
        return pick(
            sorted(
                generator.choices(self.data, k=k),
                reverse=not self.ascending
            ),
            generator
        )

    def __str__(self):
        return f"Choices({self.data}, k={self.k})"

    def __repr__(self):
        return f"Choices({self.data}, k={self.k})"


def pick(search_space, r):
    if isinstance(search_space, Param):
        return search_space.pick(r)
    if isinstance(search_space, dict):
        return {k: pick(v, r) for k, v in search_space.items()}
    if isinstance(search_space, list):
        return [pick(x, r) for x in search_space]
    else:
        return search_space


def flatten(nested_dict):
    def _flatten(name, item):
        if isinstance(item, dict):
            return flatten({f"{name}_{k}": v for k, v in item.items()})
        elif isinstance(item, list):
            return flatten({f"{name}_{i}": v for i, v in enumerate(item)})
        else:
            return {name: item}

    return merge([_flatten(k, v) for k, v in nested_dict.items()])


def merge(list_of_dicts):
    result = {}
    for dictionary in list_of_dicts:
        result = {**result, **dictionary}

    return result
