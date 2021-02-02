class Param:
    def pick(self, generator):
        raise NotImplementedError


class Int(Param):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def pick(self, generator):
        return generator.randint(
            pick(self.min_value, generator),
            pick(self.max_value, generator)
        )


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


class Choice(Param):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def pick(self, generator):
        return pick(generator.choice(self.data), generator)


class Choices(Param):
    def __init__(self, data, k=1):
        super().__init__()
        self.data = data
        self.k = k

    def pick(self, generator):
        k = pick(self.k, generator)
        return pick(generator.choices(self.data, k=k), generator)


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
