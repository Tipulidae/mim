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


def pick(val, r):
    if isinstance(val, Param):
        return val.pick(r)
    if isinstance(val, dict):
        return {k: pick(v, r) for k, v in val.items()}
    if isinstance(val, list):
        return [pick(x, r) for x in val]
    else:
        return val
