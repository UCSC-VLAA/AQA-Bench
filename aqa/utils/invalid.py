from json import JSONEncoder


class Invalid():
    # TODO: make `Invalid` JSON serializable
    def __init__(self, output):
        self.output = output

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"output={repr(self.output)}"
        s += ")"
        return s


class FormatInvalid(Invalid):
    def __init__(self, output):
        super().__init__(output)


class ValueInvalid(Invalid):
    def __init__(self, output):
        super().__init__(output)


class InvalidEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Invalid):
            return str(o)

        return super(InvalidEncoder, self).default(o)
