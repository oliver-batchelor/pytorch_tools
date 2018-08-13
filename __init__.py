from collections import Counter

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getitem__(self, index):
        return self.__dict__[index]

    def __iter__(self):
        return self.__dict__.__iter__()

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()        


    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()

    def __floordiv__(self, divisor):
        return Struct(**{k: v/divisor for (k, v) in self.items()})

    def __truediv__(self, divisor):
        return self.__floordiv__(divisor)


    def __add__(self, other):
        if other == 0:
            return self

        assert isinstance(other, Struct)

        r = {}
        for k in (self.keys() & other.keys()):
            r[k] = self[k] + other[k]

        for k in (self.keys() - other.keys()):
            r[k] = self[k]

        for k in (other.keys() - self.keys()):
            r[k] = other[k]

        return Struct(**r)


    def merge(self, other):
        """
        returns a struct which is a merge of this struct and another.
        """

        assert isinstance(other, Struct)
        d = self.__dict__.copy()
        d.update(other.__dict__)

        return Struct(**d)

    def __radd__(self, other):
        return self.__add__(other)



def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))
