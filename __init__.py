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


    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()

    def __add__(self, other):
        if other == 0:
            return self

        assert isinstance(other, Struct)



        return Statistics(
            self.error     + other.error,
            self.size      + other.size,
            self.confusion + other.confusion)


    def __radd__(self, other):
        return self.__add__(other)



def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))
