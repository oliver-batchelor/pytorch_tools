from collections import Counter
from collections.abc import Mapping
import torch
from torch import Tensor

from numbers import Number
import math

import operator
import itertools

import pprint

from functools import reduce



def to_dicts(s):
    if isinstance(s, Struct):
        return {k:to_dicts(v) for k, v in s.__dict__.items()}
    if isinstance(s, dict):
        return {k:to_dicts(v) for k, v in s.items()}
    if isinstance(s, list):
        return [to_dicts(v) for v in s]        
    if isinstance(s, tuple):
        return tuple(to_dicts(v) for v in s)
    if isinstance(s, Tensor):
        return s.tolist()
    else:
        return s

def to_structs(d):
    if isinstance(d, dict):
        return Struct( {k : to_structs(v) for k, v in d.items()} )
    if isinstance(d, list):
        return [to_structs(v) for v in d]
    if isinstance(d, tuple):
        return tuple(to_structs(v) for v in d)
    else:
        return d

    


class Struct(Mapping):
    def __init__(self, entries):
        assert type(entries) == dict
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

    def __eq__(self, other):
        if isinstance(other, Struct):
            return self.__dict__ == other.__dict__
        else:
            return False

    def _to_dicts(self):
        return to_dicts(self)

    def _subset(self, *keys):
        d = {k:self[k] for k in keys}
        return self.__class__(d)

    def _filter_none(self):
        return self.__class__({k: v for k, v in self.items() if v is not None})


    def _map(self, f, *args, **kwargs):
        return self.__class__({k: f(v, *args, **kwargs) for k, v in self.items()})

    def _mapWithKey(self, f):
        m = {k: f(k, v) for k, v in self.__dict__.items()}
        return self.__class__(m)

    def __repr__(self):
        commaSep = ", ".join(["{}={}".format(str(k), repr(v)) for k, v in self.items()])
        return "{" + commaSep + "}"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.__dict__.__len__()

    def __floordiv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.floordiv, other)
        else:
            return self._zipWith(operator.floordiv, other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.truediv, other)
        else:
            return self._zipWith(operator.truediv, other)

    def __add__(self, other):
        if isinstance(other, Number):
            return self._map(operator.add, other)
        else:
            return self._zipWith(operator.add, other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._map(operator.mul, other)
        else:
            return self._zipWith(operator.mul, other)


    def _zipWith(self, f, other):
        assert isinstance(other, Struct)
        assert self.keys() == other.keys()

        r = {k:f(self[k], other[k]) for k in self.keys()}
        return self.__class__(r)


    def _merge(self, other):
        """
        returns a struct which is a merge of this struct and another.
        """

        assert isinstance(other, Struct)
        d = self.__dict__.copy()
        d.update(other.__dict__)

        return self.__class__(d)

    def _extend(self, **values):
        d = self.__dict__.copy()
        d.update(values)

        return self.__class__(d)


    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)





class ZipList():
    def __init__(self, elems):
        self.elems = list(elems)


    def __getitem__(self, index):
        return self.elems[index]

    def __iter__(self):
        return self.elems.__iter__()

    def __eq__(self, other):
        if isinstance(other, ZipList):
            return self.elems == other.elems
        else:
            return False

    def __repr__(self):
        return self.elems.__repr__()

    def __str__(self):
        return self.elems.__str__()

    def __len__(self):
        return self.elems.__len__()

    def __floordiv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.floordiv, other)
        else:
            return self._zipWith(operator.floordiv, other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._map(operator.truediv, other)
        else:
            return self._zipWith(operator.truediv, other)


    def _map(self, f, *args, **kwargs):
        return ZipList([f(v, *args, **kwargs) for v in self.elems])

    def _zipWith(self, f, other):

        assert isinstance(other, ZipList)
        assert len(self) == len(other)

        r = [f(x, y) for x, y in zip(self.elems, other.elems)]
        return ZipList(r)


    def __add__(self, other):
        if isinstance(other, Number):
            return self._map(operator.add, other)
        else:
            return self._zipWith(operator.add, other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self._map(operator.mul, other)
        else:
            return self._zipWith(operator.mul, other)        


    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def append(x):
        self.elems.append(x)
        return self



class Table(Struct):
    def __init__(self, d):

        assert len(d) > 0, "empty Table"
        t = next(iter(d.values()))
      
        for (k, v) in d.items():
            assert type(v) == torch.Tensor, "expected tensor, got " + type(t).__name__
            assert v.size(0) == t.size(0), "mismatched column sizes: " + str(show_shapes(d))

        super(Table, self).__init__(d)


    def __getitem__(self, index):
        return self.__dict__[index]


    def _index_select(self, index):
        if type(index) is torch.Tensor:
            assert index.dtype == torch.int64 
            assert index.dim() == 1
            
            return self._map(lambda t: t[index])

        elif type(index) is int:
            return Struct({k: v[index] for k, v in self.items()})
        assert False, "Table.index_select: unsupported index type" + type(index).__name__

        
    def _narrow(self, start, n):
        return self._map(lambda t: t.narrow(0, start, n))

    def _take(self, n):
        return self._narrow(0, min(self._size, n))

    def _drop(self, n):
        n = min(self._size, n)
        return self._narrow(n, self._size - n)


    def _sequence(self):
        return (self._index_select(i) for i in range(0, self._size))

    def _sort_on(self, key, descending=False):
        assert key in self
        assert self[key].dim() == 1

        values, inds = self[key].sort(descending = descending)
        return Table({k: values if k == key else v[inds] for k, v in self.items()})

    @property
    def _head(self):
        return next(iter(self.__dict__.values()))

    @property
    def _size(self):
        return self._head.size(0)

    @property
    def _device(self):
        return self._head.device

    def _to(self, device):
        return self._map(lambda t: t.to(device))

    def _cpu(self):
        return self._map(lambda t: t.cpu())



def struct(**d):
    return Struct(d)

def table(**d):
    return Table(d)



class Histogram:
    def __init__(self, values = torch.FloatTensor(0), range = (0, 1), num_bins = 10, trim = True):
        assert len(range) == 2

        self.range = range
        lower, upper = self.range

        bin_indexes = (values - lower) * num_bins / (upper - lower) 
        bin_indexes = bin_indexes.long()

        if trim:
            valid = (bin_indexes >= 0) & (bin_indexes < num_bins)

            values = values[valid]
            bin_indexes = bin_indexes[valid]
        
        bin_indexes.clamp_(0, num_bins - 1)

        self.sum         = values.sum().item()
        self.sum_squares = values.norm(2).item()
        self.counts = torch.bincount(bin_indexes, minlength = num_bins)

    def __repr__(self):
        return self.counts.tolist().__repr__()

    def bins(self):
        lower, upper = self.range
        d = (upper - lower) / self.counts.size(0)

        return torch.FloatTensor([lower + i * d for i in range(0, self.counts.size(0) + 1)])

    def to_struct(self):
        return struct(sum=self.sum, sum_squares=self.sum_squares, counts=self.counts)

    def __add__(self, other):
        assert isinstance(other, Histogram)
        assert other.counts.size(0) == self.counts.size(0), "mismatched histogram sizes"
        
        total = Histogram(range = self.range, num_bins = self.counts.size(0))
        total.sum = self.sum + other.sum
        total.sum_squares = self.sum_squares + other.sum_squares
        total.counts = self.counts + other.counts

        return total


    def __truediv__(self, x):
        assert isinstance(x, Number)

        total = Histogram(range = self.range, num_bins =  self.counts.size(0))
        total.sum = self.sum / x
        total.sum_squares = self.sum_squares / x
        total.counts = self.counts / x

        return total

    @property
    def std(self):

        n = self.counts.sum().item()
        if n > 1:
            sum_squares = self.sum_squares - (self.sum * self.sum / n)
            var = max(0, sum_squares / (n - 1))

            return math.sqrt(var)
        else:
            return 0

    @property
    def mean(self):
        n = self.counts.sum().item()
        if n > 0:
            return self.sum / self.counts.sum().item()
        else:
            return 0


def show_shapes_info(x):

    if type(x) == torch.Tensor:
        return tuple([*x.size(), x.dtype, x.device])
    elif type(x) == list:
        return list(map(show_shapes_info, x))
    elif type(x) == tuple:
        return tuple(map(show_shapes_info, x))
    elif isinstance(x, Mapping):
        return {k : show_shapes_info(v) for k, v in x.items()}
    else:
        return str(x)

def show_shapes(x):

    if type(x) == torch.Tensor:
        return tuple([*x.size()])
    elif type(x) == list:
        return list(map(show_shapes, x))
    elif type(x) == tuple:
        return tuple(map(show_shapes, x))
    elif isinstance(x, Mapping):
        return {k : show_shapes(v) for k, v in x.items()}
    else:
        return str(x)


def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))


def replace(d, key, value):
    return {**d, key:value}

def over_struct(key, f):
    def modify(d):
        value = f(d[key])
        return Struct(replace(d, key, value))
    return modify

def over(key, f):
    def modify(d):
        value = f(d[key])
        return replace(d, key, value)
    return modify

def transpose_partial(dicts):
    accum = {}
    for d in dicts:
        for k, v in d.items():
            if k in accum:
                accum[k].append(v)
            else:
                accum[k] = [v]
    return accum

def transpose_partial_structs(structs):
    return Struct(transpose_partial(d.__dict__ for d in structs))


def transpose_structs(structs):
    elem = structs[0]
    d =  {key: [d[key] for d in structs] for key in elem}
    return Struct(d) 



def transpose_lists(lists):
    return list(zip(*lists))

def cat_tables(tables):
    t = transpose_structs(tables)
    return Table(dict(t._map(torch.cat))) 

def drop_while(f, xs):
    while(len(xs) > 0 and f(xs[0])):
        _, *xs = xs    

    return xs


def filter_none(xs):
    return [x for x in xs if x is not None]

def filter_map(f, xs):
    return filter_none(map(f, xs))

def pluck(k, xs, default=None):
    return [d.get(k, default) for d in xs]

def pluck_struct(k, xs, default=None):
    return xs._map(lambda x: x.get(k, default))


def const(x):
    def f(*y):
        return x
    return f

def concat_lists(xs):
    return list(itertools.chain.from_iterable(xs))


def map_dict(f, d):
    return {k :  f(v) for k, v in d.items()}

def pprint_struct(s, indent=2, width=160):
    pp = pprint.PrettyPrinter(indent=indent, width=width)
    pp.pprint(s._to_dicts())


def sum_list(xs):
    assert len(xs) > 0
    return reduce(operator.add, xs)


def append_dict(d, k, v):
    xs = d.get(k) or []
    xs.append(v)

    d[k] = xs
    return d


def transpose_dicts(d):
    r = {}
    for k, v in d.items():
        for j, u in v.items():
            inner = r.get(j) or {}
            inner[k] = u
            r[j] = inner
    return r    

def add_dict(d, k):
    d[k] = d[k] + 1 if k in d else 1
    return d


def count_dict(xs):
    counts = {}
    for k in xs:
        add_dict(counts, k)

    return counts

def sum_dicts(ds):
    r = {}

    for d in ds:
        for k, v in d.items():
            r[k] = r.get(k, 0) + v

    return r


def partition_by(xs, f):
    partitions = {}

    for x in xs:
        k, v = f(x)
        append_dict(partitions, k, v)    

    return partitions