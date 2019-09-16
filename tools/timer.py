import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start



def benchmark(f, n=100):
    try:
        with Timer() as t:
            [f() for i in range(0, n)]
    finally:
        print('Completed {:d} iterations in {:.2f} sec'.format(int(n), t.interval))

    
 