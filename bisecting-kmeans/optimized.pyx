import cython
from cython.parallel import prange, parallel
def f(x,y):
    return x*y

@cython.boundscheck(False)
cpdef double calculateDist(double[:] p1,double[:] p2):
    cdef double d = 0
    cdef int n=p1.shape[0]
    cdef int i
    with nogil,parallel(num_threads=4):
        for i in prange(n,schedule='dynamic'):
            d+=(p1[i]-p2[i])**2
    d=d**0.5
    return d



