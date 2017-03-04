import numpy

def compute_pi(samples):
    count = 0
    for x, y in samples:
        if x**2 + y**2 <= 1:
            count += 1
    pi = 4*float(count)/len(samples)
    return pi

N = 100000
samples = numpy.random.random((N, 2))
pi = compute_pi(samples)

error = abs(pi - numpy.pi)
print("pi is approximately %.16f, error is %.16f" % (pi, error))
