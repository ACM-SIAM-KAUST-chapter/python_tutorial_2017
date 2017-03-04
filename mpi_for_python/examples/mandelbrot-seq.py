import numpy

x1, x2 = -2.0, 1.0
y1, y2 = -1.0, 1.0
w,  h  = 150, 100
maxit  = 127

def mandelbrot(x, y, maxit):
    c = x + y*1j
    z = 0 + 0j
    it = 0
    while abs(z) < 2 and it < maxit:
        z = z**2 + c
        it += 1
    return it

C = numpy.empty([h, w], dtype='i')

dx = (x2 - x1) / w
dy = (y2 - y1) / h
for i in range(h):
    y = y1 + i * dy
    for j in range(w):
        x = x1 + j * dx
        C[i, j] = mandelbrot(x, y, maxit)

try:
    from matplotlib import pyplot as plt
    plt.imshow(C, aspect='equal')
    plt.spectral()
    if not plt.isinteractive():
        import signal
        def action(*args): raise SystemExit
        signal.signal(signal.SIGALRM, action)
        signal.alarm(2)
    plt.show()
except:
    pass
