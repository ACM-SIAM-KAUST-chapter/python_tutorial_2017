def integrand(x, n):
    """
    Bessel function of first kind and order n. 
    """
    return special.jv(n, x)

xl = 0  # the lower limit of x
xu = 10 # the upper limit of x

val, abserr = integrate.quad(integrand, xl, xu, args=(0,))

print(val, abserr)