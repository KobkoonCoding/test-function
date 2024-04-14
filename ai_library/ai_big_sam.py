import numpy as np

#contraction map
def F(x):
    return 0.999*x
def theta_n(x0,x1,n):
    if (x0==x1).all():
        return 0
    else:
        T0 = 1
        T1 = (1+np.sqrt(1+4*T0**2))/2
        return min( T0/T1, 1/(2**n*np.linalg.norm(x1-x0,'fro')) )

def theta_iBIGSAM(x0,x1,i,alpha,eta):
    if (x0==x1).all():
        return (i-1)/(i+alpha-1)
    else:
        return min((i-1)/(i+alpha-1), eta/ np.linalg.norm(x1-x0,'fro')  ) 

def aiBIGSAM5(prox, L1, L2, Lip, m, n, itrs_max):
    # Initialization and setup
    x1 = np.zeros((m, n))
    x0 = np.zeros((m, n))
    lam1 = 1e-5  # 10^-5
    for i in range(1, itrs_max + 1):
        cn = Lip
        alpha = 3
        alpha_n = 1 / (n + 1)
        eta = 1 / (n + 1) ** 2

        if i % 2 == 0:
            v = x1
        else:
            v = x1 + theta_iBIGSAM(x0, x1, i, alpha, eta) * (x0 - x1)

        y = prox(v,cn,lam1,L1,L2)
        z = F(v)
        x = alpha_n * z + (1 - alpha_n) * y  # x_{n+1}

        x0, x1 = x1, x  # Update values for the next iteration
    Beta = x  # Final result
    return Beta

# Proximity operator definition 2
def proxB(x, cn, lam, L1, L2):
    # Compute the gradient of f(x)
    gradf = 2*( np.dot(L1,x) - L2)
    # Compute y = x - cn * gradf
    y = x - cn * gradf
    # Compute the proximal operator value
    value = np.maximum(np.abs(y) - lam * cn, 0) * np.sign(y)

    return value  # Return the prox_cn g(I - cn * gradf)(x) value
###############################################################################
    # Proximity operator definition for finding Beta
def proxH(x, cn, lam, L1, L2):
    # Compute the gradient of f(x)
    gradf = 2*( np.dot(x,L1) - L2)
    # Compute y = x - cn * gradf
    y = x - cn * gradf
    # Compute the proximal operator value
    value2 = np.maximum(np.abs(y) - lam * cn, 0) * np.sign(y)