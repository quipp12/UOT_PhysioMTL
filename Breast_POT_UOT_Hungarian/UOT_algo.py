import numpy as np
from numpy.linalg import norm
from copy import copy
import time


float_type = np.double

def sinkhorn_uot(C, a, b, eta=0.1, tau1=2, tau2=4, k=3 , compute_optimal=True):
    """
    Sinkhorn algorithm for entropic-regularized Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param eta:
    :param tau1:
    :param tau2:
    :param k:
    :param epsilon:
    :return:
    """






    output = {
        "u": list(),
        "v": list(),
        "f": list(),
        "g_dual": list()
    }

    # Compute optimal value and X for unregularized UOT
    # if compute_optimal:
    #     f_optimal, X_optimal = solve_f_cp(C, a, b, tau1=tau1, tau2=tau2)
    #     output["f_optimal"] = f_optimal
    #     output["X_optimal"] = X_optimal

    # Initialization
    u = np.zeros_like(a).astype(float_type)
    v = np.zeros_like(b).astype(float_type)
    # u = np.zeros_like(a)
    # v = np.zeros_like(b)

    output["u"].append(copy(u))
    output["v"].append(copy(v))

    # # Compute initial value of f
    # B = compute_B(C, u, v, eta)
    # f = compute_f_primal(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)
    # output["f"].append(f)

    for i in range(k):
        # u_old = copy(u)
        # v_old = copy(v)
        B = np.exp((u + v.T - C) / eta)
        #
        # f = compute_f(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)
        #
        # output["f"].append(f)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(a) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(b) - np.log(Bb)) * (tau2 * eta / (eta + tau2))
        #
        # g_dual = compute_g_dual(C=C, u=u, v=v, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2)

        output["u"].append(copy(u))
        output["v"].append(copy(v))


        # output["g_dual"].append(g_dual)

        # err = norm(u - u_old, ord=1) + norm(v - v_old, ord=1)

        # if err < 1e-10:
        #     break
        #
        # if np.abs(f - output["f_optimal"]) < epsilon:
        #     break

    u = output["u"][-1]

    v = output["v"][-1]

    transport_matrix = np.exp((u + v.T - C) / eta)

    return transport_matrix

