def v_to_v2(v):
    """Convert v to v^2

    """
    v0 = v * (0.7 - 0.3) / (10) + 0.3
    v0_min2 = 0.3 ** 2
    v0_max2 = 0.7 ** 2
    v2 = 10 * (v0 ** 2 - v0_min2) / (v0_max2 - v0_min2)
    return v2


def P_RPS(v, rho):
    """Defining RPS pressure as potential prediction variable

    """
    rho0 = rho * (1 - 0.1) / (10) + 0.1
    v0 = v * (0.7 - 0.3) / (10) + 0.3
    P0 = rho0 * v0 ** 2
    P0_min = 0.009
    P0_max = 0.49
    P = 10.0 * (P0 - P0_min) / (P0_max - P0_min)
    return P
