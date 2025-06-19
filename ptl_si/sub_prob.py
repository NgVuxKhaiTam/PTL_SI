import numpy as np
from numpy.linalg import pinv


def _interval_bounds(psi, gamma):
    """Vectorized computation of interval bounds.

    Parameters
    ----------
    psi : np.ndarray
        Coefficients for the interval constraints.
    gamma : np.ndarray
        Offsets for the interval constraints.

    Returns
    -------
    list
        Lower and upper bounds for the parameter.
    """

    if psi.size == 0:
        return [-np.inf, np.inf]

    mask_zero = psi == 0
    if np.any(mask_zero & (gamma < 0)):
        return [np.inf, -np.inf]

    pos = psi > 0
    neg = psi < 0

    ru = np.min(gamma[pos] / psi[pos]) if np.any(pos) else np.inf
    lu = np.max(gamma[neg] / psi[neg]) if np.any(neg) else -np.inf

    return [lu, ru]


def compute_Zu(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    a_tilde_O = a_tilde[O]
    a_tilde_Oc = a_tilde[Oc]

    psi0 = np.array([])
    gamma0 = np.array([])
    psi1 = np.array([])
    gamma1 = np.array([])

    if len(O) > 0:
        inv = pinv(XO.T @ XO)
        XO_plus = inv @ XO.T

        # Calculate psi0
        XO_plus_b = XO_plus @ b
        psi0 = (-SO * XO_plus_b).ravel()

        # Calculate gamma0
        XO_plus_a = XO_plus @ a
        gamma0_term_inv = inv @ (a_tilde_O * SO)

        gamma0 = SO * XO_plus_a - N * lambda_0 * SO * gamma0_term_inv
        gamma0 = gamma0.ravel()

    if len(Oc) > 0:
        if len(O) == 0:
            proj = np.eye(N)
            temp2 = 0

        else:
            proj = np.eye(N) - XO @ XO_plus
            XO_O_plus = XO @ inv
            temp2 = (XOc.T @ XO_O_plus) @ (a_tilde_O * SO)
            temp2 = temp2 / a_tilde_Oc

        XOc_O_proj = XOc.T @ proj
        temp1 = (XOc_O_proj / a_tilde_Oc) / (lambda_0 * N)

        # Calculate psi1
        term_b = temp1 @ b
        psi1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        # Calculate gamma1
        term_a = temp1 @ a
        ones_vec = np.ones_like(term_a)

        gamma1 = np.concatenate([(ones_vec - temp2 - term_a).ravel(), (ones_vec + temp2 + term_a).ravel()])

    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    lu = -np.inf
    ru = np.inf

    for i in range(len(psi)):
        if psi[i] == 0:
            if gamma[i] < 0:
                return [np.inf, -np.inf]
        elif psi[i] > 0:
            val = gamma[i] / psi[i]
            if val < ru:
                ru = val
        else:
            val = gamma[i] / psi[i]
            if val > lu:
                lu = val
    return [lu, ru]


def compute_Zu_ver2(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    """Vectorized version of :func:`compute_Zu` for better performance."""
    a_tilde_O = a_tilde[O]
    a_tilde_Oc = a_tilde[Oc]

    psi0 = np.array([])
    gamma0 = np.array([])
    psi1 = np.array([])
    gamma1 = np.array([])

    if len(O) > 0:
        inv = pinv(XO.T @ XO)
        XO_plus = inv @ XO.T

        XO_plus_b = XO_plus @ b
        psi0 = (-SO * XO_plus_b).ravel()

        XO_plus_a = XO_plus @ a
        gamma0_term_inv = inv @ (a_tilde_O * SO)
        gamma0 = SO * XO_plus_a - N * lambda_0 * SO * gamma0_term_inv
        gamma0 = gamma0.ravel()

    if len(Oc) > 0:
        if len(O) == 0:
            proj = np.eye(N)
            temp2 = 0
        else:
            proj = np.eye(N) - XO @ XO_plus
            XO_O_plus = XO @ inv
            temp2 = (XOc.T @ XO_O_plus) @ (a_tilde_O * SO)
            temp2 = temp2 / a_tilde_Oc

        XOc_O_proj = XOc.T @ proj
        temp1 = (XOc_O_proj / a_tilde_Oc) / (lambda_0 * N)

        term_b = temp1 @ b
        psi1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        term_a = temp1 @ a
        ones_vec = np.ones_like(term_a)
        gamma1 = np.concatenate([(ones_vec - temp2 - term_a).ravel(),
                                 (ones_vec + temp2 + term_a).ravel()])

    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    return _interval_bounds(psi, gamma)

def compute_Zv(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    nu0 = np.array([])
    kappa0 = np.array([])
    nu1 = np.array([])
    kappa1 = np.array([])

    phi_a_iota = (phi_u @  a) + iota_u
    phi_b = phi_u @ b

    if len(L) > 0:
        inv = pinv(X0L.T @ X0L)
        X0L_plus = inv @ X0L.T

        # Calculate nu0
        X0L_plus_phi_b = X0L_plus @ phi_b
        nu0 = (- SL * X0L_plus_phi_b).ravel()

        # Calculate kappa0
        X0L_plus_a = X0L_plus @ phi_a_iota
        kappa0_term_inv = inv @ SL
        kappa0 = SL * X0L_plus_a - (nT * lambda_tilde) * SL * kappa0_term_inv
        kappa0 = kappa0.ravel()

    if len(Lc) > 0:
        if len(L) == 0:
            proj = np.eye(nT)
            temp2 = 0

        else:
            proj = np.eye(nT) - X0L@X0L_plus

            X0L_T_plus = X0L @ inv
            temp2 = (X0Lc.T @ X0L_T_plus) @ SL


        X0Lc_T_proj = X0Lc.T @ proj
        temp1 = X0Lc_T_proj / (lambda_tilde * nT)

        # Calculate nu1
        term_b = temp1 @ phi_b
        nu1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        # Calculate kappa1
        term_a = temp1 @ phi_a_iota
        ones_vec = np.ones_like(term_a)
        kappa1 = np.concatenate([(ones_vec - temp2 - term_a).ravel(), (ones_vec + temp2 + term_a).ravel()])

    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))

    lv = -np.inf
    rv = np.inf

    for i in range(len(nu)):
        if nu[i] == 0:
            if kappa[i] < 0:
                return [np.inf, -np.inf]
        elif nu[i] > 0:
            val = kappa[i] / nu[i]
            if val < rv:
                rv = val
        else:
            val = kappa[i] / nu[i]
            if val > lv:
                lv = val

    return [lv, rv]


def compute_Zv_ver2(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    """Vectorized version of :func:`compute_Zv` for better performance."""
    nu0 = np.array([])
    kappa0 = np.array([])
    nu1 = np.array([])
    kappa1 = np.array([])

    phi_a_iota = (phi_u @ a) + iota_u
    phi_b = phi_u @ b

    if len(L) > 0:
        inv = pinv(X0L.T @ X0L)
        X0L_plus = inv @ X0L.T

        X0L_plus_phi_b = X0L_plus @ phi_b
        nu0 = (-SL * X0L_plus_phi_b).ravel()

        X0L_plus_a = X0L_plus @ phi_a_iota
        kappa0_term_inv = inv @ SL
        kappa0 = SL * X0L_plus_a - (nT * lambda_tilde) * SL * kappa0_term_inv
        kappa0 = kappa0.ravel()

    if len(Lc) > 0:
        if len(L) == 0:
            proj = np.eye(nT)
            temp2 = 0
        else:
            proj = np.eye(nT) - X0L @ X0L_plus
            X0L_T_plus = X0L @ inv
            temp2 = (X0Lc.T @ X0L_T_plus) @ SL

        X0Lc_T_proj = X0Lc.T @ proj
        temp1 = X0Lc_T_proj / (lambda_tilde * nT)

        term_b = temp1 @ phi_b
        nu1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        term_a = temp1 @ phi_a_iota
        ones_vec = np.ones_like(term_a)
        kappa1 = np.concatenate([(ones_vec - temp2 - term_a).ravel(),
                                 (ones_vec + temp2 + term_a).ravel()])

    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))

    return _interval_bounds(nu, kappa)


def compute_Zt(M, SM, Mc, xi_uv, zeta_uv, a, b):
    omega0 = np.array([])
    rho0 = np.array([])
    omega1 = np.array([])
    rho1 = np.array([])

    xi_a_zeta = (xi_uv @  a) + zeta_uv
    xi_b = xi_uv @ b

    if len(M) > 0:
        Dt_xi_a_zeta = xi_a_zeta[M]
        Dt_xi_b = xi_b[M]

        # Calculate omega0, rho0
        omega0 = (-SM * Dt_xi_b).ravel()
        rho0 = (SM * Dt_xi_a_zeta).ravel()

    if len(Mc) > 0:
        Dtc_xi_a_zeta = xi_a_zeta[Mc]
        Dtc_xi_b = xi_b[Mc]

        # Calculate omega1, rho1
        omega1 = np.concatenate([Dtc_xi_b.ravel(), -Dtc_xi_b.ravel()])
        rho1 = np.concatenate([-Dtc_xi_a_zeta.ravel(), Dtc_xi_a_zeta.ravel()])

    omega = np.concatenate((omega0, omega1))
    rho = np.concatenate((rho0, rho1))

    lt = -np.inf
    rt = np.inf

    for i in range(len(omega)):
        if omega[i] == 0:
            if rho[i] < 0:
                return [np.inf, -np.inf]
        elif omega[i] > 0:
            val = rho[i] / omega[i]
            if val < rt:
                rt = val
        else:
            val = rho[i] / omega[i]
            if val > lt:
                lt = val

    return [lt, rt]


def compute_Zt_ver2(M, SM, Mc, xi_uv, zeta_uv, a, b):
    """Vectorized version of :func:`compute_Zt` for better performance."""
    omega0 = np.array([])
    rho0 = np.array([])
    omega1 = np.array([])
    rho1 = np.array([])

    xi_a_zeta = (xi_uv @ a) + zeta_uv
    xi_b = xi_uv @ b

    if len(M) > 0:
        Dt_xi_a_zeta = xi_a_zeta[M]
        Dt_xi_b = xi_b[M]

        omega0 = (-SM * Dt_xi_b).ravel()
        rho0 = (SM * Dt_xi_a_zeta).ravel()

    if len(Mc) > 0:
        Dtc_xi_a_zeta = xi_a_zeta[Mc]
        Dtc_xi_b = xi_b[Mc]

        omega1 = np.concatenate([Dtc_xi_b.ravel(), -Dtc_xi_b.ravel()])
        rho1 = np.concatenate([-Dtc_xi_a_zeta.ravel(), Dtc_xi_a_zeta.ravel()])

    omega = np.concatenate((omega0, omega1))
    rho = np.concatenate((rho0, rho1))

    return _interval_bounds(omega, rho)

def compute_Zu_otl(SO, O, XIO, Oc, XIOc, a, b, P, lambda_w, nI):
    psi0 = np.array([])
    gamma0 = np.array([])
    psi1 = np.array([])
    gamma1 = np.array([])

    if len(O) > 0:
        inv = pinv(XIO.T @ XIO)
        XIO_plus = inv @ XIO.T

        # Calculate psi0
        XIO_plus_Pb = XIO_plus @ P @ b
        psi0 = (-SO * XIO_plus_Pb).ravel()

        # Calculate gamma0
        XIO_plus_Pa = XIO_plus @ P @ a
        gamma0_term_inv = inv @ SO

        gamma0 = SO * XIO_plus_Pa - nI * lambda_w * SO * gamma0_term_inv
        gamma0 = gamma0.ravel()

    if len(Oc) > 0:
        if len(O) == 0:
            proj = np.eye(nI)
            temp2 = 0

        else:
            proj = np.eye(nI) - XIO @ XIO_plus
            XIO_T_plus = XIO @ inv
            temp2 = (XIOc.T @ XIO_T_plus) @ SO

        XIOc_T_proj = XIOc.T @ proj
        temp1 = XIOc_T_proj / (lambda_w * nI)

        # Calculate psi1
        term_Pb = temp1 @ P @ b
        psi1 = np.concatenate([term_Pb.ravel(), - term_Pb.ravel()])

        # Calculate gamma1
        term_Pa = temp1 @ P @ a
        ones_vec = np.ones_like(term_Pa)

        gamma1 = np.concatenate([(ones_vec - temp2 - term_Pa).ravel(), (ones_vec + temp2 + term_Pa).ravel()])


    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    lu = -np.inf
    ru = np.inf

    for i in range(len(psi)):
        if psi[i] == 0:
            if gamma[i] < 0:
                return [np.inf, -np.inf]
        elif psi[i] > 0:
            val = gamma[i] / psi[i]
            if val < ru:
                ru = val
        else:
            val = gamma[i] / psi[i]
            if val > lu:
                lu = val

    return [lu, ru]

def calculate_phi_iota_xi_zeta(X, SO, O, XO, X0, SL, L, X0L, p, B, Q, lambda_0, lambda_tilde, a_tilde, N, nT):
    phi_u = Q.copy()
    iota_u = np.zeros((nT, 1))
    xi_uv = np.zeros((p, N))
    zeta_uv = np.zeros((p, 1))

    if len(O) > 0:
        a_tilde_O = a_tilde[O]
        Eu = np.eye(X.shape[1])[:, O]
        inv_XOT_XO = pinv(XO.T @ XO)
        XO_plus = inv_XOT_XO @ XO.T
        X0_B_Eu = X0 @ B @ Eu
        B_Eu_inv_XOT_XO = B @ Eu @ inv_XOT_XO

        phi_u -= (1.0 / N) * (X0_B_Eu @ XO_plus)
        iota_u = lambda_0 * (X0_B_Eu @ inv_XOT_XO) @ (a_tilde_O * SO)

        xi_uv += (1.0 / N) * (B_Eu_inv_XOT_XO @ XO.T)
        zeta_uv += -lambda_0 * B_Eu_inv_XOT_XO @ (a_tilde_O * SO)

    if len(L) > 0:
        Fv = np.eye(p)[:, L]
        inv_X0LT_X0L = pinv(X0L.T @ X0L)
        X0L_plus = inv_X0LT_X0L @ X0L.T

        xi_uv += Fv @ X0L_plus @ phi_u
        zeta_uv += Fv @ inv_X0LT_X0L @ (X0L.T @ iota_u - (nT * lambda_tilde) * SL)

    return phi_u, iota_u, xi_uv, zeta_uv


def calculate_phi_iota_xi_zeta_otl(XI, SO, O, XIO, X0, SL, L, X0L, p, Q, P, lambda_w, lambda_del, nI, nT):
    phi_u = Q.copy()
    iota_u = np.zeros((nT, 1))
    xi_uv = np.zeros((p, nI + nT))
    zeta_uv = np.zeros((p, 1))

    if len(O) > 0:
        Eu = np.eye(XI.shape[1])[:, O]
        inv_XIO_T_XIO = pinv(XIO.T @ XIO)
        XIO_plus = inv_XIO_T_XIO @ XIO.T
        X0_Eu = X0 @ Eu

        phi_u -= X0_Eu @ XIO_plus @ P
        iota_u = (nI * lambda_w) * (X0_Eu @ inv_XIO_T_XIO @ SO)
        xi_uv += Eu @ XIO_plus @ P
        zeta_uv += -(nI * lambda_w) * (Eu @  inv_XIO_T_XIO @ SO)

    if len(L) > 0:
        Fv = np.eye(p)[:, L]
        inv_X0L_T_X0L = pinv(X0L.T @ X0L)
        X0L_plus = inv_X0L_T_X0L @ X0L.T

        xi_uv += Fv @ X0L_plus @ phi_u
        zeta_uv += Fv @ inv_X0L_T_X0L @ (X0L.T @ iota_u - (nT * lambda_del) * SL)

    return phi_u, iota_u, xi_uv, zeta_uv
