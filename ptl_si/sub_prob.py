import numpy as np
from numpy.linalg import pinv


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














import numpy as np
from numpy.linalg import pinv

def compute_bounds_2(psi, gamma):
    """Optimized bound computation using vectorized operations"""
    # Handle zero psi values
    zero_mask = (psi == 0)
    if np.any(zero_mask & (gamma < 0)):
        return [np.inf, -np.inf]
    
    # Remove zero psi values for bound computation
    non_zero_mask = ~zero_mask
    if not np.any(non_zero_mask):
        return [-np.inf, np.inf]
    
    psi_nz = psi[non_zero_mask]
    gamma_nz = gamma[non_zero_mask]
    
    # Vectorized bound computation
    ratios = gamma_nz / psi_nz
    pos_mask = psi_nz > 0
    neg_mask = psi_nz < 0
    
    ru = np.min(ratios[pos_mask]) if np.any(pos_mask) else np.inf
    lu = np.max(ratios[neg_mask]) if np.any(neg_mask) else -np.inf
    
    return [lu, ru]

def compute_Zu_2(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    """Optimized version of compute_Zu"""
    # Pre-allocate arrays with known maximum size
    max_size_0 = len(O) if len(O) > 0 else 0
    max_size_1 = 2 * len(Oc) if len(Oc) > 0 else 0
    
    psi = np.empty(max_size_0 + max_size_1)
    gamma = np.empty(max_size_0 + max_size_1)
    idx = 0
    
    # Cache frequently used values
    a_tilde_O = a_tilde[O] if len(O) > 0 else None
    a_tilde_Oc = a_tilde[Oc] if len(Oc) > 0 else None
    
    if len(O) > 0:
        inv = pinv(XO.T @ XO)
        XO_plus = inv @ XO.T
        
        # Vectorized calculations
        psi[idx:idx+len(O)] = (-SO * (XO_plus @ b)).ravel()
        
        gamma_temp = SO * (XO_plus @ a) - N * lambda_0 * SO * (inv @ (a_tilde_O * SO))
        gamma[idx:idx+len(O)] = gamma_temp.ravel()
        idx += len(O)
    
    if len(Oc) > 0:
        if len(O) == 0:
            proj = np.eye(N)
            temp2 = np.zeros(len(Oc))
        else:
            proj = np.eye(N) - XO @ XO_plus
            XO_O_plus = XO @ inv
            temp2 = ((XOc.T @ XO_O_plus) @ (a_tilde_O * SO)) / a_tilde_Oc
        
        # Vectorized operations
        XOc_O_proj = XOc.T @ proj
        temp1 = XOc_O_proj / (a_tilde_Oc * lambda_0 * N)
        
        term_b = temp1 @ b
        psi[idx:idx+len(Oc)] = term_b
        psi[idx+len(Oc):idx+2*len(Oc)] = -term_b
        
        term_a = temp1 @ a
        ones_vec = np.ones_like(term_a)
        
        gamma[idx:idx+len(Oc)] = ones_vec - temp2 - term_a
        gamma[idx+len(Oc):idx+2*len(Oc)] = ones_vec + temp2 + term_a
        idx += 2 * len(Oc)
    
    # Use optimized bound computation
    return compute_bounds_2(psi[:idx], gamma[:idx])

def compute_Zv_2(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    """Optimized version of compute_Zv"""
    max_size_0 = len(L) if len(L) > 0 else 0
    max_size_1 = 2 * len(Lc) if len(Lc) > 0 else 0
    
    nu = np.empty(max_size_0 + max_size_1)
    kappa = np.empty(max_size_0 + max_size_1)
    idx = 0
    
    # Pre-compute common terms
    phi_a_iota = (phi_u @ a) + iota_u
    phi_b = phi_u @ b
    
    if len(L) > 0:
        inv = pinv(X0L.T @ X0L)
        X0L_plus = inv @ X0L.T
        
        nu[idx:idx+len(L)] = (-SL * (X0L_plus @ phi_b)).ravel()
        
        kappa_temp = SL * (X0L_plus @ phi_a_iota) - (nT * lambda_tilde) * SL * (inv @ SL)
        kappa[idx:idx+len(L)] = kappa_temp.ravel()
        idx += len(L)
    
    if len(Lc) > 0:
        if len(L) == 0:
            proj = np.eye(nT)
            temp2 = np.zeros(len(Lc))
        else:
            proj = np.eye(nT) - X0L @ X0L_plus
            X0L_T_plus = X0L @ inv
            temp2 = (X0Lc.T @ X0L_T_plus) @ SL
        
        X0Lc_T_proj = X0Lc.T @ proj
        temp1 = X0Lc_T_proj / (lambda_tilde * nT)
        
        term_b = temp1 @ phi_b
        nu[idx:idx+len(Lc)] = term_b
        nu[idx+len(Lc):idx+2*len(Lc)] = -term_b
        
        term_a = temp1 @ phi_a_iota
        ones_vec = np.ones_like(term_a)
        
        kappa[idx:idx+len(Lc)] = ones_vec - temp2 - term_a
        kappa[idx+len(Lc):idx+2*len(Lc)] = ones_vec + temp2 + term_a
        idx += 2 * len(Lc)
    
    return compute_bounds_2(nu[:idx], kappa[:idx])

def compute_Zt_2(M, SM, Mc, xi_uv, zeta_uv, a, b):
    """Optimized version of compute_Zt"""
    max_size_0 = len(M) if len(M) > 0 else 0
    max_size_1 = 2 * len(Mc) if len(Mc) > 0 else 0
    
    omega = np.empty(max_size_0 + max_size_1)
    rho = np.empty(max_size_0 + max_size_1)
    idx = 0
    
    # Pre-compute common terms
    xi_a_zeta = (xi_uv @ a) + zeta_uv
    xi_b = xi_uv @ b
    
    if len(M) > 0:
        omega[idx:idx+len(M)] = (-SM * xi_b[M]).ravel()
        rho[idx:idx+len(M)] = (SM * xi_a_zeta[M]).ravel()
        idx += len(M)
    
    if len(Mc) > 0:
        Dtc_xi_b = xi_b[Mc]
        Dtc_xi_a_zeta = xi_a_zeta[Mc]
        
        omega[idx:idx+len(Mc)] = Dtc_xi_b
        omega[idx+len(Mc):idx+2*len(Mc)] = -Dtc_xi_b
        
        rho[idx:idx+len(Mc)] = -Dtc_xi_a_zeta
        rho[idx+len(Mc):idx+2*len(Mc)] = Dtc_xi_a_zeta
        idx += 2 * len(Mc)
    
    return compute_bounds_2(omega[:idx], rho[:idx])




import numpy as np
from numpy.linalg import pinv
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    cp_pinv = cp.linalg.pinv  # Dùng pinv của cupy
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    cp_pinv = pinv
    print("CuPy not available, using CPU only")


def compute_bounds(psi, gamma):
    """Tính toán bound cho bài toán tối ưu (NumPy hoặc CuPy)"""
    # Đảm bảo là vector 1 chiều
    psi = psi.ravel()
    gamma = gamma.ravel()
    lu = -np.inf
    ru = np.inf
    for i in range(len(psi)):
        if psi[i] == 0:
            if gamma[i] < 0:
                return np.inf, -np.inf
        elif psi[i] > 0:
            val = gamma[i] / psi[i]
            if val < ru:
                ru = val
        else:
            val = gamma[i] / psi[i]
            if val > lu:
                lu = val
    return lu, ru

class OptimizedCompute:
    def __init__(self, use_gpu=None):
        self.use_gpu = GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)
        self.xp = cp if self.use_gpu else np
        self.pinv_func = cp_pinv if self.use_gpu else pinv
        print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")

    def _get_array(self, arr):
        if self.use_gpu and isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        elif not self.use_gpu and hasattr(arr, 'get'):
            return arr.get()
        return arr

    def _to_cpu(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()
        return arr

    def compute_bounds_3(self, psi, gamma):
        psi = self._get_array(psi).ravel()
        gamma = self._get_array(gamma).ravel()
        lu, ru = compute_bounds(psi, gamma)
        return [lu, ru]

    def compute_Zu_3(self, SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
        xp = self.xp
        SO = self._get_array(SO).ravel()
        a = self._get_array(a).ravel()
        b = self._get_array(b).ravel()
        a_tilde = self._get_array(a_tilde).ravel()
        O = np.asarray(O)
        Oc = np.asarray(Oc)
        XO = self._get_array(XO) if O.size > 0 else None
        XOc = self._get_array(XOc) if Oc.size > 0 else None

        total_size = len(O) + 2 * len(Oc)
        if total_size == 0:
            return [-xp.inf, xp.inf]

        psi = xp.empty(total_size)
        gamma = xp.empty(total_size)
        idx = 0

        if O.size > 0:
            inv = self.pinv_func(XO.T @ XO)
            XO_plus = inv @ XO.T
            a_tilde_O = a_tilde[O]
            SO_O = SO
            XO_plus_b = (XO_plus @ b).ravel()
            XO_plus_a = (XO_plus @ a).ravel()
            psi_o = (-SO_O * XO_plus_b)
            psi[idx:idx+len(O)] = psi_o

            gamma_term = inv @ (a_tilde_O * SO_O)
            gamma_o = (SO_O * XO_plus_a - N * lambda_0 * SO_O * gamma_term.ravel())
            gamma[idx:idx+len(O)] = gamma_o
            idx += len(O)

        if Oc.size > 0:
            a_tilde_Oc = a_tilde[Oc]
            if O.size == 0:
                proj = xp.eye(N)
                temp2 = xp.zeros(len(Oc))
            else:
                proj = xp.eye(N) - XO @ XO_plus
                XO_O_plus = XO @ inv
                temp2 = ((XOc.T @ XO_O_plus) @ (a_tilde[O] * SO)).ravel() / a_tilde_Oc

            XOc_O_proj = XOc.T @ proj
            temp1 = XOc_O_proj / (a_tilde_Oc * lambda_0 * N)[:, None]

            term_b = (temp1 @ b).ravel()
            term_a = (temp1 @ a).ravel()
            ones_vec = xp.ones_like(term_a)

            psi[idx:idx+len(Oc)] = term_b
            psi[idx+len(Oc):idx+2*len(Oc)] = -term_b

            gamma[idx:idx+len(Oc)] = (ones_vec - temp2 - term_a)
            gamma[idx+len(Oc):idx+2*len(Oc)] = (ones_vec + temp2 + term_a)

        return self.compute_bounds_3(psi, gamma)

    def compute_Zv_3(self, SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
        xp = self.xp
        SL = self._get_array(SL).ravel()
        a = self._get_array(a).ravel()
        b = self._get_array(b).ravel()
        phi_u = self._get_array(phi_u)
        iota_u = self._get_array(iota_u).ravel()
        L = np.asarray(L)
        Lc = np.asarray(Lc)
        X0L = self._get_array(X0L) if L.size > 0 else None
        X0Lc = self._get_array(X0Lc) if Lc.size > 0 else None

        total_size = len(L) + 2 * len(Lc)
        if total_size == 0:
            return [-xp.inf, xp.inf]

        nu = xp.empty(total_size)
        kappa = xp.empty(total_size)
        idx = 0

        phi_a_iota = (phi_u @ a) + iota_u
        phi_b = phi_u @ b

        if L.size > 0:
            inv = self.pinv_func(X0L.T @ X0L)
            X0L_plus = inv @ X0L.T
            nu_l = (-SL * (X0L_plus @ phi_b).ravel())
            kappa_l = (SL * (X0L_plus @ phi_a_iota).ravel() - nT * lambda_tilde * SL * (inv @ SL).ravel())
            nu[idx:idx+len(L)] = nu_l
            kappa[idx:idx+len(L)] = kappa_l
            idx += len(L)

        if Lc.size > 0:
            if L.size == 0:
                proj = xp.eye(nT)
                temp2 = xp.zeros(len(Lc))
            else:
                proj = xp.eye(nT) - X0L @ X0L_plus
                X0L_T_plus = X0L @ inv
                temp2 = ((X0Lc.T @ X0L_T_plus) @ SL).ravel()
            X0Lc_T_proj = X0Lc.T @ proj
            temp1 = X0Lc_T_proj / (lambda_tilde * nT)

            term_b = (temp1 @ phi_b).ravel()
            term_a = (temp1 @ phi_a_iota).ravel()
            ones_vec = xp.ones_like(term_a)

            nu[idx:idx+len(Lc)] = term_b
            nu[idx+len(Lc):idx+2*len(Lc)] = -term_b

            kappa[idx:idx+len(Lc)] = (ones_vec - temp2 - term_a)
            kappa[idx+len(Lc):idx+2*len(Lc)] = (ones_vec + temp2 + term_a)

        return self.compute_bounds_3(nu, kappa)

    def compute_Zt_3(self, M, SM, Mc, xi_uv, zeta_uv, a, b):
        xp = self.xp
        SM = self._get_array(SM).ravel()
        a = self._get_array(a).ravel()
        b = self._get_array(b).ravel()
        xi_uv = self._get_array(xi_uv)
        zeta_uv = self._get_array(zeta_uv).ravel()
        M = np.asarray(M)
        Mc = np.asarray(Mc)

        total_size = len(M) + 2 * len(Mc)
        if total_size == 0:
            return [-xp.inf, xp.inf]

        omega = xp.empty(total_size)
        rho = xp.empty(total_size)
        idx = 0

        xi_a_zeta = (xi_uv @ a) + zeta_uv
        xi_b = xi_uv @ b

        if M.size > 0:
            omega_m = (-SM * xi_b[M].ravel())
            rho_m = (SM * xi_a_zeta[M].ravel())
            omega[idx:idx+len(M)] = omega_m
            rho[idx:idx+len(M)] = rho_m
            idx += len(M)

        if Mc.size > 0:
            Dtc_xi_b = xi_b[Mc].ravel()
            Dtc_xi_a_zeta = xi_a_zeta[Mc].ravel()

            omega[idx:idx+len(Mc)] = Dtc_xi_b
            omega[idx+len(Mc):idx+2*len(Mc)] = -Dtc_xi_b

            rho[idx:idx+len(Mc)] = -Dtc_xi_a_zeta
            rho[idx+len(Mc):idx+2*len(Mc)] = Dtc_xi_a_zeta

        return self.compute_bounds_3(omega, rho)


# Tạo instance toàn cục
_optimizer = OptimizedCompute()

def compute_Zu_3(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    return _optimizer.compute_Zu_3(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N)

def compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    return _optimizer.compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT)

def compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a, b):
    return _optimizer.compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a, b)

