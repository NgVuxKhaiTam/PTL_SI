import utils
import transfer_learning_hdr
import sub_prob
import random
import numpy as np

# def divide_and_conquer_TF_recursive(X, X0, a, b, Mobs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max):
#     """
#     Phiên bản CHIA ĐỂ TRỊ cải tiến, sử dụng epsilon để đảm bảo tính đúng đắn và hiệu quả.
#     """
#     # Tính toán trước các giá trị không đổi
#     a_tilde = np.concatenate([ak_weights[k] * np.ones(p) for k in range(K)] + [np.ones(p)]).reshape(-1, 1)
    
#     intervals = []
    
#     # Định nghĩa epsilon để dễ dàng thay đổi nếu cần
#     EPSILON = 1e-6 # Sử dụng một giá trị nhỏ hơn để tăng độ chính xác

#     def _recursive_search(current_z_min, current_z_max):
#         """Hàm đệ quy thực hiện tìm kiếm."""
        
#         # Điều kiện dừng: khoảng tìm kiếm không hợp lệ hoặc quá nhỏ
#         if current_z_min > current_z_max:
#             return

#         # 1. Chọn điểm thử
#         z_test = (current_z_min + current_z_max) / 2.0

#         # 2. Thực hiện các phép tính cốt lõi để tìm khoảng ổn định [l, r]
#         Yz = a + b * z_test
#         Yz = Yz.ravel()
#         Y0z = Q @ Yz
#         tz, wz, dz, bz = transfer_learning_hdr.TransFusion(X, Yz, X0, Y0z, B, N, p, K, lambda_0, lambda_tilde, ak_weights)
#         thetaO, SO, O, XO, Oc, XOc = utils.construct_thetaO_SO_O_XO_Oc_XOc(tz, X)
#         deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(dz, X0)
#         betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)
#         phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(X, SO, O, XO, X0, SL, L, X0L, p, B, Q, lambda_0, lambda_tilde, a_tilde, N, nT)
        
#         lu, ru = sub_prob.compute_Zu_3(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N)
#         lv, rv = sub_prob.compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT)
#         lt, rt = sub_prob.compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a, b)
        
#         r = min(ru, rv, rt)
#         l = max(lu, lv, lt)

#         # Kiểm tra lỗi: khoảng không hợp lệ
#         if r < l:
#             # Nếu khoảng không hợp lệ, ta không thể tin vào l và r.
#             # Một cách xử lý là chia đôi khoảng hiện tại và tìm kiếm tiếp.
#             # Điều này giúp thoát khỏi các vùng gây lỗi số.
#             mid = (current_z_min + current_z_max) / 2.0
#             _recursive_search(current_z_min, mid - EPSILON)
#             _recursive_search(mid + EPSILON, current_z_max)
#             return

#         # 3. Lưu kết quả nếu thỏa mãn điều kiện
#         if M == Mobs:
#             intervals.append((l, r))

#         # 4. Gọi đệ quy cho các khoảng con, đã loại bỏ khoảng [l, r]
        
#         # --- ĐÂY LÀ SỰ THAY ĐỔI QUAN TRỌNG ---
#         # Gọi đệ quy cho khoảng bên trái, đảm bảo không chạm vào l
#         _recursive_search(current_z_min, l - EPSILON)
        
#         # Gọi đệ quy cho khoảng bên phải, đảm bảo không chạm vào r
#         _recursive_search(r + EPSILON, current_z_max)
#         # ------------------------------------

#     # Bắt đầu đệ quy
#     _recursive_search(z_min, z_max)
    
#     # Sắp xếp và hợp nhất các khoảng kết quả
#     if not intervals:
#         return []
    
#     intervals.sort()
    
#     merged_intervals = [intervals[0]]
#     for current_l, current_r in intervals[1:]:
#         last_l, last_r = merged_intervals[-1]
#         # Hợp nhất nếu có sự giao nhau hoặc tiếp giáp (cho phép sai số nhỏ)
#         if current_l <= last_r + EPSILON:
#             merged_intervals[-1] = (last_l, max(last_r, current_r))
#         else:
#             merged_intervals.append((current_l, current_r))

#     return merged_intervals

# def PTL_SI_TF_recursive(X0, Y0, XS_list, YS_list, lambda_0, lambda_tilde, ak_weights, SigmaS_list, Sigma0, z_min=-20, z_max=20):
#     K = len(YS_list)
#     nS = YS_list[0].shape[0]
#     nT = Y0.shape[0]
#     N = nS * K + nT
#     p = X0.shape[1]

#     X = utils.construct_X(XS_list, X0, p, K)
#     Y = np.concatenate(YS_list + [Y0])
#     B = utils.construct_B(K, p, nS, nT)
#     Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)

#     theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights)
#     M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]

#     if len(M_obs) == 0:
#         return None
    
#     X0M = X0[:, M_obs]
#     Q = utils.construct_Q(nT, N)

#     p_sel_list = []
    
#     for j in M_obs:
#         etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
#         a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
#         intervals = divide_and_conquer_TF_recursive(X, X0, a, b, M_obs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max)
#         pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

#         p_sel_list.append((j, pj_sel))
    
#     return p_sel_list



import numpy as np
import warnings

# Thử import CuPy để tăng tốc GPU
try:
    import cupy as cp
    from cupy.linalg import pinv as cp_pinv
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    cp = None
    cp_pinv = np.linalg.pinv
    GPU_AVAILABLE = False
    warnings.warn("CuPy không khả dụng, chuyển sang NumPy (CPU only)")

import transfer_learning_hdr
import utils
import sub_prob


def _to_cpu(arr):
    """
    Chuyển dữ liệu từ GPU (Cupy) về CPU (NumPy) nếu cần.
    """
    if GPU_AVAILABLE and cp is not None and hasattr(arr, 'get'):
        return arr.get()
    return arr


def _to_gpu(arr):
    """
    Chuyển dữ liệu về GPU (Cupy) nếu GPU_AVAILABLE.
    """
    if GPU_AVAILABLE and cp is not None:
        return cp.asarray(_to_cpu(arr))
    return _to_cpu(arr)


def compute_Zu_3(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    """GPU optimized version of compute_Zu"""
    xp = cp if GPU_AVAILABLE and cp is not None else np
    pinv = cp_pinv if GPU_AVAILABLE and cp is not None else np.linalg.pinv
    
    # Convert inputs to appropriate arrays
    SO = _to_gpu(SO) if GPU_AVAILABLE else np.array(SO)
    a = _to_gpu(a) if GPU_AVAILABLE else np.array(a)
    b = _to_gpu(b) if GPU_AVAILABLE else np.array(b)
    a_tilde = _to_gpu(a_tilde) if GPU_AVAILABLE else np.array(a_tilde)
    
    a_tilde_O = a_tilde[O] if len(O) > 0 else xp.array([])
    a_tilde_Oc = a_tilde[Oc] if len(Oc) > 0 else xp.array([])

    psi0 = xp.array([])
    gamma0 = xp.array([])
    psi1 = xp.array([])
    gamma1 = xp.array([])

    if len(O) > 0:
        XO_gpu = _to_gpu(XO) if GPU_AVAILABLE else XO
        inv = pinv(XO_gpu.T @ XO_gpu)
        XO_plus = inv @ XO_gpu.T

        # Calculate psi0
        XO_plus_b = XO_plus @ b
        psi0 = (-SO * XO_plus_b).ravel()

        # Calculate gamma0
        XO_plus_a = XO_plus @ a
        gamma0_term_inv = inv @ (a_tilde_O * SO)

        gamma0 = SO * XO_plus_a - N * lambda_0 * SO * gamma0_term_inv
        gamma0 = gamma0.ravel()

    if len(Oc) > 0:
        XOc_gpu = _to_gpu(XOc) if GPU_AVAILABLE else XOc
        
        if len(O) == 0:
            proj = xp.eye(N)
            temp2 = 0
        else:
            XO_gpu = _to_gpu(XO) if GPU_AVAILABLE else XO
            inv = pinv(XO_gpu.T @ XO_gpu)
            XO_plus = inv @ XO_gpu.T
            proj = xp.eye(N) - XO_gpu @ XO_plus
            XO_O_plus = XO_gpu @ inv
            temp2 = (XOc_gpu.T @ XO_O_plus) @ (a_tilde_O * SO)
            temp2 = temp2 / a_tilde_Oc

        XOc_O_proj = XOc_gpu.T @ proj
        temp1 = (XOc_O_proj / a_tilde_Oc) / (lambda_0 * N)

        # Calculate psi1
        term_b = temp1 @ b
        psi1 = xp.concatenate([term_b.ravel(), -term_b.ravel()])

        # Calculate gamma1
        term_a = temp1 @ a
        ones_vec = xp.ones_like(term_a)

        gamma1 = xp.concatenate([(ones_vec - temp2 - term_a).ravel(), 
                                (ones_vec + temp2 + term_a).ravel()])

    psi = xp.concatenate((psi0, psi1))
    gamma = xp.concatenate((gamma0, gamma1))

    # Convert back to CPU for bound computation
    psi_cpu = _to_cpu(psi)
    gamma_cpu = _to_cpu(gamma)

    lu = -np.inf
    ru = np.inf

    for i in range(len(psi_cpu)):
        if psi_cpu[i] == 0:
            if gamma_cpu[i] < 0:
                return [np.inf, -np.inf]
        elif psi_cpu[i] > 0:
            val = gamma_cpu[i] / psi_cpu[i]
            if val < ru:
                ru = val
        else:
            val = gamma_cpu[i] / psi_cpu[i]
            if val > lu:
                lu = val
    return [lu, ru]


def compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    """GPU optimized version of compute_Zv"""
    xp = cp if GPU_AVAILABLE and cp is not None else np
    pinv = cp_pinv if GPU_AVAILABLE and cp is not None else np.linalg.pinv
    
    # Convert inputs to appropriate arrays
    SL = _to_gpu(SL) if GPU_AVAILABLE else np.array(SL)
    a = _to_gpu(a) if GPU_AVAILABLE else np.array(a)
    b = _to_gpu(b) if GPU_AVAILABLE else np.array(b)
    phi_u = _to_gpu(phi_u) if GPU_AVAILABLE else np.array(phi_u)
    iota_u = _to_gpu(iota_u) if GPU_AVAILABLE else np.array(iota_u)
    
    nu0 = xp.array([])
    kappa0 = xp.array([])
    nu1 = xp.array([])
    kappa1 = xp.array([])

    phi_a_iota = (phi_u @ a) + iota_u
    phi_b = phi_u @ b

    if len(L) > 0:
        X0L_gpu = _to_gpu(X0L) if GPU_AVAILABLE else X0L
        inv = pinv(X0L_gpu.T @ X0L_gpu)
        X0L_plus = inv @ X0L_gpu.T

        # Calculate nu0
        X0L_plus_phi_b = X0L_plus @ phi_b
        nu0 = (-SL * X0L_plus_phi_b).ravel()

        # Calculate kappa0
        X0L_plus_a = X0L_plus @ phi_a_iota
        kappa0_term_inv = inv @ SL
        kappa0 = SL * X0L_plus_a - (nT * lambda_tilde) * SL * kappa0_term_inv
        kappa0 = kappa0.ravel()

    if len(Lc) > 0:
        X0Lc_gpu = _to_gpu(X0Lc) if GPU_AVAILABLE else X0Lc
        
        if len(L) == 0:
            proj = xp.eye(nT)
            temp2 = 0
        else:
            X0L_gpu = _to_gpu(X0L) if GPU_AVAILABLE else X0L
            inv = pinv(X0L_gpu.T @ X0L_gpu)
            X0L_plus = inv @ X0L_gpu.T
            proj = xp.eye(nT) - X0L_gpu @ X0L_plus
            X0L_T_plus = X0L_gpu @ inv
            temp2 = (X0Lc_gpu.T @ X0L_T_plus) @ SL

        X0Lc_T_proj = X0Lc_gpu.T @ proj
        temp1 = X0Lc_T_proj / (lambda_tilde * nT)

        # Calculate nu1
        term_b = temp1 @ phi_b
        nu1 = xp.concatenate([term_b.ravel(), -term_b.ravel()])

        # Calculate kappa1
        term_a = temp1 @ phi_a_iota
        ones_vec = xp.ones_like(term_a)
        kappa1 = xp.concatenate([(ones_vec - temp2 - term_a).ravel(), 
                                (ones_vec + temp2 + term_a).ravel()])

    nu = xp.concatenate((nu0, nu1))
    kappa = xp.concatenate((kappa0, kappa1))

    # Convert back to CPU for bound computation
    nu_cpu = _to_cpu(nu)
    kappa_cpu = _to_cpu(kappa)

    lv = -np.inf
    rv = np.inf

    for i in range(len(nu_cpu)):
        if nu_cpu[i] == 0:
            if kappa_cpu[i] < 0:
                return [np.inf, -np.inf]
        elif nu_cpu[i] > 0:
            val = kappa_cpu[i] / nu_cpu[i]
            if val < rv:
                rv = val
        else:
            val = kappa_cpu[i] / nu_cpu[i]
            if val > lv:
                lv = val

    return [lv, rv]


def compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a, b):
    """GPU optimized version of compute_Zt"""
    xp = cp if GPU_AVAILABLE and cp is not None else np
    
    # Convert inputs to appropriate arrays
    SM = _to_gpu(SM) if GPU_AVAILABLE else np.array(SM)
    a = _to_gpu(a) if GPU_AVAILABLE else np.array(a)
    b = _to_gpu(b) if GPU_AVAILABLE else np.array(b)
    xi_uv = _to_gpu(xi_uv) if GPU_AVAILABLE else np.array(xi_uv)
    zeta_uv = _to_gpu(zeta_uv) if GPU_AVAILABLE else np.array(zeta_uv)
    
    omega0 = xp.array([])
    rho0 = xp.array([])
    omega1 = xp.array([])
    rho1 = xp.array([])

    xi_a_zeta = (xi_uv @ a) + zeta_uv
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
        omega1 = xp.concatenate([Dtc_xi_b.ravel(), -Dtc_xi_b.ravel()])
        rho1 = xp.concatenate([-Dtc_xi_a_zeta.ravel(), Dtc_xi_a_zeta.ravel()])

    omega = xp.concatenate((omega0, omega1))
    rho = xp.concatenate((rho0, rho1))

    # Convert back to CPU for bound computation
    omega_cpu = _to_cpu(omega)
    rho_cpu = _to_cpu(rho)

    lt = -np.inf
    rt = np.inf

    for i in range(len(omega_cpu)):
        if omega_cpu[i] == 0:
            if rho_cpu[i] < 0:
                return [np.inf, -np.inf]
        elif omega_cpu[i] > 0:
            val = rho_cpu[i] / omega_cpu[i]
            if val < rt:
                rt = val
        else:
            val = rho_cpu[i] / omega_cpu[i]
            if val > lt:
                lt = val

    return [lt, rt]


def divide_and_conquer_TF_recursive(X, X0, a, b, M_obs, N, nT, K, p, B, Q, 
                         lambda_0, lambda_tilde, ak_weights, z_min, z_max):
    """
    Divide and conquer algorithm for Transfer Learning (GPU optimized)
    """
    intervals = []
    EPS = 1e-6
    
    def recursive_search(z_lo, z_hi):
        if z_lo > z_hi:
            return
            
        z_mid = (z_lo + z_hi) / 2.0
        
        # Calculate Y at z_mid
        Y_z = a + b * z_mid
        Y0_z = Q @ Y_z
        
        # Get TransFusion solution
        theta_z, w_z, delta_z, beta0_z = transfer_learning_hdr.TransFusion(
            X, Y_z, X0, Y0_z, B, N, p, K, lambda_0, lambda_tilde, ak_weights
        )
        
        # Construct components
        theta_O, S_O, O, X_O, Oc, X_Oc = utils.construct_thetaO_SO_O_XO_Oc_XOc(theta_z, X)
        delta_L, S_L, L, X0_L, Lc, X0_Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(delta_z, X0)
        beta_M, M, S_M, Mc = utils.construct_betaM_M_SM_Mc(beta0_z)
        
        # Calculate auxiliary variables
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(
            X, S_O, O, X_O, X0, S_L, L, X0_L, p, B, Q,
            lambda_0, lambda_tilde, 
            np.concatenate([w * np.ones(p) for w in ak_weights] + [np.ones(p)]),
            N, nT
        )
        
        # Compute bounds
        lu, ru = compute_Zu_3(S_O, O, X_O, Oc, X_Oc, a, b, lambda_0, 
                           np.concatenate([w * np.ones(p) for w in ak_weights] + [np.ones(p)]), N)
        lv, rv = compute_Zv_3(S_L, L, X0_L, Lc, X0_Lc, phi_u, iota_u, a, b, lambda_tilde, nT)
        lt, rt = compute_Zt_3(M, S_M, Mc, xi_uv, zeta_uv, a, b)
        
        l_bound = max(lu, lv, lt)
        r_bound = min(ru, rv, rt)
        
        if r_bound < l_bound:
            # Infeasible region, split
            recursive_search(z_lo, z_mid - EPS)
            recursive_search(z_mid + EPS, z_hi)
            return
        
        if M == M_obs:
            # Feasible region with correct selection
            intervals.append((l_bound, r_bound))
        
        # Continue searching outside feasible region
        recursive_search(z_lo, l_bound - EPS)
        recursive_search(r_bound + EPS, z_hi)
    
    recursive_search(z_min, z_max)
    
    # Merge overlapping intervals
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current_l, current_r in intervals[1:]:
        last_l, last_r = merged[-1]
        if current_l <= last_r + EPS:
            merged[-1] = (last_l, max(last_r, current_r))
        else:
            merged.append((current_l, current_r))
    
    return merged


def PTL_SI_TF_recursive(X0, Y0, XS_list, YS_list, lambda_0, lambda_tilde, ak_weights, 
              SigmaS_list, Sigma0, z_min=-20, z_max=20):
    """
    Post-selection Transfer Learning Statistical Inference (GPU optimized)
    """
    K = len(YS_list)
    nS = YS_list[0].shape[0]
    nT = Y0.shape[0]
    N = nS * K + nT
    p = X0.shape[1]

    X = utils.construct_X(XS_list, X0, p, K)
    Y = np.concatenate(YS_list + [Y0])
    B = utils.construct_B(K, p, nS, nT)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)

    theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(
        X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights
    )
    
    M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]

    if len(M_obs) == 0:
        return None
    
    X0M = X0[:, M_obs]
    Q = utils.construct_Q(nT, N)

    p_sel_list = []
    
    for j in M_obs:
        etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
        a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
        
        intervals = divide_and_conquer_TF_recursive(
            X, X0, a, b, M_obs, N, nT, K, p, B, Q, 
            lambda_0, lambda_tilde, ak_weights, z_min, z_max
        )
        
        pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, pj_sel))
    
    return p_sel_list


def PTL_SI_TF(X0, Y0, XS_list, YS_list, lambda_0, lambda_tilde, ak_weights, SigmaS_list, Sigma0, z_min=-20, z_max=20):
    K = len(YS_list)
    nS = YS_list[0].shape[0]
    nT = Y0.shape[0]
    N = nS * K + nT
    p = X0.shape[1]

    X = utils.construct_X(XS_list, X0, p, K)
    Y = np.concatenate(YS_list + [Y0])
    B = utils.construct_B(K, p, nS, nT)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)

    theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights)
    M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]

    if len(M_obs) == 0:
        return None
    
    X0M = X0[:, M_obs]
    Q = utils.construct_Q(nT, N)

    p_sel_list = []
    
    for j in M_obs:
        etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
        a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
        intervals = divide_and_conquer_TF(X, X0, a, b, M_obs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max)
        pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

        p_sel_list.append((j, pj_sel))
    
    return p_sel_list

def PTL_SI_TF_randj(X0, Y0, XS_list, YS_list, lambda_0, lambda_tilde, ak_weights, SigmaS_list, Sigma0, z_min=-20, z_max=20):
    K = len(YS_list)
    nS = YS_list[0].shape[0]
    nT = Y0.shape[0]
    N = nS * K + nT
    p = X0.shape[1]

    X = utils.construct_X(XS_list, X0, p, K)
    Y = np.concatenate(YS_list + [Y0])
    B = utils.construct_B(K, p, nS, nT)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)

    theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights)
    M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]
    # print(f"Number of non-zero elements in beta_hat: {np.count_nonzero(beta0_hat)} - len: {(len(beta0_hat))}")
    # print(f"Number of non-zero elements in w_hat: {np.count_nonzero(w_hat)} - len: {(len(w_hat))}")
    # print(f"Number of non-zero elements in delta_hat: {np.count_nonzero(delta_hat)} - len: {(len(delta_hat))}")
    # print(f"Number of non-zero elements in theta_hat: {np.count_nonzero(theta_hat)} - len: {(len(theta_hat))}")
    # print(f'------------------------------------------------------------------------------------------------------')

    if len(M_obs) == 0:
        return None
    
    X0M = X0[:, M_obs]
    Q = utils.construct_Q(nT, N)

    j = random.choice(M_obs)

    etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
    a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
    intervals = divide_and_conquer_TF(X, X0, a, b, M_obs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max)
    pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
    
    return pj_sel

#___________________________________________________________________________________________________


def divide_and_conquer_OTL(XI, X0, a, b, Mobs, nI, nT, p, Q, P, lambda_w, lambda_del, z_min, z_max):
    z = z_min
    intervals = []
    while z < z_max:
        Yz = a + b*z
        Yz = Yz.ravel()
        Y0z = Q @ Yz
        YIz = P @ Yz
        wz, dz, bz = transfer_learning_hdr.OracleTransLasso(XI, YIz, X0, Y0z, lambda_w, lambda_del)

        wO, SO, O, XIO, Oc, XIOc = utils.construct_wO_SO_O_XIO_Oc_XIOc(wz, XI)
        deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(dz, X0)
        betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)

        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta_otl(XI, SO, O, XIO, X0, SL, L, X0L, p, Q, P, lambda_w, lambda_del, nI, nT)
        
        # utils.check_KKT_w(XIO, XIOc, YIz, O, Oc, wO, SO, lambda_w, nI)
        
        lu, ru = sub_prob.compute_Zu_otl(SO, O, XIO, Oc, XIOc, a, b, P, lambda_w, nI)

        lv, rv = sub_prob.compute_Zv(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_del, nT)

        lt, rt = sub_prob.compute_Zt(M, SM, Mc, xi_uv, zeta_uv, a, b)

        r = min(ru, rv, rt)
        l = max(lu, lv, lt)
        if r < l or r < z: 
            print ('Err')
            return []

        if M == Mobs:
            intervals.append((l, r))

        z = r + 1e-4

    return intervals


def PTL_SI_OTL(X0, Y0, XI_list, YI_list, lambda_w, lambda_del, SigmaI_list, Sigma0, z_min=-20, z_max=20):
    nS = YI_list[0].shape[0]
    nT = Y0.shape[0]
    nI = nS * len(YI_list)
    N = nT + nI
    p = X0.shape[1]

    XI = np.concatenate(XI_list)
    YI = np.concatenate(YI_list)
    Y = np.concatenate(YI_list + [Y0])
    Sigma = utils.construct_Sigma(SigmaI_list, Sigma0)

    w_hat, delta_hat, beta0_hat = transfer_learning_hdr.OracleTransLasso(XI, YI, X0, Y0, lambda_w, lambda_del)    
    M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]
    
    if len(M_obs) == 0:
        return None
    
    X0M = X0[:, M_obs]
    Q = utils.construct_Q(nT, N)
    P = utils.construct_P(nT, nI)

    p_sel_list = []
    
    for j in M_obs:
        etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
        a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
        intervals = divide_and_conquer_OTL(XI, X0, a, b, M_obs, nI, nT, p, Q, P, lambda_w, lambda_del, z_min, z_max)
        pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, pj_sel))
    
    return p_sel_list


def PTL_SI_OTL_randj(X0, Y0, XI_list, YI_list, lambda_w, lambda_del, SigmaI_list, Sigma0, z_min=-20, z_max=20):
    nS = YI_list[0].shape[0]
    nT = Y0.shape[0]
    nI = nS * len(YI_list)
    N = nT + nI
    p = X0.shape[1]

    XI = np.concatenate(XI_list)
    YI = np.concatenate(YI_list)
    Y = np.concatenate(YI_list + [Y0])
    Sigma = utils.construct_Sigma(SigmaI_list, Sigma0)

    w_hat, delta_hat, beta0_hat = transfer_learning_hdr.OracleTransLasso(XI, YI, X0, Y0, lambda_w, lambda_del)   
    # print(f"Number of non-zero elements in beta_hat: {np.count_nonzero(beta0_hat)} - len: {(len(beta0_hat))}")
    # print(f"Number of non-zero elements in w_hat: {np.count_nonzero(w_hat)} - len: {(len(w_hat))}")
    # print(f"Number of non-zero elements in delta_hat: {np.count_nonzero(delta_hat)} - len: {(len(delta_hat))}")
    # print(f'------------------------------------------------------------------------------------------------------') 
    M_obs = [i for i in range(p) if beta0_hat[i] != 0.0]
    
    if len(M_obs) == 0:
        return None
    
    X0M = X0[:, M_obs]
    Q = utils.construct_Q(nT, N)
    P = utils.construct_P(nT, nI)

    j = random.choice(M_obs)

    etaj, etajTY = utils.construct_test_statistic(j, X0M, Y, M_obs, nT, N)
    a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
    intervals = divide_and_conquer_OTL(XI, X0, a, b, M_obs, nI, nT, p, Q, P, lambda_w, lambda_del, z_min, z_max)
    pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return pj_sel
