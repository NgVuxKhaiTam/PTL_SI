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
    
    # Convert inputs
    SO = _to_gpu(SO)
    a = _to_gpu(a)
    b = _to_gpu(b)
    a_tilde = _to_gpu(a_tilde)
    
    # 1-D indices
    O = np.asarray(O, dtype=np.int64).ravel()
    Oc = np.asarray(Oc, dtype=np.int64).ravel()
    a_tilde_O = a_tilde[O] if O.size > 0 else xp.array([])
    a_tilde_Oc = a_tilde[Oc] if Oc.size > 0 else xp.array([])

    psi0 = xp.array([])
    gamma0 = xp.array([])
    psi1 = xp.array([])
    gamma1 = xp.array([])

    # phần O
    if O.size > 0:
        XO_gpu = _to_gpu(XO)
        inv = pinv(XO_gpu.T @ XO_gpu)
        XO_plus = inv @ XO_gpu.T

        psi0 = (-SO * (XO_plus @ b)).ravel()
        gamma0 = (SO * (XO_plus @ a) - N * lambda_0 * SO * (inv @ (a_tilde_O * SO))).ravel()

    # phần Oc
    if Oc.size > 0:
        XOc_gpu = _to_gpu(XOc)
        if O.size == 0:
            proj = xp.eye(N)
            temp2 = 0
        else:
            XO_gpu = _to_gpu(XO)
            inv = pinv(XO_gpu.T @ XO_gpu)
            XO_plus = inv @ XO_gpu.T
            proj = xp.eye(N) - XO_gpu @ XO_plus
            temp2 = (XOc_gpu.T @ (XO_gpu @ inv)) @ (a_tilde_O * SO)
            temp2 = temp2 / a_tilde_Oc

        temp1 = (XOc_gpu.T @ proj) / (lambda_0 * N * a_tilde_Oc)
        term_b = temp1 @ b
        psi1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        term_a = temp1 @ a
        ones = xp.ones_like(term_a)
        gamma1 = np.concatenate([(ones - temp2 - term_a).ravel(), (ones + temp2 + term_a).ravel()])

    psi = xp.concatenate((psi0, psi1))
    gamma = xp.concatenate((gamma0, gamma1))

    psi_cpu = _to_cpu(psi)
    gamma_cpu = _to_cpu(gamma)

    lu, ru = -np.inf, np.inf
    for psi_i, gamma_i in zip(psi_cpu, gamma_cpu):
        if psi_i == 0:
            if gamma_i < 0:
                return [np.inf, -np.inf]
        elif psi_i > 0:
            ru = min(ru, gamma_i / psi_i)
        else:
            lu = max(lu, gamma_i / psi_i)
    return [lu, ru]


def compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    """GPU optimized version of compute_Zv"""
    xp = cp if GPU_AVAILABLE and cp is not None else np
    pinv = cp_pinv if GPU_AVAILABLE and cp is not None else np.linalg.pinv
    
    SL = _to_gpu(SL)
    a = _to_gpu(a)
    b = _to_gpu(b)
    phi_u = _to_gpu(phi_u)
    iota_u = _to_gpu(iota_u)

    # 1-D indices
    L  = np.asarray(L, dtype=np.int64).ravel()
    Lc = np.asarray(Lc, dtype=np.int64).ravel()

    phi_b = phi_u @ b
    phi_ai = (phi_u @ a) + iota_u

    nu0, kappa0 = xp.array([]), xp.array([])
    nu1, kappa1 = xp.array([]), xp.array([])

    if L.size > 0:
        inv = pinv(_to_gpu(X0L).T @ _to_gpu(X0L))
        X0L_plus = inv @ _to_gpu(X0L).T
        nu0 = (-SL * (X0L_plus @ phi_b)).ravel()
        kappa0 = (SL * (X0L_plus @ phi_ai) - nT * lambda_tilde * SL * (inv @ SL)).ravel()

    if Lc.size > 0:
        X0Lc_gpu = _to_gpu(X0Lc)
        if L.size == 0:
            proj = xp.eye(nT)
            temp2 = 0
        else:
            inv = pinv(_to_gpu(X0L).T @ _to_gpu(X0L))
            X0L_plus = inv @ _to_gpu(X0L).T
            proj = xp.eye(nT) - _to_gpu(X0L) @ X0L_plus
            temp2 = (X0Lc_gpu.T @ (_to_gpu(X0L) @ inv)) @ SL
        temp1 = (X0Lc_gpu.T @ proj) / (lambda_tilde * nT)
        term_b = temp1 @ phi_b
        nu1 = np.concatenate([term_b.ravel(), -term_b.ravel()])
        term_a = temp1 @ phi_ai
        ones = xp.ones_like(term_a)
        kappa1 = np.concatenate([(ones - temp2 - term_a).ravel(), (ones + temp2 + term_a).ravel()])

    nu = xp.concatenate((nu0, nu1))
    kappa = xp.concatenate((kappa0, kappa1))

    nu_cpu = _to_cpu(nu)
    kappa_cpu = _to_cpu(kappa)

    lv, rv = -np.inf, np.inf
    for nu_i, kappa_i in zip(nu_cpu, kappa_cpu):
        if nu_i == 0:
            if kappa_i < 0:
                return [np.inf, -np.inf]
        elif nu_i > 0:
            rv = min(rv, kappa_i / nu_i)
        else:
            lv = max(lv, kappa_i / nu_i)
    return [lv, rv]


def compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a, b):
    """GPU optimized version of compute_Zt"""
    xp = cp if GPU_AVAILABLE and cp is not None else np
    
    SM = _to_gpu(SM)
    a = _to_gpu(a)
    b = _to_gpu(b)
    xi_uv = _to_gpu(xi_uv)
    zeta_uv = _to_gpu(zeta_uv)

    # 1-D indices
    M  = np.asarray(M,  dtype=np.int64).ravel()
    Mc = np.asarray(Mc, dtype=np.int64).ravel()

    xi_b  = xi_uv @ b
    xi_az = (xi_uv @ a) + zeta_uv

    omega0, rho0 = xp.array([]), xp.array([])
    omega1, rho1 = xp.array([]), xp.array([])

    if M.size > 0:
        omega0 = (-SM * xi_b[M]).ravel()
        rho0   = (SM * xi_az[M]).ravel()
    if Mc.size > 0:
        w = xi_b[Mc]
        z = xi_az[Mc]
        omega1 = np.concatenate([w.ravel(), -w.ravel()])
        rho1   = np.concatenate([-z.ravel(), z.ravel()])

    omega = xp.concatenate((omega0, omega1))
    rho   = xp.concatenate((rho0, rho1))

    omega_cpu = _to_cpu(omega)
    rho_cpu   = _to_cpu(rho)

    lt, rt = -np.inf, np.inf
    for om_i, r_i in zip(omega_cpu, rho_cpu):
        if om_i == 0:
            if r_i < 0:
                return [np.inf, -np.inf]
        elif om_i > 0:
            rt = min(rt, r_i / om_i)
        else:
            lt = max(lt, r_i / om_i)
    return [lt, rt]

# ————————————————————————————————————————
# Hàm đệ quy tách ra khỏi main

def divide_and_conquer_TF_recursive(X, X0, a, b, M_obs, N,
                                    nT, K, p, B, Q,
                                    lambda_0, lambda_tilde,
                                    ak_weights, z_min, z_max):
    """
    Divide-and-conquer recursive for inference bounds
    """
    # ensure M_obs 1-D int array
    M_obs = np.asarray(M_obs, dtype=np.int64).ravel()
    intervals = []
    EPS = 1e-6

    def _search(z_lo, z_hi):
        if z_lo > z_hi:
            return
        z_mid = (z_lo + z_hi) / 2.0

        # 1) Fit TransFusion tại z_mid
        Y_z = a + b * z_mid
        Y0_z = Q @ Y_z
        theta_z, w_z, delta_z, beta0_z = transfer_learning_hdr.TransFusion(
            X, Y_z, X0, Y0_z, B, N, p, K, lambda_0, lambda_tilde, ak_weights
        )

        # 2) Xây dựng O, L, M và chỉ số 1-D
        theta_O, S_O, O, X_O, Oc, X_Oc = utils.construct_thetaO_SO_O_XO_Oc_XOc(theta_z, X)
        delta_L, S_L, L, X0_L, Lc, X0_Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(delta_z, X0)
        beta_M, M, S_M, Mc         = utils.construct_betaM_M_SM_Mc(beta0_z)
        O, Oc = np.asarray(O, dtype=np.int64).ravel(), np.asarray(Oc, dtype=np.int64).ravel()
        L, Lc = np.asarray(L, dtype=np.int64).ravel(), np.asarray(Lc, dtype=np.int64).ravel()
        M, Mc = np.asarray(M, dtype=np.int64).ravel(), np.asarray(Mc, dtype=np.int64).ravel()

        # 3) Tính phi_u, iota_u, xi_uv, zeta_uv
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(
            X, S_O, O, X_O, X0, S_L, L, X0_L, p, B, Q,
            lambda_0, lambda_tilde,
            np.concatenate([w * np.ones(p) for w in ak_weights] + [np.ones(p)]),
            N, nT
        )

        # 4) Compute individual bounds
        lu, ru = compute_Zu_3(S_O, O, X_O, Oc, X_Oc, a, b, lambda_0,
                              np.concatenate([w * np.ones(p) for w in ak_weights] + [np.ones(p)]), N)
        lv, rv = compute_Zv_3(S_L, L, X0_L, Lc, X0_Lc, phi_u, iota_u, a, b, lambda_tilde, nT)
        lt, rt = compute_Zt_3(M, S_M, Mc, xi_uv, zeta_uv, a, b)

        l_bound, r_bound = max(lu, lv, lt), min(ru, rv, rt)
        if r_bound < l_bound:
            _search(z_lo,   z_mid - EPS)
            _search(z_mid + EPS, z_hi)
            return

        if np.array_equal(M, M_obs):
            intervals.append((l_bound, r_bound))
        _search(z_lo,   l_bound - EPS)
        _search(r_bound + EPS, z_hi)

    _search(z_min, z_max)

    # Merge các intervals
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for l, r in intervals[1:]:
        L0, R0 = merged[-1]
        if l <= R0 + EPS:
            merged[-1] = (L0, max(R0, r))
        else:
            merged.append((l, r))
    return merged


def PTL_SI_TF_recursive(X0, Y0, XS_list, YS_list,
                        lambda_0, lambda_tilde, ak_weights,
                        SigmaS_list, Sigma0,
                        z_min=-20, z_max=20):
    """
    Post-selection Transfer Learning Statistical Inference (recursive)
    """
    K = len(YS_list)
    nS = YS_list[0].shape[0]
    nT = Y0.shape[0]
    N  = nS * K + nT
    p  = X0.shape[1]

    X = utils.construct_X(XS_list, X0, p, K)
    Y = np.concatenate(YS_list + [Y0])
    B = utils.construct_B(K, p, nS, nT)
    Q = utils.construct_Q(nT, N)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)

    # Fit TransFusion đầu tiên
    theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(
        X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights
    )
    M_obs = np.nonzero(beta0_hat)[0]
    if M_obs.size == 0:
        return None

    p_sel_list = []
    for j in M_obs:
        etaj, etajTY = utils.construct_test_statistic(j, X0[:, M_obs], Y, M_obs, nT, N)
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
