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


class OptimizedCompute:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.pinv = cp_pinv if self.use_gpu else np.linalg.pinv
        print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")

    def _get_array(self, arr):
        """
        Lấy mảng phù hợp với xp; nếu arr là NumPy và GPU đang bật, chuyển về Cupy; ngược lại nếu arr là Cupy và CPU, chuyển về NumPy.
        """
        if self.use_gpu and isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        if not self.use_gpu and hasattr(arr, 'get'):
            return arr.get()
        return arr

    def compute_bounds(self, psi, gamma):
        """
        Tính khoảng chung: giải bất phương trình gamma_i + z * psi_i >= 0 với tất cả i.
        Trả về (lower, upper).
        """
        lu, ru = -np.inf, np.inf
        for pi, gi in zip(psi, gamma):
            if pi == 0:
                if gi < 0:
                    return np.inf, -np.inf
            elif pi > 0:
                ru = min(ru, gi / pi)
            else:
                lu = max(lu, gi / pi)
        return lu, ru

    def compute_bounds_3(self, psi, gamma):
        psi_cpu = _to_cpu(psi).ravel()
        gamma_cpu = _to_cpu(gamma).ravel()
        lu, ru = self.compute_bounds(psi_cpu, gamma_cpu)
        return [lu, ru]

    def compute_Zu_3(self, SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
        xp = self.xp
        SO = xp.asarray(SO).ravel()
        a = xp.asarray(a).ravel()
        b = xp.asarray(b).ravel()
        a_tilde = xp.asarray(a_tilde).ravel()
        O = np.asarray(O, dtype=int)
        Oc = np.asarray(Oc, dtype=int)

        psi_list = []
        gamma_list = []

        # Phần quan sát O
        if O.size > 0:
            XO_gpu = xp.asarray(XO)
            inv = self.pinv(XO_gpu.T @ XO_gpu)
            XO_plus = inv @ XO_gpu.T
            psi0 = (-SO * (XO_plus @ b)).ravel()
            term_inv = inv @ (a_tilde[O] * SO)
            gamma0 = (SO * (XO_plus @ a) - N * lambda_0 * SO * term_inv).ravel()
            psi_list.append(psi0)
            gamma_list.append(gamma0)

        # Phần không quan sát Oc
        if Oc.size > 0:
            XOc_gpu = xp.asarray(XOc)
            if O.size > 0:
                XO_gpu = xp.asarray(XO)
                inv = self.pinv(XO_gpu.T @ XO_gpu)
                proj = xp.eye(N) - XO_gpu @ (inv @ XO_gpu.T)
                temp2 = (XOc_gpu.T @ (XO_gpu @ (inv))) @ (a_tilde[O] * SO)
                temp2 = (temp2 / a_tilde[Oc]).ravel()
            else:
                proj = xp.eye(N)
                temp2 = xp.zeros(Oc.size)

            XOc_proj = XOc_gpu.T @ proj
            temp1 = (XOc_proj / a_tilde[Oc][:, None]) / (lambda_0 * N)
            term_b = (temp1 @ b).ravel()
            term_a = (temp1 @ a).ravel()
            ones = xp.ones_like(term_a)

            psi1 = xp.concatenate([term_b, -term_b])
            gamma1 = xp.concatenate([ones - temp2 - term_a, ones + temp2 + term_a])
            psi_list.append(psi1)
            gamma_list.append(gamma1)

        if not psi_list:
            return [-xp.inf, xp.inf]
        psi = xp.concatenate(psi_list)
        gamma = xp.concatenate(gamma_list)
        return self.compute_bounds_3(psi, gamma)

    def compute_Zv_3(self, SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
        xp = self.xp
        SL = xp.asarray(SL).ravel()
        a = xp.asarray(a).ravel()
        b = xp.asarray(b).ravel()
        phi_u = xp.asarray(phi_u)
        iota_u = xp.asarray(iota_u).ravel()
        L = np.asarray(L, dtype=int)
        Lc = np.asarray(Lc, dtype=int)

        nu_list = []
        kappa_list = []
        phi_a = (phi_u @ a) + iota_u
        phi_b = phi_u @ b

        # Phần L
        if L.size > 0:
            X0L_gpu = xp.asarray(X0L)
            inv = self.pinv(X0L_gpu.T @ X0L_gpu)
            X0L_plus = inv @ X0L_gpu.T
            nu0 = (-SL * (X0L_plus @ phi_b)).ravel()
            term_inv = inv @ SL
            kappa0 = (SL * (X0L_plus @ phi_a) - nT * lambda_tilde * SL * term_inv).ravel()
            nu_list.append(nu0)
            kappa_list.append(kappa0)

        # Phần Lc
        if Lc.size > 0:
            X0Lc_gpu = xp.asarray(X0Lc)
            if L.size > 0:
                X0L_gpu = xp.asarray(X0L)
                inv = self.pinv(X0L_gpu.T @ X0L_gpu)
                proj = xp.eye(nT) - X0L_gpu @ (inv @ X0L_gpu.T)
                temp2 = (X0Lc_gpu.T @ (X0L_gpu @ (inv))) @ SL
                temp2 = temp2.ravel()
            else:
                proj = xp.eye(nT)
                temp2 = xp.zeros(Lc.size)

            temp1 = (X0Lc_gpu.T @ proj) / (lambda_tilde * nT)
            tb = (temp1 @ phi_b).ravel()
            ta = (temp1 @ phi_a).ravel()
            ones = xp.ones_like(ta)
            nu1 = xp.concatenate([tb, -tb])
            kappa1 = xp.concatenate([ones - temp2 - ta, ones + temp2 + ta])
            nu_list.append(nu1)
            kappa_list.append(kappa1)

        if not nu_list:
            return [-xp.inf, xp.inf]
        nu = xp.concatenate(nu_list)
        kappa = xp.concatenate(kappa_list)
        return self.compute_bounds_3(nu, kappa)

    def compute_Zt_3(self, M, SM, Mc, xi_uv, zeta_uv, a, b):
        xp = self.xp
        SM = xp.asarray(SM).ravel()
        a = xp.asarray(a).ravel()
        b = xp.asarray(b).ravel()
        xi_uv = xp.asarray(xi_uv)
        zeta_uv = xp.asarray(zeta_uv).ravel()
        M = np.asarray(M, dtype=int)
        Mc = np.asarray(Mc, dtype=int)

        omega_list = []
        rho_list = []
        xi_a = (xi_uv @ a) + zeta_uv
        xi_b = xi_uv @ b

        # Phần M
        if M.size > 0:
            omega0 = (-SM * xi_b[M]).ravel()
            rho0 = (SM * xi_a[M]).ravel()
            omega_list.append(omega0)
            rho_list.append(rho0)

        # Phần Mc
        if Mc.size > 0:
            tb = xi_b[Mc].ravel()
            ta = xi_a[Mc].ravel()
            omega1 = xp.concatenate([tb, -tb])
            rho1 = xp.concatenate([-ta, ta])
            omega_list.append(omega1)
            rho_list.append(rho1)

        if not omega_list:
            return [-xp.inf, xp.inf]
        omega = xp.concatenate(omega_list)
        rho = xp.concatenate(rho_list)
        return self.compute_bounds_3(omega, rho)

# Khởi tạo global optimizer
_optimizer = OptimizedCompute()

# Tiện ích
compute_Zu_3 = _optimizer.compute_Zu_3
compute_Zv_3 = _optimizer.compute_Zv_3
compute_Zt_3 = _optimizer.compute_Zt_3


def divide_and_conquer_TF_recursive(
    X, X0, a, b, Mobs, N, nT, K, p, B, Q,
    lambda_0, lambda_tilde, ak_weights,
    z_min, z_max, use_gpu=True
):
    """
    Phân chia để trị (recursive) cho bài TransFusion, chọn CPU/GPU.
    """
    use_gpu = use_gpu and GPU_AVAILABLE
    xp = cp if (use_gpu and cp is not None) else np

    # Đưa dữ liệu lên GPU nếu cần
    X_gpu = _to_gpu(X)
    X0_gpu = _to_gpu(X0)
    B_gpu = _to_gpu(B)
    Q_gpu = _to_gpu(Q)

    a_gpu = _to_gpu(a).ravel()
    b_gpu = _to_gpu(b).ravel()
    a_tilde = xp.concatenate([w * xp.ones(p) for w in ak_weights] + [xp.ones(p)]).reshape(-1, 1)

    intervals = []
    EPS = 1e-6

    def rec(z_lo, z_hi):
        if z_lo > z_hi:
            return
        z_mid = (z_lo + z_hi) / 2.0
        Yz = (a_gpu + b_gpu * z_mid).ravel()
        Y0z = Q_gpu @ Yz
        Xc, Yc = _to_cpu(X_gpu), _to_cpu(Yz)
        X0c, Y0c = _to_cpu(X0_gpu), _to_cpu(Y0z)
        Bc, Qc = _to_cpu(B_gpu), _to_cpu(Q_gpu)

        tz, wz, dz, bz = transfer_learning_hdr.TransFusion(
            Xc, Yc, X0c, Y0c, Bc,
            N, p, K, lambda_0, lambda_tilde, ak_weights
        )
        thetaO, SO, O, XO, Oc, XOc = utils.construct_thetaO_SO_O_XO_Oc_XOc(tz, Xc)
        deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_XLc(dz, X0c)
        betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(
            Xc, SO, O, XO, X0c,
            SL, L, X0L, p, Bc, Qc,
            lambda_0, lambda_tilde, a_tilde,
            N, nT
        )

        lu, ru = compute_Zu_3(SO, O, XO, Oc, XOc, a_gpu, b_gpu, lambda_0, a_tilde, N)
        lv, rv = compute_Zv_3(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a_gpu, b_gpu, lambda_tilde, nT)
        lt, rt = compute_Zt_3(M, SM, Mc, xi_uv, zeta_uv, a_gpu, b_gpu)

        l = float(max(lu, lv, lt))
        r = float(min(ru, rv, rt))
        if r < l:
            rec(z_lo, z_mid - EPS)
            rec(z_mid + EPS, z_hi)
            return

        if M == Mobs:
            intervals.append((l, r))

        rec(z_lo, l - EPS)
        rec(r + EPS, z_hi)

    rec(float(z_min), float(z_max))
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for cl, cr in intervals[1:]:
        il, ir = merged[-1]
        if cl <= ir + EPS:
            merged[-1] = (il, max(ir, cr))
        else:
            merged.append((cl, cr))
    return merged


def PTL_SI_TF_recursive(
    X0, Y0, XS_list, YS_list,
    lambda_0, lambda_tilde, ak_weights,
    SigmaS_list, Sigma0,
    z_min=-20, z_max=20, use_gpu=True
):
    """
    Hàm chính PTL-SI-TF (recursive), cho chọn CPU/GPU.
    """
    use_gpu = use_gpu and GPU_AVAILABLE
    X0c, Y0c = _to_cpu(X0), _to_cpu(Y0)
    XS = [_to_cpu(x) for x in XS_list]
    YS = [_to_cpu(y).ravel() for y in YS_list]
    K = len(YS)
    nS = YS[0].shape[0]
    nT = Y0c.shape[0]
    N = nS * K + nT
    p = X0c.shape[1]

    X = utils.construct_X(XS, X0c, p, K)
    Y = np.concatenate(YS + [Y0c])
    B = utils.construct_B(K, p, nS, nT)
    Q = utils.construct_Q(nT, N)

    theta_hat, w_hat, delta_hat, beta0_hat = transfer_learning_hdr.TransFusion(
        X, Y, X0c, Y0c, B, N, p, K, lambda_0, lambda_tilde, ak_weights
    )
    M_obs = [i for i in range(p) if beta0_hat[i] != 0]
    if not M_obs:
        return None

    results = []
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    for j in M_obs:
        etaj, etajTY = utils.construct_test_statistic(j, X0c[:, M_obs], Y, M_obs, nT, N)
        a, b = utils.calculate_a_b(etaj, Y, Sigma, N)
        intervals = divide_and_conquer_TF_recursive(
            X, X0c, a, b, M_obs, N, nT, K, p, B, Q,
            lambda_0, lambda_tilde, ak_weights,
            z_min, z_max, use_gpu=use_gpu
        )
        p_val = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        results.append((j, p_val))
    return results




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
