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
from numpy.linalg import pinv

def compute_Zu_2(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
    a_tilde_O = a_tilde[O]
    a_tilde_Oc = a_tilde[Oc]

    if len(O) > 0:
        XO_plus = pinv(XO)
        inv = XO_plus.T @ XO_plus
        XO_plus_b = XO_plus @ b
        psi0 = (-SO * XO_plus_b).ravel()
        XO_plus_a = XO_plus @ a
        gamma0_term = inv @ (a_tilde_O * SO)
        gamma0 = (SO * XO_plus_a - N * lambda_0 * SO * gamma0_term).ravel()
    else:
        psi0 = np.array([])
        gamma0 = np.array([])

    if len(Oc) > 0:
        if len(O) == 0:
            proj_b = b
            proj_a = a
            temp2 = 0
        else:
            XO_XO_plus_b = XO @ XO_plus_b
            proj_b = b - XO_XO_plus_b
            XO_XO_plus_a = XO @ XO_plus_a
            proj_a = a - XO_XO_plus_a
            XO_inv = XO @ inv
            temp2 = (XOc.T @ XO_inv @ (a_tilde_O * SO)) / a_tilde_Oc

        XOc_T_proj_b = XOc.T @ proj_b
        temp1_b = XOc_T_proj_b / a_tilde_Oc / (lambda_0 * N)
        psi1 = np.concatenate([temp1_b.ravel(), -temp1_b.ravel()])

        XOc_T_proj_a = XOc.T @ proj_a
        temp1_a = XOc_T_proj_a / a_tilde_Oc / (lambda_0 * N)
        ones_vec = np.ones_like(temp1_a)
        gamma1 = np.concatenate([(ones_vec - temp2 - temp1_a).ravel(), (ones_vec + temp2 + temp1_a).ravel()])
    else:
        psi1 = np.array([])
        gamma1 = np.array([])

    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    mask_zero = psi == 0
    if np.any(gamma[mask_zero] < 0):
        return [np.inf, -np.inf]

    mask_pos = psi > 0
    ru = np.min(gamma[mask_pos] / psi[mask_pos]) if np.any(mask_pos) else np.inf

    mask_neg = psi < 0
    lu = np.max(gamma[mask_neg] / psi[mask_neg]) if np.any(mask_neg) else -np.inf

    return [lu, ru]

def compute_Zv_2(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT):
    phi_a_iota = (phi_u @ a) + iota_u
    phi_b = phi_u @ b

    if len(L) > 0:
        X0L_plus = pinv(X0L)
        inv = X0L_plus.T @ X0L_plus
        X0L_plus_phi_b = X0L_plus @ phi_b
        nu0 = (-SL * X0L_plus_phi_b).ravel()
        X0L_plus_phi_a_iota = X0L_plus @ phi_a_iota
        kappa0_term = inv @ SL
        kappa0 = (SL * X0L_plus_phi_a_iota - nT * lambda_tilde * SL * kappa0_term).ravel()
    else:
        nu0 = np.array([])
        kappa0 = np.array([])

    if len(Lc) > 0:
        if len(L) == 0:
            proj_phi_b = phi_b
            proj_phi_a_iota = phi_a_iota
            temp2 = 0
        else:
            X0L_X0L_plus_phi_b = X0L @ X0L_plus_phi_b
            proj_phi_b = phi_b - X0L_X0L_plus_phi_b
            X0L_X0L_plus_phi_a_iota = X0L @ X0L_plus_phi_a_iota
            proj_phi_a_iota = phi_a_iota - X0L_X0L_plus_phi_a_iota
            X0L_inv = X0L @ inv
            temp2 = X0Lc.T @ X0L_inv @ SL

        X0Lc_T_proj_phi_b = X0Lc.T @ proj_phi_b
        temp1_b = X0Lc_T_proj_phi_b / (lambda_tilde * nT)
        nu1 = np.concatenate([temp1_b.ravel(), -temp1_b.ravel()])

        X0Lc_T_proj_phi_a_iota = X0Lc.T @ proj_phi_a_iota
        temp1_a = X0Lc_T_proj_phi_a_iota / (lambda_tilde * nT)
        ones_vec = np.ones_like(temp1_a)
        kappa1 = np.concatenate([(ones_vec - temp2 - temp1_a).ravel(), (ones_vec + temp2 + temp1_a).ravel()])
    else:
        nu1 = np.array([])
        kappa1 = np.array([])

    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))

    mask_zero = nu == 0
    if np.any(kappa[mask_zero] < 0):
        return [np.inf, -np.inf]

    mask_pos = nu > 0
    rv = np.min(kappa[mask_pos] / nu[mask_pos]) if np.any(mask_pos) else np.inf

    mask_neg = nu < 0
    lv = np.max(kappa[mask_neg] / nu[mask_neg]) if np.any(mask_neg) else -np.inf

    return [lv, rv]

def compute_Zt_2(M, SM, Mc, xi_uv, zeta_uv, a, b):
    xi_a_zeta = (xi_uv @ a) + zeta_uv
    xi_b = xi_uv @ b

    if len(M) > 0:
        Dt_xi_a_zeta = xi_a_zeta[M]
        Dt_xi_b = xi_b[M]
        omega0 = (-SM * Dt_xi_b).ravel()
        rho0 = (SM * Dt_xi_a_zeta).ravel()
    else:
        omega0 = np.array([])
        rho0 = np.array([])

    if len(Mc) > 0:
        Dtc_xi_a_zeta = xi_a_zeta[Mc]
        Dtc_xi_b = xi_b[Mc]
        omega1 = np.concatenate([Dtc_xi_b.ravel(), -Dtc_xi_b.ravel()])
        rho1 = np.concatenate([-Dtc_xi_a_zeta.ravel(), Dtc_xi_a_zeta.ravel()])
    else:
        omega1 = np.array([])
        rho1 = np.array([])

    omega = np.concatenate((omega0, omega1))
    rho = np.concatenate((rho0, rho1))

    mask_zero = omega == 0
    if np.any(rho[mask_zero] < 0):
        return [np.inf, -np.inf]

    mask_pos = omega > 0
    rt = np.min(rho[mask_pos] / omega[mask_pos]) if np.any(mask_pos) else np.inf

    mask_neg = omega < 0
    lt = np.max(rho[mask_neg] / omega[mask_neg]) if np.any(mask_neg) else -np.inf

    return [lt, rt]
    

import utils
import transfer_learning_hdr
import sub_prob
import random
import numpy as np


def divide_and_conquer_TF_op(X, X0, a, b, Mobs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max):
    z = z_min
    a_tilde = np.concatenate([ak_weights[k] * np.ones(p) for k in range(K)] + [np.ones(p)]).reshape(-1, 1)
    intervals = []
    while z < z_max:
        Yz = a + b*z
        Yz = Yz.ravel()
        Y0z = Q @ Yz
        tz, wz, dz, bz = transfer_learning_hdr.TransFusion(X, Yz, X0, Y0z, B, N, p, K, lambda_0, lambda_tilde, ak_weights)

        thetaO, SO, O, XO, Oc, XOc = utils.construct_thetaO_SO_O_XO_Oc_XOc(tz, X)
        deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(dz, X0)
        betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(X, SO, O, XO, X0, SL, L, X0L, p, B, Q, lambda_0, lambda_tilde, a_tilde, N, nT)
        
        # utils.check_KKT_theta(XO, XOc, Yz, O, Oc, thetaO, SO, lambda_0, a_tilde, N)
        # utils.check_KKT_delta(X0L, X0Lc, Yz, L, Lc, deltaL, SL, phi_u, iota_u, lambda_tilde, nT)
       
        lu, ru = compute_Zu_2(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N)

        lv, rv = compute_Zv_2(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT)

        lt, rt = compute_Zt_2(M, SM, Mc, xi_uv, zeta_uv, a, b)

        r = min(ru, rv, rt)
        l = max(lu, lv, lt)
        if r < l or r < z: 
            print ('Err')
            return []

        if M == Mobs:
            intervals.append((l, r))

        z = r + 1e-4

    return intervals


def PTL_SI_TF_op(X0, Y0, XS_list, YS_list, lambda_0, lambda_tilde, ak_weights, SigmaS_list, Sigma0, z_min=-20, z_max=20):
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
        intervals = divide_and_conquer_TF_op(X, X0, a, b, M_obs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max)
        pj_sel = utils.calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

        p_sel_list.append((j, pj_sel))
    
    return p_sel_list



def divide_and_conquer_TF(X, X0, a, b, Mobs, N, nT, K, p, B, Q, lambda_0, lambda_tilde, ak_weights, z_min, z_max):
    z = z_min
    a_tilde = np.concatenate([ak_weights[k] * np.ones(p) for k in range(K)] + [np.ones(p)]).reshape(-1, 1)
    intervals = []
    while z < z_max:
        Yz = a + b*z
        Yz = Yz.ravel()
        Y0z = Q @ Yz
        tz, wz, dz, bz = transfer_learning_hdr.TransFusion(X, Yz, X0, Y0z, B, N, p, K, lambda_0, lambda_tilde, ak_weights)

        thetaO, SO, O, XO, Oc, XOc = utils.construct_thetaO_SO_O_XO_Oc_XOc(tz, X)
        deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(dz, X0)
        betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(X, SO, O, XO, X0, SL, L, X0L, p, B, Q, lambda_0, lambda_tilde, a_tilde, N, nT)
        
        # utils.check_KKT_theta(XO, XOc, Yz, O, Oc, thetaO, SO, lambda_0, a_tilde, N)
        # utils.check_KKT_delta(X0L, X0Lc, Yz, L, Lc, deltaL, SL, phi_u, iota_u, lambda_tilde, nT)
       
        lu, ru = sub_prob.compute_Zu(SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N)

        lv, rv = sub_prob.compute_Zv(SL, L, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT)

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
