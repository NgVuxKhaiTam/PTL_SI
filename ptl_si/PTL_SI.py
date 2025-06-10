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
        Tính bound chung, chạy trên CPU.
        """
        psi = psi.ravel()
        gamma = gamma.ravel()
        lu, ru = -np.inf, np.inf
        for pi, gi in zip(psi, gamma):
            if pi == 0:
                if gi < 0:
                    return np.inf, -np.inf
            elif pi > 0:
                ru = min(ru, gi/pi)
            else:
                lu = max(lu, gi/pi)
        return lu, ru

    def compute_bounds_3(self, psi, gamma):
        psi_arr = self._get_array(psi)
        gamma_arr = self._get_array(gamma)
        lu, ru = self.compute_bounds(_to_cpu(psi_arr), _to_cpu(gamma_arr))
        return [lu, ru]

    def compute_Zu_3(self, SO, O, XO, Oc, XOc, a, b, lambda_0, a_tilde, N):
        xp = self.xp
        # Prepare arrays
        SO = self._get_array(SO).ravel()
        a = self._get_array(a).ravel()
        b = self._get_array(b).ravel()
        a_tilde = self._get_array(a_tilde).ravel()
        O = np.asarray(O)
        Oc = np.asarray(Oc)
        XO_gpu = self._get_array(XO) if O.size > 0 else None
        XOc_gpu = self._get_array(XOc) if Oc.size > 0 else None

        total = len(O) + 2*len(Oc)
        if total == 0:
            return [-xp.inf, xp.inf]

        psi = xp.empty(total)
        gamma = xp.empty(total)
        idx = 0

        # Case O
        if O.size > 0:
            inv = self.pinv(XO_gpu.T @ XO_gpu)
            XO_plus = inv @ XO_gpu.T
            psi_o = -SO * (XO_plus @ b).ravel()
            gamma_o = SO*(XO_plus@ a).ravel() - N*lambda_0*SO*(inv @ (a_tilde[O]*SO)).ravel()
            psi[idx:idx+len(O)] = psi_o
            gamma[idx:idx+len(O)] = gamma_o
            idx += len(O)

        # Case Oc
        if Oc.size > 0:
            # projection
            if O.size > 0:
                proj = xp.eye(N) - XO_gpu @ (self.pinv(XO_gpu.T@XO_gpu) @ XO_gpu.T)
            else:
                proj = xp.eye(N)
            XOc_proj = XOc_gpu.T @ proj
            denom = (a_tilde[Oc] * lambda_0 * N)[:,None]
            temp = XOc_proj / denom
            tb = (temp @ b).ravel()
            ta = (temp @ a).ravel()
            ones = xp.ones_like(ta)
            # positive
            psi[idx:idx+len(Oc)] = tb
            gamma[idx:idx+len(Oc)] = ones - (XOc_proj@(a_tilde[O]*SO) if O.size>0 else 0)/a_tilde[Oc] - ta
            # negative
            psi[idx+len(Oc):idx+2*len(Oc)] = -tb
            gamma[idx+len(Oc):idx+2*len(Oc)] = ones + (XOc_proj@(a_tilde[O]*SO) if O.size>0 else 0)/a_tilde[Oc] + ta

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
        X0L_gpu = self._get_array(X0L) if L.size>0 else None
        X0Lc_gpu = self._get_array(X0Lc) if Lc.size>0 else None

        total = len(L)+2*len(Lc)
        if total==0:
            return [-xp.inf, xp.inf]

        nu = xp.empty(total)
        kappa = xp.empty(total)
        idx=0
        fai = (phi_u@a)+iota_u
        fb = phi_u@b
        # L
        if L.size>0:
            inv = self.pinv(X0L_gpu.T@X0L_gpu)
            plus = inv@X0L_gpu.T
            nu[idx:idx+len(L)] = -SL*(plus@fb).ravel()
            kappa[idx:idx+len(L)] = SL*(plus@fai).ravel() - nT*lambda_tilde*SL*(inv@SL).ravel()
            idx+=len(L)
        # Lc
        if Lc.size>0:
            if L.size>0:
                proj = xp.eye(nT) - X0L_gpu@ (self.pinv(X0L_gpu.T@X0L_gpu)@X0L_gpu.T)
            else:
                proj = xp.eye(nT)
            temp = (X0Lc_gpu.T@proj)/(lambda_tilde*nT)
            tb = (temp@fb).ravel()
            ta = (temp@fai).ravel()
            ones = xp.ones_like(ta)
            nu[idx:idx+len(Lc)] = tb
            nu[idx+len(Lc):idx+2*len(Lc)] = -tb
            kappa[idx:idx+len(Lc)] = ones - (X0Lc_gpu.T@(SL if L.size>0 else 0)).ravel()/1 - ta
            kappa[idx+len(Lc):idx+2*len(Lc)] = ones + (X0Lc_gpu.T@(SL if L.size>0 else 0)).ravel()/1 + ta

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
        total = len(M)+2*len(Mc)
        if total==0:
            return [-xp.inf, xp.inf]
        omega = xp.empty(total)
        rho = xp.empty(total)
        idx=0
        fi = (xi_uv@a)+zeta_uv
        fb = xi_uv@b
        if M.size>0:
            omega[idx:idx+len(M)] = -SM*fb[M].ravel()
            rho[idx:idx+len(M)] = SM*fi[M].ravel()
            idx+=len(M)
        if Mc.size>0:
            tb = fb[Mc].ravel()
            ta = fi[Mc].ravel()
            omega[idx:idx+len(Mc)] = tb
            omega[idx+len(Mc):idx+2*len(Mc)] = -tb
            rho[idx:idx+len(Mc)] = -ta
            rho[idx+len(Mc):idx+2*len(Mc)] = ta
        return self.compute_bounds_3(omega, rho)

# Instance toán tử toàn cục
_optimizer = OptimizedCompute()

def compute_Zu_3(*args, **kwargs):
    return _optimizer.compute_Zu_3(*args, **kwargs)

def compute_Zv_3(*args, **kwargs):
    return _optimizer.compute_Zv_3(*args, **kwargs)

def compute_Zt_3(*args, **kwargs):
    return _optimizer.compute_Zt_3(*args, **kwargs)


def divide_and_conquer_TF_recursive(
    X, X0, a, b, Mobs, N, nT, K, p, B, Q,
    lambda_0, lambda_tilde, ak_weights,
    z_min, z_max, use_gpu=True
):
    """
    Phiên bản chia để trị cho bài TF, chọn CPU/GPU.
    """
    use_gpu = use_gpu and GPU_AVAILABLE
    xp = cp if (use_gpu and cp is not None) else np

    # Đưa dữ liệu lên GPU nếu cần
    X_gpu = _to_gpu(X)
    X0_gpu = _to_gpu(X0)
    B_gpu = _to_gpu(B)
    Q_gpu = _to_gpu(Q)

    # Chuyển a, b lên GPU
    a_gpu = _to_gpu(a).ravel()
    b_gpu = _to_gpu(b).ravel()
    # Tạo a_tilde cho biến kiểm soát
    a_tilde = xp.concatenate([w * xp.ones(p) for w in ak_weights] + [xp.ones(p)]).reshape(-1, 1)

    intervals = []
    EPS = 1e-6

    def rec(z_lo, z_hi):
        if z_lo > z_hi:
            return
        z_mid = (z_lo + z_hi) / 2.0
        # Tính Y trên GPU
        Yz = (a_gpu + b_gpu * z_mid).ravel()
        Y0z = Q_gpu @ Yz
        # Chuyển về CPU cho routines
        Xc = _to_cpu(X_gpu)
        Yc = _to_cpu(Yz)
        X0c = _to_cpu(X0_gpu)
        Y0c = _to_cpu(Y0z)
        Bc = _to_cpu(B_gpu)

        tz, wz, dz, bz = transfer_learning_hdr.TransFusion(
            Xc, Yc, X0c, Y0c, Bc,
            N, p, K, lambda_0, lambda_tilde, ak_weights
        )
        thetaO, SO, O, XO, Oc, XOc = utils.construct_thetaO_SO_O_XO_Oc_XOc(tz, Xc)
        deltaL, SL, L, X0L, Lc, X0Lc = utils.construct_detlaL_SL_L_X0L_Lc_X0Lc(dz, X0c)
        betaM, M, SM, Mc = utils.construct_betaM_M_SM_Mc(bz)
        phi_u, iota_u, xi_uv, zeta_uv = sub_prob.calculate_phi_iota_xi_zeta(
            Xc, SO, O, XO, X0c,
            SL, L, X0L, p, Bc, _to_cpu(Q_gpu),
            lambda_0, lambda_tilde, _to_cpu(a_tilde),
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
    if not intervals:
        return []
    intervals.sort(key=lambda x:x[0])
    merged=[intervals[0]]
    for cl,cr in intervals[1:]:
        il,ir=merged[-1]
        if cl<=ir+EPS:
            merged[-1]=(il,max(ir,cr))
        else:
            merged.append((cl,cr))
    return merged


def PTL_SI_TF_recursive(
    X0, Y0, XS_list, YS_list,
    lambda_0, lambda_tilde, ak_weights,
    SigmaS_list, Sigma0,
    z_min=-20, z_max=20, use_gpu=True
):
    """
    Hàm chính PTL-SI-TF, chọn CPU/GPU.
    """
    use_gpu = use_gpu and GPU_AVAILABLE
    # Chuyển datasets CPU
    X0c=_to_cpu(X0); Y0c=_to_cpu(Y0)
    XS=[_to_cpu(x) for x in XS_list]
    YS=[_to_cpu(y).ravel() for y in YS_list]
    K=len(YS); nS=YS[0].shape[0]; nT=Y0c.shape[0]
    N=nS*K+nT; p=X0c.shape[1]
    # Build X,Y,B,Q
    X=utils.construct_X(XS,X0c,p,K)
    Y=np.concatenate(YS+[Y0c])
    B=utils.construct_B(K,p,nS,nT)
    Q=utils.construct_Q(nT,N)
    # First TransFusion
    theta_hat,w_hat,delta_hat,beta0_hat=transfer_learning_hdr.TransFusion(
        X,Y,X0c,Y0c,B,N,p,K,lambda_0,lambda_tilde,ak_weights)
    M_obs=[i for i in range(p) if beta0_hat[i]!=0]
    if not M_obs: return None
    out=[]
    for j in M_obs:
        etaj,etajTY=utils.construct_test_statistic(j,X0c[:,M_obs],Y,M_obs,nT,N)
                # Không ép về scalar, giữ etaj và etajTY dưới dạng mảng nếu cần
        a,b = utils.calculate_a_b(etaj, Y, utils.construct_Sigma(SigmaS_list, Sigma0), N(etaj, Y, utils.construct_Sigma(SigmaS_list, Sigma0), N)
        intervals = divide_and_conquer_TF_recursive(
            X, X0c, a, b, M_obs, N, nT, K, p, B, Q,
            lambda_0, lambda_tilde, ak_weights, z_min, z_max,
            use_gpu=use_gpu
        )
        pj = utils.calculate_TN_p_value(
            intervals, etaj, etajTY, utils.construct_Sigma(SigmaS_list, Sigma0), 0
        )
        out.append((j, pj))
    return out




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
