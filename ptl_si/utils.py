import numpy as np
from scipy.linalg import block_diag
from mpmath import mp
mp.dps = 500

def construct_X(XS_list, X0, p, K):
    blocks = []
    for k in range(K):
        Xk = XS_list[k]
        row_k = np.hstack([Xk if i == k else np.zeros((Xk.shape[0], p)) for i in range(K)] + [Xk])
        blocks.append(row_k)
    row_0 = np.hstack([np.zeros((X0.shape[0], K*p)), X0])
    blocks.append(row_0)
    return np.vstack(blocks)


def construct_B(K, p, nS, nT):
    B = np.hstack([np.tile(nS * np.eye(p), K), (nS * K + nT) * np.eye(p)])
    return B


def construct_Sigma(SigmaS_list, Sigma0):
    Sigma = block_diag(*SigmaS_list, Sigma0)
    return Sigma


def compute_adaptive_weights(K, nS, nT):
    ak = 8.0 * np.sqrt(nS / (K * nS + nT))
    return [ak] * K


def construct_Q(nT, N):
    Q = np.zeros((nT, N))
    Q[:, N - nT: ] = np.eye(nT)
    return Q


def construct_P(nT, nI):
    P = np.zeros((nI, nI + nT))
    P[ :, : nI] = np.eye(nI)
    return P

#______________________________________________________________________________________
# CONSTRUCT ACTIVE SET
def construct_thetaO_SO_O_XO_Oc_XOc(theta_hat, X):
    thetaO = []
    SO = []
    O = []
    Oc = []

    for i, val in enumerate(theta_hat):
        if val == 0.0:
            Oc.append(i)
        else:
            O.append(i)
            thetaO.append(val)
            SO.append(np.sign(val))


    XO = X[:, O] if O else np.zeros((X.shape[0], 0))
    XOc = X[:, Oc] if Oc else np.zeros((X.shape[0], 0))

    thetaO = np.array(thetaO).reshape(-1,1)
    SO = np.array(SO).reshape(-1,1)

    return thetaO, SO, O, XO, Oc, XOc


def construct_detlaL_SL_L_X0L_Lc_X0Lc(delta_hat, X0):
    deltaL = []
    SL = []
    L = []
    Lc = []

    for i, val in enumerate(delta_hat):
        if val == 0.0:
            Lc.append(i)

        else:
            L.append(i)
            deltaL.append(val)
            SL.append(np.sign(val))

    X0L = X0[:, L] if L else np.zeros((X0.shape[0], 0))
    X0Lc = X0[:, Lc] if Lc else np.zeros((X0.shape[0], 0))

    deltaL = np.array(deltaL).reshape(-1,1)
    SL = np.array(SL).reshape(-1,1)

    return deltaL, SL, L, X0L, Lc, X0Lc


def construct_betaM_M_SM_Mc(beta_hat):
    M = []
    betaM = []
    SM = []
    Mc = []

    for i, val in enumerate(beta_hat):
        if val != 0.0:
            M.append(i)
            SM.append(np.sign(val))
            betaM.append(val)
        else:
            Mc.append(i)

    SM = np.array(SM).reshape(-1,1)

    return betaM, M, SM, Mc


def construct_wO_SO_O_XIO_Oc_XIOc(w_hat, XI):
    wO = []
    SO = []
    O = []
    Oc = []

    for i, val in enumerate(w_hat):
        if val == 0.0:
            Oc.append(i)
        else:
            O.append(i)
            wO.append(val)
            SO.append(np.sign(val))


    XIO = XI[:, O] if O else np.zeros((XI.shape[0], 0))
    XIOc = XI[:, Oc] if Oc else np.zeros((XI.shape[0], 0))

    wO = np.array(wO).reshape(-1,1)
    SO = np.array(SO).reshape(-1,1)

    return wO, SO, O, XIO, Oc, XIOc

#_____________________________________________________________________________
# CHECK KKT
def check_KKT_theta(XO, XOc, Y, O, Oc, thetaO, SO, lambda_0, a_tilde, N):
    if len(O) > 0:
        e1 = XO @ thetaO.ravel() - Y
        active_con = ((1/N) * XO.T @ e1) + (lambda_0 * a_tilde[O] * SO).ravel()
        print ('Check Active (theta): ', np.all(np.isclose(active_con, 0)))
    if len(Oc) > 0:
        if len(O) == 0:
            SOc = ((1/N) * XOc.T @ Y) / (lambda_0 * a_tilde[Oc].ravel())
        else:
            SOc = ((-1/N) * XOc.T @ e1) / (lambda_0 * a_tilde[Oc].ravel())
        print ('Check In Active (theta): ', np.all(np.abs(SOc) < 1))


def check_KKT_w(XIO, XIOc, YI, O, Oc, wO, SO, lambda_w, nI):
    if len(O) > 0:
        e1 = XIO @ wO.ravel() - YI
        active_con = ((1/nI) * XIO.T @ e1) + (lambda_w * SO.ravel())
        print ('Check Active (w): ', np.all(np.isclose(active_con, 0)))
    
    if len(Oc) > 0:
        if len(O) == 0:
            SOc = ((1/nI) * XIOc.T @ YI) / lambda_w
        else:
            SOc = ((-1/nI) * XIOc.T @ e1) / lambda_w       
        print ('Check In Active (w): ', np.all(np.abs(SOc) < 1))


def check_KKT_delta(X0L, X0Lc, Y, L, Lc, deltaL, SL, phi_u, iota_u, lambda_tilde, nT):
    y = phi_u @ Y + iota_u.ravel()
    if len(L) > 0:
        e1 = X0L @ deltaL.ravel() - y
        active_con = ((1/nT) * X0L.T @ e1) + (lambda_tilde * SL.ravel())
        print ('Check Active (delta): ', np.all(np.isclose(active_con, 0)))

    
    if len(Lc) > 0:
        if len(L) == 0:
            SLc = ((1/nT) * X0Lc.T @ y) / lambda_tilde
        else:
            SLc = ((-1/nT) * X0Lc.T @ e1) / lambda_tilde       
        print ('Check In Active (delta): ', np.all(np.abs(SLc) < 1))

#_____________________________________________________________________________
def construct_test_statistic(j, X0M, Y, M, nT, N):
    ej = np.zeros(len(M))

    for i, ac in enumerate(M):
        if ac == j:
            ej[i] = 1
            break

    inv = np.linalg.pinv(X0M.T@X0M)
    X0M_inv = X0M @ inv

    _X = np.zeros((N, len(M)))
    _X[N - nT :, :] = X0M_inv
    etaj = _X @ ej
    etajTY = float(etaj @ Y)

    return etaj.reshape(-1, 1), etajTY


def calculate_a_b(etaj, Y, Sigma, N):
    e1 = etaj.T @ Sigma @ etaj
    b = (Sigma @ etaj)/e1

    e2 = np.eye(N) - b @ etaj.T
    a = e2 @ Y

    return a.reshape(-1, 1), b.reshape(-1, 1)


def merge_intervals(intervals, tol=1e-2):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] - merged[-1][1] > tol:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged


def pivot(intervals, etajTy, etaj, Sigma, tn_mu=0):
    if len(intervals) == 0: return None #
    intervals = merge_intervals(intervals, tol=1e-2)

    etaj = etaj.ravel()
    stdev = np.sqrt(etaj @ (Sigma @ etaj))

    numerator = mp.mpf('0')
    denominator = mp.mpf('0')

    for (left, right) in intervals:
        cdf_left= mp.ncdf((left- tn_mu)/ stdev)
        cdf_right= mp.ncdf((right- tn_mu)/ stdev)
        piece = cdf_right- cdf_left
        denominator += piece

        if etajTy >= right:
            numerator += piece
        elif left <= etajTy < right:
            numerator += mp.ncdf((etajTy - tn_mu)/ stdev) - cdf_left

    if denominator == 0:
        return None
    return float(numerator/ denominator)


def calculate_TN_p_value(intervals, etaj, etajTy, Sigma, tn_mu=0.0):
    cdf = pivot(intervals, etajTy, etaj, Sigma, tn_mu)
    return 2.0 * min(cdf, 1.0 - cdf)
