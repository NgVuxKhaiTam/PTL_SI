from skglm import WeightedLasso, Lasso
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def TransFusion(X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights):
    # Co-Training
    w_pen = np.ones((K+1) * p)
    idx = 0
    for k in range(1, K+1):
        w_pen[idx : idx+p] *= ak_weights[k-1]
        idx += p

    co_training = WeightedLasso(alpha=lambda_0, fit_intercept=False, tol=1e-10, weights=w_pen)
    co_training.fit(X, Y)
    theta_hat = co_training.coef_

    w_hat = (1.0 / N) * (B @ theta_hat)
    
    # Debias
    debias = Lasso(alpha=lambda_tilde, fit_intercept=False, tol=1e-12)
    debias.fit(X0, Y0 - X0@w_hat)
    delta_hat = debias.coef_

    beta0_hat = w_hat + delta_hat

    return theta_hat, w_hat, delta_hat, beta0_hat


def OracleTransLasso(XI, YI, X0, Y0, lambda_w, lambda_delta):
    w_hat = Lasso(alpha=lambda_w, fit_intercept=False, tol=1e-10).fit(XI, YI).coef_

    delta_hat = Lasso(alpha=lambda_delta, fit_intercept=False, tol=1e-12).fit(X0, Y0 - X0@w_hat).coef_

    beta0_hat = w_hat + delta_hat

    return w_hat, delta_hat, beta0_hat