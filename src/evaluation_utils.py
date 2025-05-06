import numpy as np
from scipy.stats import t

def correlated_t_test(r_vals, K_fold=10, alpha=0.05):

    # r_vals is differences in performance metric per outer fold
    r = np.asarray(r_vals)
    J = len(r)
    nu = J - 1

    r_mean = r.mean()
    sigma = r.std(ddof=1)

    # correlation estimate
    rho = 1.0 / K_fold

    # t statistic
    denom = sigma * np.sqrt(1.0/J + rho/(1.0 - rho))
    t_stat = r_mean / denom

    # two-sided p-value
    p_val = 2 * t.cdf(-abs(t_stat), df=nu)

    # confidence interval
    z_lower = t.ppf(alpha/2, df=nu)
    z_upper = t.ppf(1-alpha/2, df=nu)
    ci_low  = r_mean + z_lower * denom
    ci_high = r_mean + z_upper * denom

    return {
        't_stat': t_stat,
        'p_value': p_val,
        'mean_diff': r_mean,
        'ci': (ci_low, ci_high),
        'df': nu
    }
