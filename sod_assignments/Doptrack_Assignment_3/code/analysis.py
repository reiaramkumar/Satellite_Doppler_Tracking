import numpy as np
from tudatpy.astro import frame_conversion


def compute_histories(parameter_history, truth_parameters, formal_errors):
    true_errors_history = np.zeros(parameter_history.shape)
    true_to_formal_errors_history = np.zeros(parameter_history.shape)

    for i in range(parameter_history.shape[1]):
        true_errors_history[:, i] = np.abs(parameter_history[:, i] - truth_parameters)
        true_to_formal_errors_history[:, i] = np.divide(true_errors_history[:, i], formal_errors,
                                                        out = np.full_like(true_errors_history[:,i], np.nan), where = formal_errors != 0)
    return true_errors_history, true_to_formal_errors_history




def compute_summary_metrics(final_residuals, true_errors, formal_errors, covariance):
    rms_final_residual = np.sqrt(np.mean(final_residuals ** 2))
    median_abs_final_residual = np.median(np.abs(final_residuals))
    max_abs_final_residual = np.max(np.abs(final_residuals))
    ratio = np.divide(np.abs(true_errors), formal_errors, out = np.full_like(true_errors, np.nan), where = formal_errors != 0)
    mean_true_to_formal_ratio = np.nanmean(ratio)

    # where np.nanmean -->

    try:
        covariance_condition_number = np.linalg.cond(covariance)
        # .cond -->
    except Exception:
        covariance_condition_number = np.nan

    return {
        "rms_final_residual":           rms_final_residual,
        "median_abs_final_residual":    median_abs_final_residual,
        "max_abs_final_residual":       max_abs_final_residual,
        "mean_true_to_formal_ratio":    mean_true_to_formal_ratio,
        "covariance_condition_number":  covariance_condition_number,
    }





def compute_rsw_covariance(covariance, arc_wise_initial_states, nb_parameters, nb_arcs):
    rotation_matrix_correlations = np.identity(nb_parameters)

    for i in range(nb_arcs):
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(arc_wise_initial_states[i])
        rotation_matrix_correlations[i * 6 + 0:i * 6 + 3, i * 6 + 0:i * 6 + 3] = rotation_to_rsw
        rotation_matrix_correlations[i * 6 + 3:i * 6 + 6, i * 6 + 3:i * 6 + 6] = rotation_to_rsw

    rsw_covariance = rotation_matrix_correlations @ covariance @ rotation_matrix_correlations.T
    rsw_formal_errors = np.sqrt(np.diagonal(rsw_covariance))

    rsw_correlations = rsw_covariance.copy()
    for i in range(nb_parameters):
        for j in range(nb_parameters):
            denom = rsw_formal_errors[i] * rsw_formal_errors[j]

            if denom !=0:
                rsw_correlations[i, j] = rsw_covariance[i, j] / denom
            else:
                rsw_correlations[i, j] = np.nan

    return rsw_covariance, rsw_formal_errors, rsw_correlations

