import numpy as np
from scipy.sparse import csr_array, diags



def checks_likelihoods(
        exc_matrix: csr_array
        ) -> csr_array: 
    # === 1. Check negative values ===
    tol_neg = 1e-6  # tolerance for the negligible negative values 
    negative_mask = exc_matrix.data < - tol_neg
    if np.any(negative_mask):
        raise ValueError(f"Significant negative values found in the matrix: {exc_matrix.data[negative_mask]}")
    else:
        small_neg_mask = (exc_matrix.data < 0) & (exc_matrix.data >= -tol_neg)
        exc_matrix.data[small_neg_mask] = 0.0

    # === 2. Check sum rows ≈ 1 ===
    row_sums = np.array(exc_matrix.sum(axis=0)).flatten()
    tol_row = 1e-3  # tolleranza per la somma delle righe
    bad_rows = np.abs(row_sums - 1.0) > tol_row

    if np.any(bad_rows):
        raise ValueError(f"The sum of some rows is signicantly different from 1: {row_sums[bad_rows]}")
    else:
        # Renormalize if necessary (within tolerance) 
        needs_norm = (np.abs(row_sums - 1.0) > 1e-12)
        if np.any(needs_norm):
            correction_factors = np.ones_like(row_sums)
            correction_factors[needs_norm] = 1.0 / row_sums[needs_norm]
            
            row_scaling = diags(correction_factors)
            exc_matrix = row_scaling @ exc_matrix  # normalize the rows

    return exc_matrix



def cleaning_convergence(max_step, curves_by_label, misfrequency_by_label, variance_by_label):
    threshold = max_step - int(0.03 * max_step)

    curves_by_label_filtered = {}
    misfrequency_by_label_filtered = {}
    variance_by_label_filtered = {}

    curves_by_label_not_converged = {}
    misfrequency_by_label_not_converged = {}
    variance_by_label_not_converged = {}

    total_runs = 0
    converged_runs = 0

    for label in curves_by_label:
        curves_by_label_filtered[label] = []
        misfrequency_by_label_filtered[label] = []
        variance_by_label_filtered[label] = []

        curves_by_label_not_converged[label] = []
        misfrequency_by_label_not_converged[label] = []
        variance_by_label_not_converged[label] = []

        for i, (steps, ce_curve) in enumerate(curves_by_label[label]):
            last_step = steps[-1] if len(steps) > 0 else -1
            total_runs += 1
            if last_step < threshold:
                converged_runs += 1
                curves_by_label_filtered[label].append((steps, ce_curve))
                misfrequency_by_label_filtered[label].append(misfrequency_by_label[label][i])
                variance_by_label_filtered[label].append(variance_by_label[label][i])
            else:
                curves_by_label_not_converged[label].append((steps, ce_curve))
                misfrequency_by_label_not_converged[label].append(misfrequency_by_label[label][i])
                variance_by_label_not_converged[label].append(variance_by_label[label][i])

    fraction_converged = converged_runs / total_runs
    fraction_not_converged = 1.0 - fraction_converged

    return (curves_by_label_filtered, misfrequency_by_label_filtered, variance_by_label_filtered,
            curves_by_label_not_converged, misfrequency_by_label_not_converged, variance_by_label_not_converged,
            fraction_converged, fraction_not_converged)
