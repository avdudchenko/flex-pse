"""Placeholder — Pump-hydraulic recovery tests (M11b, not yet implemented).

The motivating case: `Pump`'s hydraulic power law
(`power == delta_pressure * flow / efficiency`) is nonlinear in `efficiency`,
so no existing `flexparameterize.regression` regressor can fit it.

Planned tests (`component` tier, `needs_ipopt`) — see
``plan/milestones/M11b_native_nlp_estimation.md`` for the full spec:

- test_recovers_efficiency_from_noise_free_hydraulic_data
- test_recovers_efficiency_with_seeded_noise_and_finite_covariance
- test_cov_method_reduced_hessian_vs_finite_difference_agree
- test_efficiency_out_of_bounds_data_reports_low_confidence_not_a_crash
"""
