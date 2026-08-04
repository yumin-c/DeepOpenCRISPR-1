Feature-vs-DeepOC vs reproducibility-ceiling analysis — output files
====================================================================

model_performance.csv
    One row per (subset, feature_set, regressor).
    r2                out-of-fold coefficient of determination on Fold0..4
    ci_lo, ci_hi      95% bootstrap CI (B=2000, percentile)
    is_primary        1 = the regressor used as the headline for that feature set
                      (GBM for feature models; 'raw' for DeepOC = its own OOF
                      prediction, no refit)
    feature_set values (handcrafted features = the ones characterized in the
    manuscript: GC content, positional nucleotide composition, and PAM):
        GC                raw GC content of spacer and target
        positional        position x nucleotide one-hot of the protospacer
        PAM               PAM 4-nt position x nucleotide one-hot (16 features)
        features          GC + positional + PAM (all handcrafted features)
        DeepOC            final sequence-only DeepOC model, out-of-fold prediction
        DeepOC+features   DeepOC prediction plus the handcrafted features
    regressor values: RF (random forest), GBM (hist. gradient boosting),
        lin (ridge), raw (no model; DeepOC OOF prediction used directly)

reproducibility_ceiling.csv
    Replicate-based upper bound on achievable R^2, per subset. The target
    (`activity`) is a single background-subtracted value rather than the mean of
    the two replicates, so the ceiling band is [rho^2, rho]:
    rho                        Pearson r between Replicate 1 and Replicate 2;
                               reliability of a single measurement
    r2_ceiling                 = rho (upper bound; max R^2 for a single-
                               measurement target under classical test theory)
    r2_ceiling_conservative    rho^2  (R^2 of one experiment predicting an
                               independent repeat; conservative lower bound)
    *_lo/_hi                   95% bootstrap CIs

paired_deltas.csv
    Paired bootstrap (same resample indices within a subset) for:
    DeepOC_minus_features   DeepOC (raw) minus features=GC+positional+PAM (GBM)
    combined_minus_DeepOC   DeepOC+features (GBM) minus DeepOC (raw)
    ceiling_minus_DeepOC    conservative ceiling (rho^2) minus DeepOC (raw)
    delta_r2, ci_lo, ci_hi  point estimate and 95% CI of the difference
    frac_positive           fraction of bootstrap resamples with delta > 0

bootstrap_draws.csv
    Long-format raw bootstrap R^2 draws for the primary model of each feature
    set (columns: subset, feature_set, regressor, draw, r2). Use for violin /
    custom interval plots.

oof_predictions_<subset>.csv
    Per-row out-of-fold predictions for scatter/agreement plots:
    activity (measured), pred_deepoc, pred_features_gbm (GC+positional+PAM),
    pred_combined_gbm, plus Replicate 1/2 and the Info label.

Subsets: matched_NGG (on-target, single optimal PAM class; primary),
    all_matched (on-target, all PAMs), full (adds mismatched/off-target rows).
Rows are restricted to the Fold0..4 cross-validation partition.
