"""
VoltIQ -- app/models/leakage_analysis.py
==========================================
Phase 2: Feature Leakage Analysis

Identifies and documents features that should be excluded from ML training
because they are derived from, directly encode, or are extremely highly
correlated with the training target variable.

Leakage classification
-----------------------
CONFIRMED_LEAKAGE
    Features that are mathematically derived from the target or that would
    not be available at inference time (e.g. Health_Zone is derived from
    State_of_Health -- you would not know the zone without knowing the SOH).

HIGH_CORRELATION_RISK (threshold: |r| >= 0.98)
    Features whose Pearson correlation with the target exceeds 0.98.
    These may not be definitively leaking, but the risk is high enough to
    exclude them and document the decision.

MODERATE_CORRELATION (threshold: 0.90 <= |r| < 0.98)
    Flagged for review but retained unless also in CONFIRMED_LEAKAGE.

SAFE
    Features that pass all checks and are admitted to the final feature set.

Usage
-----
    from app.models.leakage_analysis import LeakageAnalyzer

    analyzer = LeakageAnalyzer()
    result   = analyzer.analyze(df, target="State_of_Health",
                                 candidates=ALL_BATTERY_FEATURES)
    clean_features = result.safe_features
    report_md      = result.to_markdown()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known derived / post-hoc columns per model target
# ---------------------------------------------------------------------------
# These are confirmed leakage columns regardless of correlation score.
# They are derived from the target after the fact and would not be
# available in a real production inference scenario.

#: Battery SOH -- State_of_Health
_BATTERY_SOH_CONFIRMED_LEAKAGE: Set[str] = {
    # Phase 1 FeatureEngineer columns derived directly from SOH
    "SOH_Category",          # binned from State_of_Health
    "SOH_Flag",              # derived flag
    # Categorical encodings also derived from SOH
    "Health_Zone",           # categorical label derived from SOH
    # End-of-life flag derived from SOH reaching threshold
    "Is_End_of_Life",
    # Capacity_Fade_Pct = (1 - SOH) * 100 in many formulations
    # Confirmed by correlation > 0.99 on this dataset
    "Capacity_Fade_Pct",
    # Target itself
    "State_of_Health",
}

#: Battery RUL -- Remaining_Useful_Life_Cycles
_BATTERY_RUL_CONFIRMED_LEAKAGE: Set[str] = {
    # End-of-life flag is derived from RUL reaching 0
    "Is_End_of_Life",
    # Health zone is derived from SOH which correlates almost 1:1 with RUL
    "Health_Zone",
    # SOH category (binned from SOH) -- SOH is the strongest RUL predictor
    "SOH_Category",
    # Target itself
    "Remaining_Useful_Life_Cycles",
    # State of Health is a near-perfect proxy for RUL in this dataset
    # (RUL = remaining_to_EOL - cycle, SOH = f(cycle))
    # Keep SOH as a legitimate operational measurement -- a technician
    # can measure SOH with a tester without knowing RUL.
    # Therefore SOH is NOT excluded from RUL model.
}

#: Fleet Readiness -- EV_Readiness_Score
_FLEET_CONFIRMED_LEAKAGE: Set[str] = {
    # Phase 1 FeatureEngineer columns derived from EV_Readiness_Score
    "EV_Priority_Score",       # composite score = EV_Readiness_Score * saving factor
    "Readiness_Label",         # categorical label binned from EV_Readiness_Score
    "Readiness_Category",      # alternative label column
    # Target itself
    "EV_Readiness_Score",
}

CONFIRMED_LEAKAGE_MAP: Dict[str, Set[str]] = {
    "State_of_Health":              _BATTERY_SOH_CONFIRMED_LEAKAGE,
    "Remaining_Useful_Life_Cycles": _BATTERY_RUL_CONFIRMED_LEAKAGE,
    "EV_Readiness_Score":           _FLEET_CONFIRMED_LEAKAGE,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeakageReport:
    """Complete leakage analysis result for one model target."""

    target: str
    all_candidates: List[str]
    safe_features: List[str]
    confirmed_leakage: List[str]
    high_correlation_risk: List[str]        # |r| >= high_thresh
    moderate_correlation: List[str]         # moderate_thresh <= |r| < high_thresh
    correlations: Dict[str, float]          # feature -> Pearson r with target
    high_corr_threshold: float
    moderate_corr_threshold: float

    # Computed counts
    n_candidates: int = field(init=False)
    n_excluded: int   = field(init=False)
    n_safe: int       = field(init=False)

    def __post_init__(self):
        self.n_candidates = len(self.all_candidates)
        self.n_excluded   = len(self.confirmed_leakage) + len(self.high_correlation_risk)
        self.n_safe       = len(self.safe_features)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown leakage report."""
        lines = [
            f"## Feature Leakage Analysis -- `{self.target}`",
            "",
            "### Summary",
            "",
            f"| Item | Count |",
            f"|---|---|",
            f"| Candidate features | {self.n_candidates} |",
            f"| Confirmed leakage (excluded) | {len(self.confirmed_leakage)} |",
            f"| High correlation risk (excluded, |r|>={self.high_corr_threshold}) "
            f"| {len(self.high_correlation_risk)} |",
            f"| Moderate correlation (retained, flagged) | {len(self.moderate_correlation)} |",
            f"| Safe features admitted to model | **{self.n_safe}** |",
            "",
        ]

        if self.confirmed_leakage:
            lines += [
                "### Confirmed Leakage -- EXCLUDED",
                "",
                "These features are derived from, encode, or would not be available "
                "at inference time without knowing the target.",
                "",
                "| Feature | Pearson r | Reason |",
                "|---|---|---|",
            ]
            for f in sorted(self.confirmed_leakage):
                r = self.correlations.get(f, float("nan"))
                r_str = f"{r:.4f}" if not np.isnan(r) else "N/A (categorical)"
                lines.append(f"| `{f}` | {r_str} | Confirmed derived/post-hoc feature |")
            lines.append("")

        if self.high_correlation_risk:
            lines += [
                f"### High Correlation Risk -- EXCLUDED (|r| >= {self.high_corr_threshold})",
                "",
                "| Feature | Pearson r |",
                "|---|---|",
            ]
            for f in sorted(self.high_correlation_risk,
                             key=lambda x: abs(self.correlations.get(x, 0)), reverse=True):
                r = self.correlations.get(f, float("nan"))
                lines.append(f"| `{f}` | {r:.4f} |")
            lines.append("")

        if self.moderate_correlation:
            lines += [
                f"### Moderate Correlation -- RETAINED, FLAGGED "
                f"({self.moderate_corr_threshold} <= |r| < {self.high_corr_threshold})",
                "",
                "| Feature | Pearson r |",
                "|---|---|",
            ]
            for f in sorted(self.moderate_correlation,
                             key=lambda x: abs(self.correlations.get(x, 0)), reverse=True):
                r = self.correlations.get(f, float("nan"))
                lines.append(f"| `{f}` | {r:.4f} |")
            lines.append("")

        lines += [
            "### Safe Features -- ADMITTED TO MODEL",
            "",
            "| Feature | Pearson r |",
            "|---|---|",
        ]
        for f in self.safe_features:
            r = self.correlations.get(f, float("nan"))
            r_str = f"{r:.4f}" if not np.isnan(r) else "N/A"
            lines.append(f"| `{f}` | {r_str} |")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialisable dict for metadata JSON."""
        return {
            "target":                   self.target,
            "n_candidates":             self.n_candidates,
            "n_safe":                   self.n_safe,
            "n_excluded_total":         self.n_excluded,
            "safe_features":            self.safe_features,
            "confirmed_leakage":        self.confirmed_leakage,
            "high_correlation_risk":    self.high_correlation_risk,
            "moderate_correlation":     self.moderate_correlation,
            "correlations":             {k: round(v, 6) for k, v in self.correlations.items()
                                         if not np.isnan(v)},
            "high_corr_threshold":      self.high_corr_threshold,
            "moderate_corr_threshold":  self.moderate_corr_threshold,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class LeakageAnalyzer:
    """
    Identifies data leakage in ML feature candidates.

    Parameters
    ----------
    high_corr_threshold : float
        Pearson |r| threshold above which a feature is excluded as
        high-correlation risk. Default: 0.98.
    moderate_corr_threshold : float
        |r| threshold above which a feature is flagged (but retained).
        Default: 0.90.
    """

    def __init__(
        self,
        high_corr_threshold:     float = 0.98,
        moderate_corr_threshold: float = 0.90,
    ) -> None:
        self.high_corr_threshold     = high_corr_threshold
        self.moderate_corr_threshold = moderate_corr_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        candidates: List[str],
    ) -> LeakageReport:
        """
        Run leakage analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset (pre-split).
        target : str
            Name of the target column.
        candidates : list[str]
            Candidate feature column names to evaluate.

        Returns
        -------
        LeakageReport
            Complete leakage analysis result.
        """
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")

        logger.info("[LeakageAnalyzer] Analyzing %d candidates for target '%s'",
                    len(candidates), target)

        # --- Get confirmed leakage set for this target
        confirmed_leakage_set = CONFIRMED_LEAKAGE_MAP.get(target, set())

        # --- Compute Pearson correlations for numeric candidates
        y = df[target]
        correlations: Dict[str, float] = {}

        for feat in candidates:
            if feat not in df.columns:
                continue
            if feat == target:
                correlations[feat] = 1.0
                continue
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                try:
                    r = float(col.corr(y))
                    correlations[feat] = r if not np.isnan(r) else 0.0
                except Exception:
                    correlations[feat] = 0.0
            else:
                correlations[feat] = float("nan")   # categorical -- can't compute Pearson

        # --- Categorise features
        confirmed_leakage:    List[str] = []
        high_correlation:     List[str] = []
        moderate_correlation: List[str] = []
        safe_features:        List[str] = []

        for feat in candidates:
            if feat not in df.columns:
                logger.debug("Candidate '%s' absent from DataFrame -- skipped.", feat)
                continue

            # Confirmed leakage always excluded
            if feat in confirmed_leakage_set:
                confirmed_leakage.append(feat)
                logger.info("  [CONFIRMED LEAKAGE] %s", feat)
                continue

            r     = correlations.get(feat, float("nan"))
            abs_r = abs(r) if not np.isnan(r) else 0.0

            if abs_r >= self.high_corr_threshold:
                high_correlation.append(feat)
                logger.warning(
                    "  [HIGH CORRELATION RISK] %s  r=%.4f  (threshold=%.2f) -- EXCLUDED",
                    feat, r, self.high_corr_threshold,
                )
            elif abs_r >= self.moderate_corr_threshold:
                moderate_correlation.append(feat)
                logger.info(
                    "  [MODERATE CORRELATION] %s  r=%.4f  (flagged, retained)",
                    feat, r,
                )
                safe_features.append(feat)
            else:
                safe_features.append(feat)
                logger.debug("  [SAFE] %s  r=%.4f", feat, r if not np.isnan(r) else 0.0)

        logger.info(
            "[LeakageAnalyzer] Result: %d safe | %d confirmed leakage | "
            "%d high-corr risk | %d moderate",
            len(safe_features), len(confirmed_leakage),
            len(high_correlation), len(moderate_correlation),
        )

        return LeakageReport(
            target=target,
            all_candidates=[f for f in candidates if f in df.columns],
            safe_features=safe_features,
            confirmed_leakage=confirmed_leakage,
            high_correlation_risk=high_correlation,
            moderate_correlation=moderate_correlation,
            correlations=correlations,
            high_corr_threshold=self.high_corr_threshold,
            moderate_corr_threshold=self.moderate_corr_threshold,
        )
