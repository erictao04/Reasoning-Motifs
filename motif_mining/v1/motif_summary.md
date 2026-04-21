# Motif Subsequences Summary

- Input file: `C:\Users\erict\Programming\School\CPSC 440\Reasoning-Motifs\motif_mining\v1\all_motif_subsequences.csv`
- Total motifs: **2190**
- Motifs with support >= 10: **204**

## Label Split

- correct-dominant=1587, incorrect-dominant=428, tie=175

## Distribution of p_success (p_correct)

- min=0.000, p25=0.500, median=1.000, mean=0.724, p75=1.000, max=1.000

| Bin        | Count |
| ---------- | ----: |
| [0.0, 0.1) |   330 |
| [0.1, 0.2) |     9 |
| [0.2, 0.3) |    13 |
| [0.3, 0.4) |    59 |
| [0.4, 0.5) |    17 |
| [0.5, 0.6) |   202 |
| [0.6, 0.7) |   146 |
| [0.7, 0.8) |   134 |
| [0.8, 0.9) |   124 |
| [0.9, 1.0] |  1156 |

## Best motifs (highest p_success)

- **compare -> analyze -> compute** (support=12, p_success=1.000, p_fail=0.000, enrich_success=13.000, enrich_fail=0.077)
- **compute -> count -> case-split** (support=10, p_success=1.000, p_fail=0.000, enrich_success=11.000, enrich_fail=0.091)
- **check-constraint -> compute -> analyze** (support=20, p_success=0.950, p_fail=0.050, enrich_success=10.000, enrich_fail=0.100)
- **substitute -> compute -> analyze** (support=30, p_success=0.933, p_fail=0.067, enrich_success=9.667, enrich_fail=0.103)
- **compute -> substitute -> compute -> analyze** (support=14, p_success=0.929, p_fail=0.071, enrich_success=7.000, enrich_fail=0.143)
- **count -> case-split** (support=26, p_success=0.923, p_fail=0.077, enrich_success=8.333, enrich_fail=0.120)
- **rewrite -> analyze -> instantiate -> compute** (support=12, p_success=0.917, p_fail=0.083, enrich_success=6.000, enrich_fail=0.167)
- **analyze -> substitute -> compute -> analyze** (support=12, p_success=0.917, p_fail=0.083, enrich_success=6.000, enrich_fail=0.167)
- **compute -> conclude -> analyze** (support=12, p_success=0.917, p_fail=0.083, enrich_success=6.000, enrich_fail=0.167)
- **analyze -> compute -> analyze -> substitute** (support=11, p_success=0.909, p_fail=0.091, enrich_success=5.500, enrich_fail=0.182)

## Worst motifs (highest p_fail)

- **compute -> substitute -> analyze** (support=12, p_success=0.167, p_fail=0.833, enrich_success=0.273, enrich_fail=3.667)
- **apply-formula -> instantiate -> rewrite** (support=10, p_success=0.200, p_fail=0.800, enrich_success=0.333, enrich_fail=3.000)
- **substitute -> analyze -> compute** (support=13, p_success=0.231, p_fail=0.769, enrich_success=0.364, enrich_fail=2.750)
- **substitute -> analyze** (support=18, p_success=0.278, p_fail=0.722, enrich_success=0.429, enrich_fail=2.333)
- **rewrite -> apply-formula** (support=13, p_success=0.308, p_fail=0.692, enrich_success=0.500, enrich_fail=2.000)
- **conclude -> compute -> analyze** (support=21, p_success=0.333, p_fail=0.667, enrich_success=0.533, enrich_fail=1.875)
- **apply-formula -> substitute** (support=14, p_success=0.357, p_fail=0.643, enrich_success=0.600, enrich_fail=1.667)
- **rewrite -> conclude** (support=14, p_success=0.357, p_fail=0.643, enrich_success=0.600, enrich_fail=1.667)
- **conclude -> compute -> instantiate** (support=11, p_success=0.364, p_fail=0.636, enrich_success=0.625, enrich_fail=1.600)
- **backtrack -> instantiate** (support=11, p_success=0.364, p_fail=0.636, enrich_success=0.625, enrich_fail=1.600)

## Most frequent motifs

- **compute -> analyze** (support=192, p_success=0.708, p_fail=0.292, enrich_success=2.404, enrich_fail=0.416)
- **analyze -> compute** (support=186, p_success=0.747, p_fail=0.253, enrich_success=2.917, enrich_fail=0.343)
- **instantiate -> compute** (support=167, p_success=0.719, p_fail=0.281, enrich_success=2.521, enrich_fail=0.397)
- **analyze -> instantiate** (support=144, p_success=0.743, p_fail=0.257, enrich_success=2.842, enrich_fail=0.352)
- **instantiate -> analyze** (support=130, p_success=0.723, p_fail=0.277, enrich_success=2.568, enrich_fail=0.389)
- **compute -> instantiate** (support=124, p_success=0.637, p_fail=0.363, enrich_success=1.739, enrich_fail=0.575)
- **compute -> analyze -> instantiate** (support=101, p_success=0.703, p_fail=0.297, enrich_success=2.323, enrich_fail=0.431)
- **check-constraint -> compute** (support=82, p_success=0.744, p_fail=0.256, enrich_success=2.818, enrich_fail=0.355)
- **analyze -> compute -> analyze** (support=79, p_success=0.772, p_fail=0.228, enrich_success=3.263, enrich_fail=0.306)
- **analyze -> instantiate -> compute** (support=79, p_success=0.709, p_fail=0.291, enrich_success=2.375, enrich_fail=0.421)
