# Truncation Certificate for Weil Positivity (Unconditional)

What we certify. For each test window h we compute

Q_T(h) = Z_{≤T}(h) − A_{≤t_max}(h) − P_{≤N}(h)

and an unconditional tail upper bound

Tail(h) = Z_{>T}(h) + A_{|t|>t_max}(h) + P_{>N}(h),

so that the lower bound

Q_lower(h) = Q_T(h) − Tail(h)

is provably positive. This yields a RH-consistent Weil positivity certificate without any circular use of average zero density.

Unconditional bounds used
- Z-tail: dN/dt ≤ c0 log t + c1 for t ≥ T0 ⇒ ∫_T^∞ |h(t)|(c0 log t + c1) dt.
- P-tail: ψ(x) ≤ Cψ x for x ≥ x0 ⇒ (Cψ/π) ∫_{log N}^∞ |ĥ(y)| e^{y/2} dy.
- A-tail: |Re ψ(1/4 + i t/2) − log π| ≤ log|t| + CA for |t|≥ x0 ⇒ (1/2π) ∫_{|t|>t_max} |h(t)| (log|t| + CA) dt.

All constants live in experiments/bounds_config.py. Replace placeholders with explicit literature-backed values for a final unconditional statement.

How to run
- Gaussian family on a grid (100k zeros):
  - python3 -m experiments.weil_truncated --zeros=100000 --family gaussian --grid "0.6:1.4:0.02" --N-max 100000 --t-max 50 --plot
- Adversarial search (find hardest σ):
  - python3 -m experiments.inverse_h_search.py --zeros 100000 --family gaussian --sigma-grid "0.6:1.4:0.01" --heatmap

Interpretation
If Q_lower > 0 across a covering grid and the Lipschitz-based expansion closes the intervals with δ − LΔ > 0, we obtain a uniform positivity certificate on the target parameter range.

Note. This approach directly addresses the Positive Wall barrier via Weil’s explicit formula on a rich class of test functions (not a single window; no Gram/PSD constructs). We explicitly avoid “unified field” narratives and symbolic limit tricks; they are orthogonal to a math-proof-grade certificate.

