# Mathematical formulation

## 1. Sparse engram

For episode \(e\), model activations at layer/site \((\ell,s)\) are
\(H_e\in\mathbb R^{n_e\times d}\). A feature dictionary \(F\) produces
\(U_e=F(H_e)\in\mathbb R^{n_e\times m}\). The episode address is

\[
z_e=\frac{\operatorname{TopK}_k(P(U_e))}{
\|\operatorname{TopK}_k(P(U_e))\|_2+\epsilon},
\qquad \|z_e\|_0\le k\ll m.
\]

Pooling \(P\) is max, positive-max, mean, sum, last-token, boundary mean, or learned attention.
The experiment reports both \(m\) and \(k\); “sparse” without those values is underspecified.

## 2. Plasticity rules

Let \(g_e\) be the modulatory salience signal.

### Hebb

\[
\Delta A_{ij}=\eta g_e z_{e,i}z_{e,j}.
\]

This is simple but over-strengthens frequent features.

### Centered covariance

\[
\Delta A_{ij}=\eta g_e(z_{e,i}-\mu_i)(z_{e,j}-\mu_j),
\]

where \(\mu_i\) is an exponential activity estimate. A complete covariance update would include
inactive-feature terms and is dense; the sparse implementation updates active pairs only. That
approximation must be named in papers and ablated against raw Hebb.

### Oja-like normalization

\[
\Delta A_{ij}=\eta g_e\left[z_i z_j-\tfrac12(z_i^2+z_j^2)A_{ij}\right].
\]

### BCM-like threshold

\[
\Delta A_{ij}\approx \frac{\eta g_e}{2}
[z_i z_j(z_i-\vartheta_i)+z_jz_i(z_j-\vartheta_j)].
\]

The implementation treats these as computational analogues, not detailed biological models.

### Decay/LTD and homeostasis

\[
A\leftarrow(1-\rho)A,
\]

followed by row top-degree pruning and norm caps. Contradiction-specific LTD is a planned GPU
experiment:

\[
A_{ij}\leftarrow A_{ij}-\eta_-g^- z_i z_j.
\]

## 3. Temporal links

For adjacent episodes:

\[
\Delta T=\eta_Tg_e z_{e-1}z_e^\top.
\]

Forward-only, backward-only, and bidirectional readout are separate conditions. Temporal edges are
not folded into \(A\), because order and semantic coactivation answer different questions.

## 4. Attractor-style recall

Define normalized association

\[
\widetilde A=D^{-1/2}AD^{-1/2}.
\]

A query-anchored update is

\[
u^{(r+1)}=\alpha q+\beta\widetilde A a^{(r)}+
\gamma_fTa^{(r)}+\gamma_bT^\top a^{(r)}-\theta,
\]

\[
a^{(r+1)}=\operatorname{Normalize}(\operatorname{TopK}_K(\phi(u^{(r+1)}))).
\]

For a symmetric nonnegative \(A\), this resembles sparse descent on

\[
E(a)=-\tfrac12a^\top\widetilde Aa-q^\top a+\lambda\|a\|_1,
\]

but top-k pruning, temporal asymmetry, and query injection mean the implementation is not guaranteed
to minimize a single global energy. Convergence, cycles, and drift are measured rather than assumed.

## 5. Episode score

\[
S_e=\lambda_xz_e^\top q+
\lambda_az_e^\top a^{(R)}+
\lambda_tz_e^\top a_T+
\lambda_r2^{-\Delta t/h}+
\lambda_cc_e.
\]

Primary confirmatory comparisons keep candidate count and read compute fixed. Allowing AAM more
candidates would confound association quality with brute-force recall.

## 6. Salience

\[
g_e=\operatorname{clip}_{[g_{min},g_{max}]}
(b+w_sS_e+w_tT_e+w_uU_e+w_nN_e-w_dD_e).
\]

Surprise may be token loss under the writer, task relevance may come from supervision, user
importance must be explicit, novelty is one minus maximum prior similarity, and redundancy is a
near-duplicate estimate.

## 7. Payload objective

A full-context teacher and memory-conditioned student define

\[
\mathcal L_{KD}=D_{KL}
(p_\theta(\cdot|X_{full})\|p_{\theta,\phi}(\cdot|X_{recent},M)).
\]

A complete objective is

\[
\mathcal L=\mathcal L_{task}
+\lambda_{KD}\mathcal L_{KD}
+\lambda_{src}\mathcal L_{source}
+\lambda_{sep}\mathcal L_{separation}
+\lambda_{gate}\mathcal L_{irrelevant\ gate}
+\lambda_{rec}\|D(C(H))-H\|^2.
\]

The source loss predicts the supporting episode ID. It prevents a system from receiving full credit
for a plausible answer unsupported by retrieved evidence.

## 8. Information bound

A memory code containing \(B\) effective bits cannot losslessly recover an arbitrary independent
string with entropy much greater than \(B\). The model may reconstruct predictable semantic content
from priors, but random identifiers, exact quotations, numbers, and cryptographic strings require
source or payload capacity. Random-string diagnostics therefore separate semantic recall from hidden
copying.

## 9. Capacity accounting

For \(N\) episodes with sparse-code bytes \(B_z\), payload bytes \(B_c\), source bytes \(B_p\),
metadata bytes \(B_m\), posting-index bytes \(B_I\), graph node-state bytes \(B_N\), and edge bytes
\(B_E\):

\[
B_{total}=\sum_{e=1}^{N}(B_{z,e}+B_{c,e}+B_{p,e}+B_{m,e})+B_I+B_N+B_E.
\]

The reference logical accountant uses explicit scalar widths and UTF-8/source metadata sizes, and
reports physical SQLite/WAL/SHM size separately. A fixed-byte controller evicts episodes using the
same episode-size estimator, then rebuilds the graph and rechecks total logical bytes. Every
quality-versus-memory chart must use total bytes or clearly show each component and any excluded
runtime allocator overhead.

## 10. Compute proxies

The reference runner logs association updates, temporal updates, and message-passing edge visits.
These are deterministic algorithmic proxies, not hardware FLOPs. Accelerator experiments must add
synchronized writer forward time, SAE time, payload decode time, reader prefill/decode time, peak
memory, and architecture-specific FLOP estimates. A recurrence condition is compute matched either
by fixing edge visits/candidate count or by reporting a quality-versus-read-compute curve.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
