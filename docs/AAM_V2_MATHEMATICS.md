# AAM-v2 mathematical formulation

## Sparse address

Given hidden states or feature activations `H_t`, an encoder produces sparse hidden code

\[
h_t=\operatorname{TopK}(F_\phi(H_t)), \quad \|h_t\|_0=k_h.
\]

Context encoder:

\[
r_t=\operatorname{TopK}(G_\rho(\mathrm{speaker},\mathrm{time},\mathrm{tool},\mathrm{source},\mathrm{modality},\mathrm{state\ tensor},\ldots)).
\]

Bound address:

\[
z_t=\operatorname{Norm}\operatorname{TopK}([\lambda_hh_t;\lambda_rr_t;\lambda_b(h_t\odot Rr_t);n_t]).
\]

`n_t` is empty unless neurogenesis births a reserved sidecar feature.

## Dopamine/write gate

For metadata or model-derived metrics `m_t`, AAM-v2 computes separate LTP and LTD gates:

\[
\delta_t^+=s_{max}\sigma(b+w_\ell\ell_t+w_HH_t+w_KK_t+w_NN_t-w_RR_t+w_UU_t+w_CC_t+w_EE_t-w_PP_t)
\]

\[
\delta_t^-=s_{max}\sigma(b_-+w_{contra}C_t+w_{stale}S_t+w_{false}F_t+w_PP_t-w_{trust}T_t).
\]

Implemented in `memory/dopamine.py`.  The CPU path accepts metrics from metadata; open-model GPU paths should fill loss, entropy, and KL probes from model forwards.

## Allocation

Eligibility:

\[
e_i(t+1)=\lambda_e e_i(t)+|z_{t,i}|.
\]

Anti-hub feature gain:

\[
g_i=\mathrm{clip}\left(\frac{1+\lambda_NN_t+\lambda_UU_t+\lambda_e\log(1+e_i)}{(1+d_i)^{\lambda_h}},g_{min},g_{max}\right).
\]

The written address is `Normalize(g ⊙ z_t)`.

## Coactivation/LTP/LTD

Centered Hebbian update over active features:

\[
\Delta A^+_{ij}=\eta_+\delta_t^+g_ig_j(z_{t,i}-\mu_i)(z_{t,j}-\mu_j).
\]

Negative update / decay:

\[
A\leftarrow\operatorname{RowTopD}\operatorname{NormRows}((1-\rho)A+\Delta A^+-\eta_-\delta_t^-z_tz_t^\top).
\]

The existing `SparseAssociationGraph` supports Hebb, covariance, Oja, BCM, zero, shuffled, and random-degree-matched controls.

## Temporal/STDP-like association

For ordered episodes:

\[
\Delta T_{ij}=\eta_T\delta_t^+z_{t-1,i}z_{t,j}e^{-\Delta t/\tau}.
\]

This is implemented as directed previous-to-current feature updates in `memory/associations.py`; recurrence reads both forward and backward directions when configured.

## Episode-index identity

Episode-specific feature matrix `B`:

\[
B_{ei}\leftarrow \eta_B\delta_t^+z_{t,i}.
\]

Episode activation:

\[
m_e^r=\sigma(B_e a^r + C_e r_q - \theta_m).
\]

Feature reinstatement:

\[
B^\top m^r.
\]

This preserves identity of overlapping memories whose sparse feature vectors are similar.

## Pattern completion

\[
a^{0}=q
\]

\[
a^{r+1}=\operatorname{TopK}\phi(\alpha q+\beta\tilde Aa^r+\gamma_f\tilde Ta^r+\gamma_b\tilde T^\top a^r+\xi B^\top m^r+\chi C^\top r_q-\lambda_hh-\theta).
\]

The persistent `αq` anchor is mandatory; depth-zero and no-anchor are ablations.

## Retrieval scoring

\[
S(e)=w_dz_e^\top q+w_cz_e^\top a^R+w_mm_e^R+w_Tz_e^\top t^R+w_Rr_e^\top r_q+w_{conf}c_e-w_{risk}p_e.
\]

Hypothetical/superseded traces are down-weighted by authority.  Generated traces have no factual authority.

## Neurogenesis

Birth score:

\[
\Omega_t=\sigma(w_NN_t+w_II_t+w_EE_t-w_BB_t).
\]

If `Ω_t > τ_birth`, allocate a reserved feature ID and attach it to the address with immature authority.  Maturation requires verified hits; false hits prune authority.

## Replay objective

Replay strengthens only verified traces:

\[
\mathcal L=\lambda_{src}\mathcal L_{sourceID}+\lambda_H\|\hat H-H\|^2+\lambda_{KL}D_{KL}(p_{full}\|p_{memory})+\lambda_m\max(0,S_{false}-S_{true}+m).
\]

The current CPU implementation updates graph/episode authority and quarantines unverified generated traces.  GPU payload training hooks should implement hidden-state reconstruction and teacher-KL terms.
