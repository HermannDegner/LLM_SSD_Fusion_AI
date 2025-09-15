#define SSD_ALIGN_LEAP_DLL_EXPORTS
#include "ssd_align_leap_dll.h"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

/*
  SSD Core (Alignment + Leap dynamics)

  改善点:
    - softmax_temp: 最大値探索のバグ修正 / exp オーバーフロー防止 (z clip)
    - hrate 計算で指数クリップ
    - 数値安定性微調整
    - F フィールドは現在未使用で常に 0。将来的な指標拡張用 (fatigue/friction など)。
*/

struct SSDHandle {
  int N;
  int current;
  std::vector<double> kappa; /* N*N */
  std::vector<double> w;     /* N*N (将来のポリシー重み拡張用) */
  double E;                  /* Heat (accumulated) */
  double F;                  /* Reserved / 未使用 */
  double T;                  /* Temperature */
  std::vector<double> pi;    /* last policy distribution (size N) */
  SSDParams prm;
  std::mt19937_64 rng;
  std::normal_distribution<double> norm01;
  std::uniform_real_distribution<double> uni01;

  SSDHandle(int n, const SSDParams& p, uint64_t seed)
  : N(n),
    current(0),
    kappa(n*n, 0.0),
    w(n*n, 0.0),
    E(0.0),
    F(0.0),
    T(p.T0),
    pi(n, 1.0 / std::max(1,n)),
    prm(p),
    rng(seed),
    norm01(0.0, 1.0),
    uni01(0.0, 1.0) {}
};

/* --- helpers --- */
static inline double clip(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline int idx(int i, int j, int N){ return i*N + j; }

static void softmax_temp(const std::vector<double>& logits, double T, std::vector<double>& out) {
  int n = (int)logits.size();
  out.resize(n);
  if (n == 0) return;

  if (T <= 1e-8) {
    // argmax one-hot
    int arg = 0;
    for (int i=1;i<n;++i)
      if (logits[i] > logits[arg]) arg = i;
    std::fill(out.begin(), out.end(), 0.0);
    out[arg] = 1.0;
    return;
  }

  double maxv = logits[0];
  for (int i=1;i<n;++i)
    if (logits[i] > maxv) maxv = logits[i];

  double sum = 0.0;
  for (int i=0;i<n;++i) {
    double z = (logits[i] - maxv) / T;
    if (z > 60.0) z = 60.0;
    else if (z < -60.0) z = -60.0;
    double e = std::exp(z);
    out[i] = e;
    sum += e;
  }
  if (sum <= 0.0 || !std::isfinite(sum)) sum = 1.0;
  for (int i=0;i<n;++i) out[i] /= sum;
}

static double entropy_norm(const std::vector<double>& p) {
  int n = (int)p.size();
  if (n == 0) return 0.0;
  double H = 0.0;
  for (int i=0;i<n;++i) {
    double x = p[i] <= 1e-12 ? 1e-12 : p[i];
    H += - x * std::log(x);
  }
  double Hmax = std::log((double)n);
  return Hmax > 0.0 ? (H / Hmax) : 0.0;
}

/* --- API impl --- */

extern "C" SSDHandle* ssd_create(int32_t N, const SSDParams* params, uint64_t seed) {
  if (N <= 0) return nullptr;
  SSDParams p;
  if (params) {
    std::memcpy(&p, params, sizeof(SSDParams));
  } else {
    // defaults (matching Python Params)
    p.G0=0.5; p.g=0.7; p.eps_noise=0.0; p.eta=0.3; p.rho=0.3; p.lam=0.02; p.kappa_min=0.0;
    p.alpha=0.6; p.beta_E=0.15;
    p.Theta0=1.0; p.a1=0.5; p.a2=0.4; p.h0=0.2; p.gamma=0.8;
    p.T0=0.3; p.c1=0.5; p.c2=0.6;
    p.sigma=0.2;
    p.delta_w=0.2; p.delta_kappa=0.2; p.c0_cool=0.6; p.q_relax=0.1; p.eps_relax=0.01;
    p.eps0=0.02; p.d1=0.2; p.d2=0.2; p.b_path=0.5;
  }
  if (seed==0) seed=123456789ULL;
  return new SSDHandle(N, p, seed);
}

extern "C" void ssd_destroy(SSDHandle* h) {
  delete h;
}

extern "C" void ssd_get_params(SSDHandle* h, SSDParams* out) {
  if (!h || !out) return;
  std::memcpy(out, &h->prm, sizeof(SSDParams));
}

extern "C" void ssd_set_params(SSDHandle* h, const SSDParams* in) {
  if (!h || !in) return;
  std::memcpy(&h->prm, in, sizeof(SSDParams));
}

extern "C" int32_t ssd_get_N(SSDHandle* h) {
  return h ? h->N : 0;
}

extern "C" int32_t ssd_get_kappa_row(SSDHandle* h, int32_t row, double* out_buf, int32_t len) {
  if (!h || !out_buf || row<0 || row>=h->N) return 0;
  int N = h->N;
  int m = std::min(N, (int)len);
  std::memcpy(out_buf, h->kappa.data() + row*N, sizeof(double)*m);
  return m;
}

extern "C" void ssd_step(SSDHandle* h, double p, double dt, SSDTelemetry* out) {
  if (!h || h->N <= 0) return;
  int N = h->N;
  auto& prm = h->prm;

  /* ----- 1) AlignFlow ----- */
  std::vector<double> j(N*N, 0.0);
  double J_norm_sq = 0.0;
  for (int i=0;i<N*N;++i) {
    double val = (prm.G0 + prm.g * h->kappa[i]) * p;
    if (prm.eps_noise > 0.0) {
      val += prm.eps_noise * h->norm01(h->rng);
    }
    j[i] = val;
    J_norm_sq += val*val;
  }
  double J_norm = std::sqrt(J_norm_sq);

  /* ----- 2) UpdateKappa ----- */
  for (int i=0;i<N*N;++i) {
    double gain = prm.eta * (p * j[i] - prm.rho * j[i]*j[i]);
    double decay = prm.lam * (h->kappa[i] - prm.kappa_min);
    double k = h->kappa[i] + (gain - decay) * dt;
    h->kappa[i] = (k < prm.kappa_min) ? prm.kappa_min : k;
  }

  /* ----- 3) UpdateHeat ----- */
  double dE = prm.alpha * std::max(std::abs(p) - J_norm, 0.0) - prm.beta_E * h->E;
  h->E += dE * dt;
  if (h->E < 0.0) h->E = 0.0;

  /* ----- 4) Theta / Jump rate / Temperature ----- */
  double km = 0.0;
  for (double x: h->kappa) km += x;
  km /= (double)(N*N);

  // h->F は未使用 (将来拡張用)
  double Theta = prm.Theta0 + prm.a1*km - prm.a2*h->F;

  double x_rate = (h->E - Theta) / std::max(1e-8, prm.gamma);
  x_rate = clip(x_rate, -60.0, 60.0);
  double hrate = prm.h0 * std::exp(x_rate);

  double Hn = h->pi.empty() ? 1.0 : entropy_norm(h->pi);
  h->T = std::max(1e-6, prm.T0 + prm.c1*h->E - prm.c2*Hn);

  /* ----- 5) Jump decision ----- */
  bool did_jump = false;
  int rewired_to = h->current;

  double p_jump = 1.0 - std::exp(-hrate * dt);
  if (p_jump > 1.0) p_jump = 1.0;
  if (p_jump < 0.0) p_jump = 0.0;

  if (h->uni01(h->rng) < p_jump) {
    did_jump = true;

    std::vector<double> logits(N, 0.0);
    for (int k=0;k<N;++k) {
      logits[k] = h->kappa[idx(h->current,k,N)];
      if (k == h->current) logits[k] -= 1.0; // discourage self-loop
      logits[k] += prm.sigma * h->norm01(h->rng);
    }
    softmax_temp(logits, h->T, h->pi);

    // sample categorical
    double r = h->uni01(h->rng);
    double cdf = 0.0;
    int ksel = N-1;
    for (int k=0;k<N;++k) {
      cdf += h->pi[k];
      if (r <= cdf) { ksel = k; break; }
    }

    // Rewire & cooling
    h->w[idx(h->current, ksel, N)] += prm.delta_w;
    h->kappa[idx(h->current, ksel, N)] += prm.delta_kappa;
    h->E *= prm.c0_cool;
    h->current = ksel;
    rewired_to = ksel;

    // Relax top-q%
    int M = N*N;
    int qn = std::max(1, (int)std::round(prm.q_relax * M));
    std::vector<int> indices(M);
    for (int i=0;i<M;++i) indices[i]=i;
    std::partial_sort(indices.begin(), indices.begin()+qn, indices.end(),
      [&](int a, int b){ return std::abs(j[a]) > std::abs(j[b]); });
    for (int ii=0; ii<qn; ++ii) {
      int pos = indices[ii];
      double k = h->kappa[pos] - prm.eps_relax;
      h->kappa[pos] = (k < prm.kappa_min) ? prm.kappa_min : k;
    }
  } else {
    // epsilon-random exploration
    double eps = prm.eps0 + prm.d1*h->E - prm.d2*km;
    eps = clip(eps, 0.0, 1.0);
    if (h->uni01(h->rng) < eps) {
      int k = (int)std::floor(h->uni01(h->rng) * N);
      if (k == N) k = N-1;
      if (k != h->current) {
        h->w[idx(h->current, k, N)] += 0.05;
        h->kappa[idx(h->current, k, N)] += 0.05;
      }
    }
    // deterministic greedy
    int s = h->current;
    int best = s;
    double bestv = -std::numeric_limits<double>::infinity();
    for (int k=0;k<N;++k) {
      double v = h->kappa[idx(s,k,N)];
      if (k==s) v -= 1e-6;
      if (v > bestv){
        bestv=v;
        best=k;
      }
    }
    h->current = best;
    rewired_to = best;
  }

  double eff = (std::abs(p) > 1e-12) ? (J_norm / std::abs(p)) : 0.0;

  if (out) {
    out->E = h->E;
    out->Theta = Theta;
    out->h = hrate;
    out->T = h->T;
    out->H = h->pi.empty()? 1.0 : entropy_norm(h->pi);
    out->J_norm = J_norm;
    out->align_eff = eff;
    out->kappa_mean = km;
    out->current = h->current;
    out->did_jump = did_jump ? 1 : 0;
    out->rewired_to = rewired_to;
  }
}