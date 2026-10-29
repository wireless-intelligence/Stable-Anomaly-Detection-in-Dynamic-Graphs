import math
import numpy as np
import scipy.sparse as sp

from .bayesian_scorer import BayesianAnomalyScorer
from .state_evolution import (
    StableStateEvolution,
    StateEvolutionNoStable,
    StateEvolutionNoDecay,
    StateEvolutionNoExposure,
)
from .structure_evidence import AttributeEvidenceEnhanced, StructuralEvidence


class SAG:
    def __init__(
        self,
        num_nodes: int,
        config,
        features=None,
        adj=None,
        use_kernel: bool = True,
        use_attribute: bool = True,
        use_structure: bool = True,
    ):
        self.num_nodes = num_nodes
        self.config = config
        self.use_kernel = use_kernel
        self.use_attribute = use_attribute
        self.use_structure = use_structure

        self.scorer = BayesianAnomalyScorer(
            prior_anomaly_rate=config.prior_anomaly_rate,
            likelihood_anomaly=config.likelihood_anomaly,
            likelihood_normal=config.likelihood_normal,
            rho=config.rho,
            epsilon=config.epsilon,
            window_size=getattr(config, "window_size", 4096),
            num_nodes=num_nodes,
        )

        self._device = str(getattr(config, "device", "cpu"))
        self._use_torch = self._device.startswith("cuda")
        if self._use_torch:
            try:
                import torch
                if not torch.cuda.is_available():
                    self._use_torch = False; self._device = "cpu"
            except Exception:
                self._use_torch = False; self._device = "cpu"

        if self._use_torch:
            from .state_evolution_torch import StableStateEvolutionTorch
            self.state = StableStateEvolutionTorch(
                num_nodes=num_nodes,
                gamma=config.gamma, beta_upd=config.beta_upd,
                alpha=config.alpha, delta=config.delta,
                h_max=getattr(config, "h_max", None),
                kernel_dim=getattr(config, "kernel_dim", 4),
                decay_lambdas=getattr(config, "kernel_lambdas", None),
                time_mode=getattr(config, "time_mode", "global"),
                device=self._device,
            )
        else:
            self.state = StableStateEvolution(
                num_nodes=num_nodes,
                gamma=config.gamma, beta_upd=config.beta_upd,
                alpha=config.alpha, delta=config.delta,
                h_max=getattr(config, "h_max", None),
                kernel_dim=getattr(config, "kernel_dim", 4),
                decay_lambdas=getattr(config, "kernel_lambdas", None),
                time_mode=getattr(config, "time_mode", "global"),
            )

        self.kernel = None

        if use_attribute and features is not None:
            self.attr = AttributeEvidenceEnhanced(features=features, adj=adj)
        else:
            self.attr = AttributeEvidenceEnhanced()

        self.trained_ae_score = None
        if use_attribute and features is not None and adj is not None and getattr(config, "use_trained_ae", False):
            cached = getattr(config, "_trained_ae_score_cache", None)
            cache_key = (id(features), id(adj),
                         tuple(sorted((getattr(config, "trained_ae_cfg", {}) or {}).items())),
                         int(getattr(config, "seed", 42)))
            if cached is not None and cached.get("key") == cache_key:
                self.trained_ae_score = cached["score"]
            else:
                try:
                    from .trained_attr import UnsupervisedGCNAutoencoder
                    cfg = dict(getattr(config, "trained_ae_cfg", {}) or {})
                    cfg.setdefault("device", getattr(config, "device", "cpu"))
                    ae = UnsupervisedGCNAutoencoder(seed=config.seed if hasattr(config, "seed") else 42, **cfg)
                    ae_score = ae.fit_score(adj, features)
                    if not np.all(np.isfinite(ae_score)) or ae_score.std() < 1e-12:
                        self.trained_ae_score = None
                    else:
                        self.trained_ae_score = ae_score
                        try:
                            setattr(config, "_trained_ae_score_cache", {"key": cache_key, "score": ae_score})
                        except (AttributeError, TypeError):
                            pass
                except Exception as e:
                    print(f"[SAG] trained AE unavailable: {e}")
                    self.trained_ae_score = None

        if use_structure and adj is not None:
            self.struct = StructuralEvidence(adj=adj)
        else:
            self.struct = StructuralEvidence()

        self._node_boost = self._build_node_boost()

        self.t = 0
        self.edge_count = 0

    def _build_node_boost(self):
        if not getattr(self.config, "use_attr_boost", False):
            return None
        a = self.attr.attribute_evidence if self.attr is not None else None
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        if a.max() > a.min():
            a = (a - a.min()) / (a.max() - a.min() + 1e-12)
        return a

    def reset(self) -> None:
        self.scorer.reset()
        self.state.reset()
        self.t = 0
        self.edge_count = 0

    def process_edge(self, u: int, v: int) -> None:
        boost = 0.0
        if self.attr.attribute_evidence is not None:
            boost = 0.5 * (self.attr.attribute_evidence[u] + self.attr.attribute_evidence[v])
        delta_e = self.scorer.score_edge(u, v, external_boost=boost)
        for node in (u, v):
            self.state.update_single(int(node), delta_e)
        self.edge_count += 1
        self.t += 1

    def process_stream_batch(self, edges: np.ndarray, passes: int = 1, shuffle: bool = True, rng=None, edge_times: np.ndarray = None) -> None:
        if self._use_torch:
            from .state_evolution_torch import process_stream_torch
            cfg = self.config
            process_stream_torch(
                state=self.state, scorer=self.scorer, edges=edges,
                alpha=cfg.alpha, beta=cfg.beta_upd, delta=cfg.delta,
                rho=cfg.rho, epsilon=cfg.epsilon,
                attr_vec_np=self._node_boost,
                passes=passes, shuffle=shuffle,
                rng=(rng if rng is not None else np.random.default_rng(cfg.seed)),
                edge_times=edge_times,
                use_continuous_dev=bool(getattr(cfg, "use_continuous_dev", False)),
                dev_smoothing_alpha=float(getattr(cfg, "dev_smoothing_alpha", 2.0)),
                window_size=getattr(cfg, "window_size", 4096),
            )
            self.edge_count += int(edges.shape[0]) * passes
            self.t += int(edges.shape[0]) * passes
            return

        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        cfg = self.config
        ws = getattr(cfg, "window_size", 4096)
        alpha = cfg.alpha
        gamma = cfg.gamma
        beta = cfg.beta_upd
        delta = cfg.delta
        rho = cfg.rho
        inv_rho = 1.0 / rho
        log_rho = math.log(rho)
        use_continuous_dev = bool(getattr(cfg, "use_continuous_dev", False))
        epsilon = cfg.epsilon
        p_dev = self.scorer._posterior_dev
        p_stb = self.scorer._posterior_stable
        p_span = p_dev - p_stb
        use_kernel = False
        attr_vec = self._node_boost
        s = self.state.s
        exposure = self.state.exposure
        h_max = self.state.h_max
        use_decay = self.state.use_decay
        use_stable = self.state.use_stable
        use_exp = self.state.use_exposure
        gammas = self.state.gammas
        lambdas = self.state.lambdas
        last_time = self.state.last_time
        time_mode = self.state.time_mode
        st_basis = self.state.basis
        readout = self.state.readout
        R = self.state.R

        num_nodes = self.num_nodes
        node_curr = np.zeros(num_nodes, dtype=np.float64)
        node_prev = np.zeros(num_nodes, dtype=np.float64)

        if use_kernel:
            K = self.kernel
            sk = K.sketch
            lastt = K.last_time
            rates = K.decay_rates
            basis = K.basis

        use_real_time = edge_times is not None and len(edge_times) == edges.shape[0]
        if use_real_time:
            edge_times = np.asarray(edge_times, dtype=np.float64)
            t_min = float(edge_times.min())
            edge_times_local = edge_times - t_min
        else:
            edge_times_local = None

        for _ in range(passes):
            if shuffle:
                if use_real_time:
                    perm = np.arange(edges.shape[0])
                    rng.shuffle(perm)
                    edges = edges[perm]
                    edge_times_local = edge_times_local[perm]
                else:
                    rng.shuffle(edges)
            edge_prev = {}
            edge_curr = {}
            wcount = 0
            for i in range(edges.shape[0]):
                u = int(edges[i, 0])
                v = int(edges[i, 1])
                k = (u, v) if u <= v else (v, u)

                edge_curr[k] = edge_curr.get(k, 0) + 1
                node_curr[u] += 1.0
                node_curr[v] += 1.0
                wcount += 1

                cp_e = edge_prev.get(k, 0) + epsilon
                cc_e = edge_curr[k] + epsilon
                cp_u = node_prev[u] + epsilon
                cc_u = node_curr[u] + epsilon
                cp_v = node_prev[v] + epsilon
                cc_v = node_curr[v] + epsilon

                if use_continuous_dev:
                    smooth = float(getattr(cfg, "dev_smoothing_alpha", 2.0))
                    a_edge = abs(math.tanh(math.log((cc_e + smooth) / (cp_e + smooth)) / log_rho))
                    a_node_u = abs(math.tanh(math.log((cc_u + smooth) / (cp_u + smooth)) / log_rho))
                    a_node_v = abs(math.tanh(math.log((cc_v + smooth) / (cp_v + smooth)) / log_rho))
                    a_e_u = max(a_edge, a_node_u)
                    a_e_v = max(a_edge, a_node_v)
                    p_base_u = p_stb + a_e_u * p_span
                    p_base_v = p_stb + a_e_v * p_span
                else:
                    a_e = 1 if (cc_e > rho * cp_e or cc_e < inv_rho * cp_e) else 0
                    if a_e == 0:
                        if (cc_u > rho * cp_u or cc_u < inv_rho * cp_u
                                or cc_v > rho * cp_v or cc_v < inv_rho * cp_v):
                            a_e = 1
                    p_base_u = p_dev if a_e == 1 else p_stb
                    p_base_v = p_base_u

                for node, p_base in ((u, p_base_u), (v, p_base_v)):
                    if attr_vec is not None:
                        boost = float(attr_vec[node])
                        p_node = p_base + (1.0 - p_base) * boost
                        if p_node > 1.0:
                            p_node = 1.0
                    else:
                        p_node = p_base
                    if use_stable:
                        b_node = math.tanh(alpha * p_node)
                    else:
                        x = alpha * p_node
                        b_node = x if -1.0 <= x <= 1.0 else (1.0 if x > 0 else -1.0)
                    exposure[node] += 1
                    if use_decay:
                        if time_mode == "per_update" or R == 1:
                            s[node] *= gammas
                        else:
                            now_t = float(edge_times_local[i]) if use_real_time else float(self.t)
                            dt = now_t - last_time[node]
                            if dt > 0:
                                s[node] *= np.exp(-lambdas * dt)
                            last_time[node] = now_t
                    lam = exposure[node]
                    g = beta / (math.log1p(lam) + delta) if use_exp else beta
                    s[node] += g * st_basis * b_node

                self.t += 1

                if wcount >= ws:
                    edge_prev = edge_curr
                    edge_curr = {}
                    node_prev = node_curr
                    node_curr = np.zeros(num_nodes, dtype=np.float64)
                    wcount = 0

            self.edge_count += int(edges.shape[0])
            edge_prev = edge_curr
            edge_curr = {}
            node_prev = node_curr
            node_curr = np.zeros(num_nodes, dtype=np.float64)
            wcount = 0

        self.scorer.edge_prev = edge_prev
        self.scorer.edge_curr = edge_curr
        self.scorer.node_prev = node_prev
        self.scorer.node_curr = node_curr

    @staticmethod
    def _norm01(v: np.ndarray) -> np.ndarray:
        if v.max() > v.min():
            return (v - v.min()) / (v.max() - v.min() + 1e-12)
        return np.zeros_like(v)

    @staticmethod
    def _rank_normalize(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(v.shape[0], dtype=np.float64)
        if v.shape[0] > 1:
            ranks /= (v.shape[0] - 1)
        return ranks

    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        ra = SAG._rank_normalize(a)
        rb = SAG._rank_normalize(b)
        return float(np.corrcoef(ra, rb)[0, 1])

    @staticmethod
    def _tail_quality(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=np.float64)
        if v.std() < 1e-12:
            return 0.0
        z = (v - v.mean()) / v.std()
        kurt = float(np.mean(z ** 4) - 3.0)
        return max(0.0, kurt)

    def _adaptive_fuse(self, h_rank, a_rank, s_rank, h_raw, a_raw, s_raw, if_raw=None, if_rank=None, ae_raw=None, ae_rank=None, pm_raw=None, pm_rank=None):
        candidates = []
        if a_raw is not None:
            candidates.append(("attr", a_raw, a_rank))
        if s_raw is not None:
            candidates.append(("struct", s_raw, s_rank))
        if if_raw is not None:
            candidates.append(("iforest", if_raw, if_rank))
        if ae_raw is not None:
            candidates.append(("trained_ae", ae_raw, ae_rank))
        if pm_raw is not None:
            candidates.append(("pca_mahal", pm_raw, pm_rank))
        if not candidates:
            return h_rank

        # ablation hook: w/o evidence selection -> naive max-pool over ALL sources
        # (skips C1/C2/C3 rank-correlation filters from paper §4.6)
        if getattr(self.config, "_ablation_no_selection", False):
            return np.maximum.reduce([h_rank] + [rk for _, _, rk in candidates])

        # Consensus-based Evidence Selection (paper §4.6, robust variant).
        #
        # The original selection ANCHORS on the streaming-state predicate: every
        # other source is kept only if it agrees with `state`. That is correct
        # when `state` is informative, but on feature-less edge-stream datasets
        # (e.g. Bitcoin) the state predicate can be *inverted* w.r.t. the label;
        # anchoring on it then DROPS the correctly-oriented attribute / structural
        # / iforest evidence and collapses the fused score onto the bad anchor
        # (observed: AUROC 0.16 where max(attr,struct,iforest) alone gives 0.66).
        #
        # The consensus variant treats ALL predicates (state + candidates)
        # symmetrically: it scores each source by its MEAN rank-agreement with the
        # others and keeps the majority-consensus cluster, dropping anti-consensus
        # minority sources (an inverted state drops out; on datasets where every
        # predicate agrees the set is unchanged). This is fully label-free — it
        # uses only inter-predicate Spearman agreement, never the labels — so it
        # cannot leak (a dataset that is genuinely inverted for *all* predicates,
        # e.g. Elliptic-tx illicit-as-inlier, is left honestly low).
        if getattr(self.config, "fusion_consensus", True):
            sources = [("state", h_raw, h_rank)] + list(candidates)
            m = len(sources)
            if m == 1:
                return h_rank
            raws = [src[1] for src in sources]
            rho = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(i + 1, m):
                    r = self._spearman(raws[i], raws[j])
                    rho[i, j] = rho[j, i] = r
            # mean agreement of each source with the OTHER sources (diag is 0)
            mean_agree = rho.sum(axis=1) / max(1, m - 1)
            tau_anchor = float(getattr(self.config, "fusion_tau_anchor", -0.05))
            keep = mean_agree >= tau_anchor
            if not keep.any():
                keep[int(np.argmax(mean_agree))] = True
            kept = [i for i in range(m) if keep[i]]
            tau_top = float(getattr(self.config, "fusion_tau_top", 0.55))
            active_ranks = []
            for i in kept:
                rk = sources[i][2]
                top_mask = rk > 0.95
                if top_mask.sum() > 0 and len(kept) > 1:
                    others = [sources[j][2] for j in kept if j != i]
                    agreement = float(np.mean([o[top_mask].mean() for o in others]))
                    if agreement < tau_top:
                        continue
                active_ranks.append(rk)
            if not active_ranks:
                active_ranks = [sources[i][2] for i in kept]
            return np.maximum.reduce(active_ranks)

        # --- legacy state-anchored selection (fusion_consensus=False) ---
        all_ranks = {n: rk for n, _, rk in candidates}
        all_ranks["state"] = h_rank

        active_raws = [("state", h_raw)]
        active_ranks = [h_rank]
        for name, raw, rk in candidates:
            rho_h = self._spearman(h_raw, raw)
            if rho_h <= -0.05:
                continue
            disagree = False
            for _, other_raw in active_raws[1:]:
                if self._spearman(other_raw, raw) < -0.2:
                    disagree = True
                    break
            if disagree:
                continue
            top_mask = rk > 0.95
            if top_mask.sum() > 0:
                others = [arank for nm, arank in all_ranks.items() if nm != name]
                agreement = float(np.mean([arank[top_mask].mean() for arank in others]))
                if agreement < 0.55:
                    continue
            active_raws.append((name, raw))
            active_ranks.append(rk)
        return np.maximum.reduce(active_ranks)

    @staticmethod
    def _excess_kurtosis(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=np.float64)
        if v.std() < 1e-12:
            return 0.0
        z = (v - v.mean()) / v.std()
        return float(np.mean(z ** 4) - 3.0)

    def get_scores(self) -> np.ndarray:
        cfg = self.config
        mode = getattr(cfg, "fusion_mode", "auto")
        attr_mode = getattr(cfg, "attr_mode", "auto")

        raw_h = self.state.scores()
        raw_a = self.attr.attribute_evidence if self.attr.attribute_evidence is not None else None
        raw_s = self.struct.structural_score if self.struct.structural_score is not None else None
        raw_if = getattr(self.attr, "iforest_evidence", None)
        raw_ae = getattr(self, "trained_ae_score", None)
        raw_pm = getattr(self.attr, "pca_mahal_evidence", None)

        if attr_mode == "iforest":
            raw_a = None
            raw_ae = None
        elif attr_mode == "gcn":
            raw_if = None

        if not bool(getattr(cfg, "use_pca_mahal", False)):
            raw_pm = None

        h_rank = self._rank_normalize(raw_h)
        a_rank = self._rank_normalize(raw_a) if raw_a is not None else None
        s_rank = self._rank_normalize(raw_s) if raw_s is not None else None
        if_rank = self._rank_normalize(raw_if) if raw_if is not None else None
        ae_rank = self._rank_normalize(raw_ae) if raw_ae is not None else None
        pm_rank = self._rank_normalize(raw_pm) if raw_pm is not None else None

        if mode in ("auto", "adaptive"):
            return self._adaptive_fuse(h_rank, a_rank, s_rank, raw_h, raw_a, raw_s, raw_if, if_rank, raw_ae, ae_rank, raw_pm, pm_rank)

        if mode == "h_only":
            return h_rank

        tau_state_gate = float(getattr(cfg, "state_gate_tau", 0.0))

        if mode == "attr_only":
            if a_rank is not None and ae_rank is not None:
                a_combined_rank = np.maximum(a_rank, ae_rank)
                a_combined_raw = raw_a if (raw_a is not None and raw_ae is not None and raw_a.shape == raw_ae.shape) else (raw_ae if raw_ae is not None else raw_a)
            elif ae_rank is not None:
                a_combined_rank, a_combined_raw = ae_rank, raw_ae
            elif a_rank is not None:
                a_combined_rank, a_combined_raw = a_rank, raw_a
            elif if_rank is not None:
                a_combined_rank, a_combined_raw = if_rank, raw_if
            else:
                return h_rank
            if raw_h is not None and a_combined_raw is not None:
                rho_h = self._spearman(raw_h, a_combined_raw)
                if rho_h >= tau_state_gate:
                    return np.maximum(h_rank, a_combined_rank)
            return a_combined_rank

        if mode == "iforest_only":
            if pm_rank is not None and if_rank is not None:
                w = float(getattr(cfg, "pm_blend_weight", 0.7))
                base_rank = (1.0 - w) * if_rank + w * pm_rank
                base_raw = raw_if
            elif pm_rank is not None:
                base_rank, base_raw = pm_rank, raw_pm
            elif if_rank is not None:
                base_rank, base_raw = if_rank, raw_if
            elif a_rank is not None:
                base_rank, base_raw = a_rank, raw_a
            else:
                return h_rank
            if raw_h is not None and base_raw is not None:
                rho_h = self._spearman(raw_h, base_raw)
                if rho_h >= tau_state_gate:
                    return np.maximum(h_rank, base_rank)
            return base_rank

        if mode == "iforest_blend":
            if if_rank is None:
                return h_rank
            s = self.state.s
            if s.ndim == 2 and s.shape[1] > 1:
                h_fast_raw = s[:, -1]
            else:
                h_fast_raw = raw_h
            h_fast_rank = self._rank_normalize(h_fast_raw)
            blend_w = float(getattr(cfg, "iforest_blend_h_weight", 0.3))
            return (1.0 - blend_w) * if_rank + blend_w * h_fast_rank

        if mode == "max_pool":
            comps = [c for c in (h_rank, a_rank, if_rank, ae_rank) if c is not None]
            if not comps:
                return s_rank if s_rank is not None else h_rank
            return np.maximum.reduce(comps)

        if mode == "struct_only":
            return s_rank if s_rank is not None else h_rank

        if mode == "no_evidence_selection":
            comps = [c for c in (h_rank, a_rank, s_rank, if_rank, ae_rank, pm_rank) if c is not None]
            if not comps:
                return h_rank
            return np.maximum.reduce(comps)

        if a_rank is not None and ae_rank is not None:
            a_combined = np.maximum(a_rank, ae_rank)
            a_raw_combined = (raw_a + raw_ae) if (raw_a.shape == raw_ae.shape) else raw_a
        elif ae_rank is not None:
            a_combined = ae_rank
            a_raw_combined = raw_ae
        else:
            a_combined = a_rank
            a_raw_combined = raw_a

        return self._adaptive_fuse(h_rank, a_combined, s_rank, raw_h, a_raw_combined, raw_s, raw_if, if_rank)


def _build_variant(state_cls, use_kernel: bool = True, use_attribute: bool = True, use_structure: bool = True):
    def _ctor(num_nodes, config, features=None, adj=None):
        m = SAG(
            num_nodes, config, features=features, adj=adj,
            use_kernel=use_kernel, use_attribute=use_attribute, use_structure=use_structure,
        )
        m.state = state_cls(
            num_nodes=num_nodes,
            gamma=config.gamma,
            beta_upd=config.beta_upd,
            alpha=config.alpha,
            delta=config.delta,
            h_max=getattr(config, "h_max", None),
            kernel_dim=getattr(config, "kernel_dim", 4),
            decay_lambdas=getattr(config, "kernel_lambdas", None),
            time_mode=getattr(config, "time_mode", "global"),
        )
        return m

    return _ctor


SAGNoStable = _build_variant(StateEvolutionNoStable)
SAGNoDecay = _build_variant(StateEvolutionNoDecay)
SAGNoExposure = _build_variant(StateEvolutionNoExposure)


def SAGNoKernel(num_nodes, config, features=None, adj=None):
    class _CfgShim:
        pass
    cfg = _CfgShim()
    for k in dir(config):
        if k.startswith("_") or callable(getattr(config, k, None)):
            continue
        try:
            setattr(cfg, k, getattr(config, k))
        except (AttributeError, TypeError):
            pass
    cfg.kernel_dim = 1
    cfg.kernel_lambdas = None
    return SAG(num_nodes, cfg, features=features, adj=adj, use_kernel=False)


def SAGNoAttribute(num_nodes, config, features=None, adj=None):
    return SAG(num_nodes, config, features=features, adj=adj, use_attribute=False)


def SAGNoStructure(num_nodes, config, features=None, adj=None):
    return SAG(num_nodes, config, features=features, adj=adj, use_structure=False)


def create_sag_model(num_nodes, config, features=None, adj=None, ablation=None):
    if ablation == "no_selection":
        # Route through _adaptive_fuse so the C1/C2/C3 filter is reachable;
        # otherwise dataset configs like fusion_mode="attr_only" short-circuit
        # the path the ablation is meant to disable.
        try:
            setattr(config, "_ablation_no_selection", True)
            setattr(config, "_ablation_no_selection_prev_fusion",
                    getattr(config, "fusion_mode", "auto"))
            setattr(config, "fusion_mode", "auto")
        except (AttributeError, TypeError):
            pass
        return SAG(num_nodes, config, features=features, adj=adj)
    try:
        setattr(config, "_ablation_no_selection", False)
        prev = getattr(config, "_ablation_no_selection_prev_fusion", None)
        if prev is not None:
            setattr(config, "fusion_mode", prev)
            setattr(config, "_ablation_no_selection_prev_fusion", None)
    except (AttributeError, TypeError):
        pass
    if ablation in (None, "full"):
        return SAG(num_nodes, config, features=features, adj=adj)
    if ablation == "no_stable":
        return SAGNoStable(num_nodes, config, features=features, adj=adj)
    if ablation == "no_decay":
        return SAGNoDecay(num_nodes, config, features=features, adj=adj)
    if ablation in ("no_exposure", "no_fairness"):
        return SAGNoExposure(num_nodes, config, features=features, adj=adj)
    if ablation == "no_kernel":
        return SAGNoKernel(num_nodes, config, features=features, adj=adj)
    if ablation == "no_attribute":
        return SAGNoAttribute(num_nodes, config, features=features, adj=adj)
    if ablation == "no_structure":
        return SAGNoStructure(num_nodes, config, features=features, adj=adj)
    raise ValueError(f"Unknown ablation: {ablation}")
