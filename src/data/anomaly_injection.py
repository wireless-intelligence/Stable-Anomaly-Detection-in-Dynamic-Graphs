import numpy as np
import scipy.sparse as sp


class AnomalyInjector:
    def __init__(
        self,
        adj,
        features,
        clique_size: int = 15,
        num_cliques: int = 10,
        k_attr: int = 50,
        seed: int = 42,
        attr_distance_percentile: float = None,
        num_attr_anomalies: int = None,
    ):
        self.adj = sp.lil_matrix(adj).astype(np.float32)
        self.features = features.toarray() if sp.issparse(features) else np.asarray(features)
        self.features = self.features.astype(np.float32).copy()
        self._features_original = self.features.copy()
        self.clique_size = clique_size
        self.num_cliques = num_cliques
        self.k_attr = k_attr
        self.seed = seed
        self.attr_distance_percentile = attr_distance_percentile
        self.num_attr_anomalies = num_attr_anomalies
        self.num_nodes = self.adj.shape[0]
        self.structural_anomalies = set()
        self.attribute_anomalies = set()
        self.rng = np.random.default_rng(seed)

    def inject_structural(self, num_cliques: int = None, clique_size: int = None):
        nc = self.num_cliques if num_cliques is None else int(num_cliques)
        cs = self.clique_size if clique_size is None else int(clique_size)
        n_total = nc * cs
        if n_total <= 0:
            return self
        all_selected = self.rng.choice(self.num_nodes, size=n_total, replace=False)
        for c_idx in range(nc):
            clique = all_selected[c_idx * cs: (c_idx + 1) * cs]
            for i in range(cs):
                for j in range(i + 1, cs):
                    a, b = int(clique[i]), int(clique[j])
                    self.adj[a, b] = 1.0
                    self.adj[b, a] = 1.0
            self.structural_anomalies.update(clique.tolist())
        return self

    def inject_attribute(self, num_anomalies: int = None):
        if num_anomalies is None:
            n_anom = (self.num_attr_anomalies
                      if self.num_attr_anomalies is not None
                      else self.clique_size * self.num_cliques)
        else:
            n_anom = int(num_anomalies)
        if n_anom <= 0:
            return self
        pool = np.array(sorted(set(range(self.num_nodes)) - self.structural_anomalies), dtype=np.int64)
        if pool.shape[0] < n_anom:
            pool = np.arange(self.num_nodes)
        selected = self.rng.choice(pool, size=n_anom, replace=False)
        Xo = self._features_original
        pct = self.attr_distance_percentile
        for node_i in selected:
            candidates = self.rng.choice(self.num_nodes, size=min(self.k_attr, self.num_nodes), replace=False)
            candidates = candidates[candidates != node_i]
            if candidates.size == 0:
                continue
            diffs = Xo[candidates] - Xo[node_i]
            dists = np.linalg.norm(diffs, axis=1)
            if pct is None:
                pick_idx = int(np.argmax(dists))
            else:
                # Pick the candidate whose distance is at the requested
                # percentile of the distance distribution.
                order = np.argsort(dists)
                p = float(pct)
                if p < 0: p = 0.0
                if p > 100: p = 100.0
                k = int(np.clip(round((p / 100.0) * (order.shape[0] - 1)),
                                0, order.shape[0] - 1))
                pick_idx = int(order[k])
            pick = candidates[pick_idx]
            self.features[node_i] = Xo[pick]
            self.attribute_anomalies.add(int(node_i))
        return self

    def inject_all(self):
        self.inject_structural()
        self.inject_attribute()
        return self

    # ------------------------------------------------------------------
    # Multi-hop / non-clique structural anomaly types used by experiments
    # 9 (anomaly-type sweep, axis D) and 13 (failure-case analysis).
    # ------------------------------------------------------------------
    def inject_star(self, num_stars: int, leaves_per_star: int):
        """Hub anomalies: pick num_stars centers; each receives leaves_per_star
        edges to fresh nodes. Centers (and leaves) are marked anomalous."""
        if num_stars <= 0 or leaves_per_star <= 0:
            return self
        used = set(self.structural_anomalies)
        avail = np.array(sorted(set(range(self.num_nodes)) - used), dtype=np.int64)
        need = num_stars * (1 + leaves_per_star)
        if avail.shape[0] < need:
            avail = np.arange(self.num_nodes)
        chosen = self.rng.choice(avail, size=need, replace=False)
        for s_idx in range(num_stars):
            base = s_idx * (1 + leaves_per_star)
            hub = int(chosen[base])
            leaves = chosen[base + 1: base + 1 + leaves_per_star].tolist()
            for v in leaves:
                self.adj[hub, int(v)] = 1.0
                self.adj[int(v), hub] = 1.0
            self.structural_anomalies.add(hub)
            self.structural_anomalies.update(int(v) for v in leaves)
        return self

    def inject_chain(self, num_chains: int, chain_length: int):
        """Long-path anomalies: walk a non-self-intersecting chain of length
        chain_length and add the edges along it. All nodes on the chain are
        marked anomalous."""
        if num_chains <= 0 or chain_length <= 1:
            return self
        used = set(self.structural_anomalies)
        avail = np.array(sorted(set(range(self.num_nodes)) - used), dtype=np.int64)
        need = num_chains * chain_length
        if avail.shape[0] < need:
            avail = np.arange(self.num_nodes)
        chosen = self.rng.choice(avail, size=need, replace=False)
        for c_idx in range(num_chains):
            chain = chosen[c_idx * chain_length: (c_idx + 1) * chain_length].tolist()
            for i in range(len(chain) - 1):
                a, b = int(chain[i]), int(chain[i + 1])
                self.adj[a, b] = 1.0
                self.adj[b, a] = 1.0
            self.structural_anomalies.update(int(v) for v in chain)
        return self

    def inject_bipartite_burst(self, num_groups: int, left_size: int, right_size: int):
        """Burst anomalies: each group is a complete bipartite (left, right)
        block of size left_size x right_size. All involved nodes are marked
        anomalous."""
        if num_groups <= 0 or left_size <= 0 or right_size <= 0:
            return self
        used = set(self.structural_anomalies)
        avail = np.array(sorted(set(range(self.num_nodes)) - used), dtype=np.int64)
        need = num_groups * (left_size + right_size)
        if avail.shape[0] < need:
            avail = np.arange(self.num_nodes)
        chosen = self.rng.choice(avail, size=need, replace=False)
        block = left_size + right_size
        for g in range(num_groups):
            base = chosen[g * block: (g + 1) * block]
            L = base[:left_size].tolist()
            R = base[left_size:].tolist()
            for u in L:
                for v in R:
                    self.adj[int(u), int(v)] = 1.0
                    self.adj[int(v), int(u)] = 1.0
            self.structural_anomalies.update(int(v) for v in L)
            self.structural_anomalies.update(int(v) for v in R)
        return self

    def inject_community_bridge(self, num_bridges: int):
        """Cross-community bridge anomalies: identify connected components of
        the (current) graph and add num_bridges edges between previously-
        disconnected components. Bridge endpoints are marked anomalous."""
        if num_bridges <= 0:
            return self
        n_comp, comp_labels = sp.csgraph.connected_components(
            sp.csr_matrix(self.adj), directed=False)
        if n_comp < 2:
            # Synthetic fallback: random partition into 2 halves.
            comp_labels = self.rng.integers(0, 2, size=self.num_nodes)
            n_comp = 2
        comp_to_nodes = {c: np.where(comp_labels == c)[0]
                         for c in range(int(n_comp))}
        comps = [c for c, arr in comp_to_nodes.items() if arr.size > 0]
        if len(comps) < 2:
            return self
        for _ in range(num_bridges):
            ca, cb = self.rng.choice(comps, size=2, replace=False)
            u = int(self.rng.choice(comp_to_nodes[int(ca)]))
            v = int(self.rng.choice(comp_to_nodes[int(cb)]))
            if u == v:
                continue
            self.adj[u, v] = 1.0
            self.adj[v, u] = 1.0
            self.structural_anomalies.add(u)
            self.structural_anomalies.add(v)
        return self

    # role_equivalence / motif anomalies remain unimplemented; documented in
    # outputs/results/revision/UNIMPLEMENTED.md
    def inject_role_equivalence(self, num_pairs: int):
        raise NotImplementedError(
            "role-equivalence anomaly injection is not implemented; see "
            "results/revision/UNIMPLEMENTED.md")

    def inject_motif(self, motif_pattern: str, num_instances: int):
        raise NotImplementedError(
            "motif anomaly injection is not implemented; see "
            "results/revision/UNIMPLEMENTED.md")

    def get_labels(self):
        labels = np.zeros(self.num_nodes, dtype=np.int64)
        for n in (self.structural_anomalies | self.attribute_anomalies):
            labels[n] = 1
        return labels

    def get_data(self):
        return {
            "adj": sp.csr_matrix(self.adj),
            "features": sp.csr_matrix(self.features),
            "labels": self.get_labels(),
            "structural_anomalies": list(self.structural_anomalies),
            "attribute_anomalies": list(self.attribute_anomalies),
            "num_structural": len(self.structural_anomalies),
            "num_attribute": len(self.attribute_anomalies),
        }
