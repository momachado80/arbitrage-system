"""
Market Dependency Graph
=======================

Infer and store relationships between markets:
  - complement     (A and not-A)
  - mutual_exclusive (outcomes of the same event)
  - conditional    (A depends on B)

Graph is bidirectional. Lookup is O(1) per market via adjacency dict.
No external dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple


VALID_RELATIONS = {"complement", "mutual_exclusive", "conditional"}


class MarketGraph:
    """Bidirectional graph of market relationships."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, List[Tuple[str, str]]] = {}

    def add_market(self, market_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.nodes[market_id] = metadata or {}

    def remove_market(self, market_id: str) -> None:
        self.nodes.pop(market_id, None)
        removed_targets = [t for t, _ in self.edges.pop(market_id, [])]
        for target in removed_targets:
            self.edges[target] = [
                (m, r) for m, r in self.edges.get(target, []) if m != market_id
            ]

    def add_edge(self, m1: str, m2: str, relation: str) -> None:
        if relation not in VALID_RELATIONS:
            return
        for mid in (m1, m2):
            if mid not in self.nodes:
                self.add_market(mid)
        if not any(t == m2 and r == relation for t, r in self.edges.get(m1, [])):
            self.edges.setdefault(m1, []).append((m2, relation))
        if not any(t == m1 and r == relation for t, r in self.edges.get(m2, [])):
            self.edges.setdefault(m2, []).append((m1, relation))

    def related_markets(self, market_id: str) -> List[Tuple[str, str]]:
        return self.edges.get(market_id, [])

    def related_by_type(self, market_id: str, relation: str) -> List[str]:
        return [m for m, r in self.edges.get(market_id, []) if r == relation]

    def get_exclusive_groups(self) -> List[List[str]]:
        """Find connected components of mutual_exclusive edges."""
        visited: set = set()
        groups: List[List[str]] = []

        for node in self.nodes:
            if node in visited:
                continue
            neighbors = self.related_by_type(node, "mutual_exclusive")
            if not neighbors:
                continue
            group = {node}
            stack = list(neighbors)
            while stack:
                n = stack.pop()
                if n in group:
                    continue
                group.add(n)
                stack.extend(
                    m for m in self.related_by_type(n, "mutual_exclusive") if m not in group
                )
            visited.update(group)
            groups.append(sorted(group))

        return groups

    def get_complement_pairs(self) -> List[Tuple[str, str]]:
        """Return deduplicated complement pairs."""
        seen: set = set()
        pairs: List[Tuple[str, str]] = []
        for node in self.nodes:
            for target in self.related_by_type(node, "complement"):
                pair = tuple(sorted([node, target]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    @property
    def market_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.edges.values()) // 2

    def summary(self) -> Dict[str, Any]:
        return {
            "markets": self.market_count,
            "edges": self.edge_count,
            "exclusive_groups": len(self.get_exclusive_groups()),
            "complement_pairs": len(self.get_complement_pairs()),
        }
