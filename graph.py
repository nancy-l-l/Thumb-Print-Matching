import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
#from functools import lru_cache
import numpy as np

class graph:
    def __init__(self) -> None:
        self.types: Dict[int, List["node"]] = defaultdict(list)
        self.nodes_at: Dict[Tuple[int, int], "node"] = {}   # quick lookup


class node:
    def __init__(self, x: int, y: int, val: int) -> None:
        #self.x = x                  # column index (j)
        #self.y = y                  # row    index (i)
        self.val = val              # feature label 1-5
        self.neighbors: List["node"] = []

    # optional, but it makes debugging more pleasant
    '''
    def __repr__(self) -> str:  
        return f"node(val={self.val}, yx=({self.y},{self.x}))"
    '''

def _vec_from_angle(theta: float) -> Tuple[float, float]:
    """
    Convert orientation in radians to a *row/col* direction vector.
    Rows grow downward → y = -sin,  cols grow rightward → x =  cos.
    """
    return -math.sin(theta), math.cos(theta)


def dfs(yy: float,
                      xx: float,
                      slopes: np.ndarray,
                      feature_map: np.ndarray,
                      visited: set,
                      foreground: set,
                      max_steps: int = 400,
                      dtheta: float = math.radians(10),
                      dtheta_increment: float = math.radians(10),
                      dtheta_max: float = math.radians(90)) -> Optional[Tuple[int, int]]:
    """
    Follow the local orientation from (yy,xx) until we hit another feature.
    If we do not hit anything inside `max_steps`, allow progressively wider
    angular deviation (dtheta) and try again.  Returns integer indices (r,c)
    of the first feature encountered or `None`.
    """
    if (int(round(xx)), int(round(yy))) not in foreground:
        return None
    H, W = slopes.shape
    start_theta = slopes[int(round(yy)), int(round(xx))]
    tolerance = dtheta

    while tolerance <= dtheta_max:
        # floating-point coordinates allow sub-block accuracy
        y_f, x_f = yy + 0.5, xx + 0.5
        visited.clear()

        for _ in range(max_steps):
            r = int(round(y_f))
            c = int(round(x_f))
            if not (0 <= r < H and 0 <= c < W):
                break  # walked out of the image

            if (r, c) in visited:          # we are cycling
                break
            visited.add((r, c))

            if feature_map[r, c] != 0 and (r, c) != (yy, xx):
                return r, c                # hit another node!

            theta = slopes[r, c]
            dy, dx = _vec_from_angle(theta)

            # take one *pixel* step in the principal direction
            y_f += dy
            x_f += dx

            # *optional*: allow sideways drift if the local direction
            # differs too much from the original – this acts like
            # a widening “cone” around the seed orientation.
            new_theta = slopes[int(round(y_f)) % H, int(round(x_f)) % W]
            if abs((new_theta - start_theta + math.pi) % (2*math.pi) - math.pi) > tolerance:
                break  # orientation drifted outside current cone

        tolerance += dtheta_increment      # widen the cone and try again

    return None

def build_graph(feature_map: np.ndarray,
                foreground: set,
                slopes: np.ndarray,
                *,
                initial_tolerance_deg: int = 10,
                tolerance_step_deg: int = 10,
                max_tolerance_deg: int = 90,
                max_steps: int = 400) -> graph:
    """
    Build the graph requested in the prompt.
    `feature_map` and `slopes` must have identical shape.
    """
    feature_map = np.asarray(feature_map)
    slopes      = np.asarray(slopes)
    assert feature_map.shape == slopes.shape, "arrays must be same shape"

    g = graph()

    # 1. create nodes for every non-zero feature
    it = np.nditer(feature_map, flags=["multi_index"])
    for val in it:
        if val == 0:
            continue
        r, c = it.multi_index
        n = node(c, r, int(val))
        g.nodes_at[(r, c)] = n
        g.types[int(val)].append(n)

    # 2. connect nodes by following the vector field
    dtheta      = math.radians(initial_tolerance_deg)
    dtheta_inc  = math.radians(tolerance_step_deg)
    dtheta_max  = math.radians(max_tolerance_deg)
    tmp_visited = set()

    for (r, c), n in g.nodes_at.items():
        nbr_loc = dfs(r,
                                    c,
                                    slopes,
                                    feature_map,
                                    tmp_visited,
                                    foreground,
                                    max_steps=max_steps,
                                    dtheta=dtheta,
                                    dtheta_increment=dtheta_inc,
                                    dtheta_max=dtheta_max)
        if nbr_loc is None:
            continue  # no neighbour found within tolerance
        if nbr_loc not in g.nodes_at:
            # this can happen if the neighbour was outside the image,
            # or if `feature_map` was updated after node creation
            continue

        m = g.nodes_at[nbr_loc]
        # undirected edge
        if m not in n.neighbors:
            n.neighbors.append(m)
        if n not in m.neighbors:
            m.neighbors.append(n)

    return g

from collections import defaultdict
from typing import Dict, List, Tuple


def all_nodes(g: "graph"):
    """Yield every node contained in g.types (value → [nodes])."""
    for lst in g.types.values():
        yield from lst

def wl_graph_similarity(
    g1: "graph",
    g2: "graph",
    k: int,
    threshold: float = 0.8,
) -> Tuple[float, bool]:
    
    labels1 = {n: n.val for n in all_nodes(g1)}
    labels2 = {n: n.val for n in all_nodes(g2)}

    def hist_sim(lbls_a: Dict["node", int], lbls_b: Dict["node", int]) -> float:
        freq_a, freq_b = defaultdict(int), defaultdict(int)
        for l in lbls_a.values():
            freq_a[l] += 1
        for l in lbls_b.values():
            freq_b[l] += 1

        common = sum(min(freq_a[l], freq_b[l]) for l in set(freq_a) | set(freq_b))
        total  = len(lbls_a) + len(lbls_b)
        return 1.0 if total == 0 else 2 * common / total

    cumulative_sim = hist_sim(labels1, labels2)

    for _ in range(1, k + 1):
        sig2colour = {}        
        new1, new2 = {}, {}

        for v in all_nodes(g1):
            neigh_labels = sorted(labels1[n] for n in v.neighbors)
            sig = (labels1[v], tuple(neigh_labels))
            if sig not in sig2colour:
                sig2colour[sig] = len(sig2colour)
            new1[v] = sig2colour[sig]

        for v in all_nodes(g2):
            neigh_labels = sorted(labels2[n] for n in v.neighbors)
            sig = (labels2[v], tuple(neigh_labels))
            if sig not in sig2colour:
                sig2colour[sig] = len(sig2colour)
            new2[v] = sig2colour[sig]

        labels1, labels2 = new1, new2
        cumulative_sim += hist_sim(labels1, labels2)

    score      = cumulative_sim / (k + 1)
    is_similar = score >= threshold
    return score, is_similar

'''
def graphs_equivalent_by_value(g1: graph, g2: graph) -> bool:
    """
    Return True iff `g1` and `g2` have
        • identical counts of each node value, and
        • identical counts of edges between every unordered pair of values.

    Two edges are considered the same if they join the same *values*
    (order-independent).  Works for undirected graphs produced by the earlier
    `build_graph()`.
    """

    def _summarise(g: graph):
        """Return (node_counts, edge_counts)."""
        # 1️⃣  how many nodes of each value?
        node_counts: dict[int, int] = {}
        for v, lst in g.types.items():          # g.types: value → [nodes]
            node_counts[v] = node_counts.get(v, 0) + len(lst)

        # 2️⃣  how many edges between each *pair* of values?
        edge_counts: dict[tuple[int, int], int] = {}
        seen_edges = set()                      # avoid double-counting

        for lst in g.types.values():            # walk every node once
            for n in lst:
                for nbr in n.neighbors:
                    # build a unique (Python-object-id) tag for the undirected edge
                    edge_id = tuple(sorted((id(n), id(nbr))))
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)

                    # unordered pair of *values*
                    key = tuple(sorted((n.val, nbr.val)))
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        #return node_counts, edge_counts
        return edge_counts
    
    #nc1, ec1 = _summarise(g1)
    #nc2, ec2 = _summarise(g2)

    #return nc1 == nc2 and ec1 == ec2
    

    ec1 = _summarise(g1)
    ec2 = _summarise(g2)

    return ec1 == ec2



#helppppp




def value_class(v: int) -> int:
    if v in (1, 2):
        return 0
    if v in (3, 4, 5):
        return 1
    raise ValueError(f"unexpected node value {v}")

def all_nodes(g: "graph"):
    for lst in g.types.values():
        yield from lst

def graphs_equivalent(g1: "graph", g2: "graph", max_iter: int = 20) -> bool:
    """
    Weisfeiler–Lehman iterative hashing.
    Converges in ≤|V| iterations; we cap at `max_iter` for safety.
    """
    def refine(g: "graph") -> Counter:
        colour = {n: value_class(n.val) for n in all_nodes(g)}   # step 0

        for _ in range(max_iter):
            bucket = {}                          # (prev_colour, multiset) → new_colour
            next_colour = {}

            for n in all_nodes(g):
                sig = (colour[n], tuple(sorted(colour[ch] for ch in n.neighbors)))
                if sig not in bucket:
                    bucket[sig] = len(bucket)    # assign fresh colour
                next_colour[n] = bucket[sig]

            if all(colour[n] == next_colour[n] for n in colour):  # stable ⇒ done
                break
            colour = next_colour

        return Counter(colour.values())          # multiset for whole graph

    return refine(g1) == refine(g2)

    """
    # 2⃣  compute canonical signature for every node ----------------
    @lru_cache(maxsize=None)
    def node_sig(n: "node") -> Tuple:
        """
        Signature is (cls, tuple(sorted(child_sigs))).
        Memoised so we can share work and handle cycles.
        """
        cls = value_class(n.val)
        child_sigs = sorted(node_sig(c) for c in n.neighbors)  # recursion
        return (cls, tuple(child_sigs))

    def graph_signature_multiset(g: "graph") -> Counter:
        """Return multiset of node signatures for the whole graph."""
        sigs = Counter()
        for n in all_nodes(g):
            sigs[node_sig(n)] += 1
        return sigs

    sigs1 = graph_signature_multiset(g1)
    sigs2 = graph_signature_multiset(g2)

    return sigs1 == sigs2
    """
'''