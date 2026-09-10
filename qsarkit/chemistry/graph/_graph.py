"""Molecular graph conversion and graph-theoretic analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


class MolecularGraph:
    """Convert RDKit molecules to NetworkX graphs and compute graph descriptors.

    Atoms become nodes carrying ``symbol``, ``atomic_num``,
    ``formal_charge``, ``is_aromatic``, ``in_ring``, ``degree`` and
    ``hybridization``; bonds become edges carrying ``bond_type``,
    ``is_aromatic``, ``in_ring`` and ``order``. This is the substrate for
    the topological indices below, and for any analysis that is easier to
    express with NetworkX than with RDKit's own graph API.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import MolecularGraph
    >>> graph = MolecularGraph()
    >>> G = graph.to_networkx(Chem.MolFromSmiles("CCO"))
    >>> G.number_of_nodes(), G.number_of_edges()
    (3, 2)
    >>> G.nodes[2]["symbol"], G.nodes[2]["hybridization"]
    ('O', 'SP3')

    Hydrogens are implicit, so a node's heavy-atom ``degree`` and its
    ``num_hs`` are reported separately:

    >>> G.nodes[0]["degree"], G.nodes[0]["num_hs"]
    (1, 3)

    :meth:`descriptors` returns the classic topological indices for a
    single molecule:

    >>> d = graph.descriptors(Chem.MolFromSmiles("c1ccccc1"))
    >>> d["num_rings"], d["cyclomatic_number"], d["diameter"]
    (1, 1, 3)
    >>> round(d["wiener_index"], 1)
    27.0

    References
    ----------
    - Hagberg, A., Schult, D. & Swart, P. (2008). "Exploring Network
      Structure, Dynamics, and Function using NetworkX." Proc. SciPy 2008.
      https://www.osti.gov/biblio/960616
    - Wiener, H. (1947). "Structural Determination of Paraffin Boiling
      Points." J. Am. Chem. Soc., 69(1), 17-20.
      https://doi.org/10.1021/ja01193a005
    - Balaban, A. T. (1982). "Highly Discriminating Distance-Based
      Topological Index." Chem. Phys. Lett., 89(5), 399-404.
      https://doi.org/10.1016/0009-2614(82)80009-2
    - RDKit graph descriptors documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.GraphDescriptors.html
    """

    def to_networkx(self, mol: Any) -> "nx.Graph":
        """Build a ``networkx.Graph`` from an RDKit Mol."""
        import networkx as nx

        graph = nx.Graph()
        for atom in mol.GetAtoms():
            graph.add_node(
                atom.GetIdx(),
                symbol=atom.GetSymbol(),
                atomic_num=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                is_aromatic=atom.GetIsAromatic(),
                in_ring=atom.IsInRing(),
                degree=atom.GetDegree(),
                hybridization=str(atom.GetHybridization()),
                num_hs=atom.GetTotalNumHs(),
            )
        for bond in mol.GetBonds():
            graph.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=str(bond.GetBondType()),
                is_aromatic=bond.GetIsAromatic(),
                in_ring=bond.IsInRing(),
                order=bond.GetBondTypeAsDouble(),
            )
        return graph

    def descriptors(self, mol: Any) -> Dict[str, float]:
        """Compute topological / graph-theoretic descriptors for one molecule."""
        import networkx as nx
        from rdkit.Chem import GraphDescriptors, rdMolDescriptors

        graph = self.to_networkx(mol)
        n = graph.number_of_nodes()
        connected = n > 0 and nx.is_connected(graph)

        return {
            "num_atoms": n,
            "num_bonds": graph.number_of_edges(),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "cyclomatic_number": graph.number_of_edges() - n + 1 if n else 0,
            "diameter": nx.diameter(graph) if connected else float("nan"),
            "radius": nx.radius(graph) if connected else float("nan"),
            "average_shortest_path": (
                nx.average_shortest_path_length(graph) if connected and n > 1 else 0.0
            ),
            "wiener_index": (
                sum(
                    length
                    for _, targets in nx.all_pairs_shortest_path_length(graph)
                    for length in targets.values()
                )
                / 2
                if connected
                else float("nan")
            ),
            "balaban_j": GraphDescriptors.BalabanJ(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol),
            "chi0": GraphDescriptors.Chi0(mol),
            "chi1": GraphDescriptors.Chi1(mol),
            "kappa1": GraphDescriptors.Kappa1(mol),
            "kappa2": GraphDescriptors.Kappa2(mol),
        }

    def transform(self, mols: Any) -> List[Optional[Dict[str, float]]]:
        """Compute :meth:`descriptors` for every molecule in an ``Iterable[Mol]``."""
        return [self.descriptors(m) if m is not None else None for m in mols]
