"""
Methods and datatypes to manipulate ontologies and taxonomies.
"""

from __future__ import annotations

import rdflib
from rdflib import Graph, URIRef, BNode
from rdflib.term import Node
import itertools
import logging
import owlrl
from typing import Iterable, List, Optional, Dict

from quangis import error
from quangis import namespace
from quangis.namespace import TOOLS, ADA, CCD, RDFS, OWL, RDF
from quangis.util import shorten


class Taxonomy(object):
    """
    A taxonomy is a subsumption tree: unique subclass relations to a single
    root. Since in general, a directed acyclic graph cannot be turned into a
    tree (see Topological ordering of a DAG), this will raise an error if there
    is a cycle or if something is a subclass of two classes that are not
    subclasses to eachother. However, transitive relations are automatically
    removed into a minimal set of subsumption relations.
    """

    def __init__(self, root: URIRef):
        self._root = root
        self._parents: Dict[URIRef, URIRef] = {}
        self._depth: Dict[URIRef, int] = {}

    def __str__(self, node=None, level=0) -> str:
        node = node or self._root

        result = "\t"*level + "`- " + shorten(node) + " (" + str(self._depth.get(node, 0)) + ")\n"
        for child in self.children(node):
            result += self.__str__(node=child, level=level+1)
        return result

    @property
    def root(self) -> URIRef:
        return self._root

    def depth(self, node: URIRef) -> int:
        d = self._depth.get(node)
        if d:
            return d
        elif node == self.root:
            return 0
        else:
            raise error.Key("node does not exist")

    def parent(self, node: URIRef) -> Optional[URIRef]:
        return self._parents.get(node)

    def children(self, node: URIRef) -> List[URIRef]:
        # Not the most efficient, but fine for our purposes since trees will be
        # small and we hardly ever need to query for children
        return [k for k, v in self._parents.items() if v == node]

    def contains(self, node: URIRef):
        return node == self.root or node in self._parents

    def subsumed(self, concept: URIRef, superconcept: URIRef) -> bool:
        parent = self.parent(concept)
        return concept == superconcept or \
            bool(parent and self.subsumed(parent, superconcept))

    def add(self, parent: URIRef, child: URIRef):
        if child == self.root:
            raise error.Cycle()

        parent_depth = self._depth.get(parent, 0)
        if parent_depth == 0 and parent != self.root:
            raise error.DisconnectedTree()

        if not self.contains(child):
            self._parents[child] = parent
            self._depth[child] = parent_depth + 1
        elif parent != child:
            # If the child already exists, things get hairy. We can overwrite
            # the current parent relation, but ONLY if the new depth is deeper
            # AND the new parent is subsumed by the old parent anyway AND we
            # don't introduce cycles

            old_depth = self._depth[child]
            old_parent = self._parents[child]
            new_parent = parent
            new_depth = parent_depth + 1

            if new_depth >= old_depth:

                if not (self.subsumed(new_parent, old_parent)):
                    raise error.NonUniqueParents(child, new_parent, old_parent)

                if self.subsumed(new_parent, child):
                    raise error.Cycle()

                self._parents[child] = new_parent
                self._depth[child] = new_depth

    @staticmethod
    def from_ontology(
            ontology: Ontology,
            root: URIRef,
            predicate: URIRef = RDFS.subClassOf) -> Taxonomy:

        result = Taxonomy(root)

        def f(node):
            for child in ontology.subjects(predicate, node):
                try:
                    result.add(node, child)
                    f(child)
                except error.NonUniqueParents as e:
                    logging.warning(
                        "{new} will not be a superclass of {child} in the "
                        "taxonomy tree of {dim}; no subsumption with the "
                        "existing superclass {old}".format(
                            new=shorten(e.new),
                            old=shorten(e.old),
                            child=shorten(e.child),
                            dim=shorten(root)
                        )
                    )

        f(root)
        return result


class Ontology(Graph):
    """
    An ontology is simply an RDF graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for prefix, ns in namespace.mapping.items():
            self.bind(prefix, str(ns))

    def dimensionality(self, concept: URIRef,
                       dimensions: Iterable[URIRef]) -> int:
        """
        By how many dimensions is the given concept subsumed?
        """
        return sum(1 for d in dimensions if self.subsumed_by(concept, d))

    def expand(self) -> None:
        """
        Expand deductive closure under RDFS semantics.
        """
        owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(self)

    def subsumed_by(self, concept: URIRef, superconcept: URIRef) -> bool:
        """
        Is a concept subsumed by a superconcept in this taxonomy?
        """
        return concept == superconcept or any(
            s == superconcept or self.subsumed_by(s, superconcept)
            for s in self.objects(subject=concept, predicate=RDFS.subClassOf))

    def contains(self, node: Node) -> bool:
        """
        Does this graph contain some node?
        """
        return (node, None, None) in self or (None, None, node) in self

    def leaves(self) -> List[Node]:
        """
        Determine the exterior nodes of a taxonomy.
        """
        return [
            n for n in self.subjects(predicate=RDFS.subClassOf, object=None)
            if not (None, RDFS.subClassOf, n) in self]

    def debug(self) -> None:
        """
        Log this ontology to the console to debug.
        """
        result = [""] + [
            "    {} {} {}".format(shorten(o), shorten(p), shorten(s))
            for (o, p, s) in self.triples((None, None, None))]
        logging.debug("\n".join(result))

    @staticmethod
    def from_rdf(path: str, format: str = None) -> Ontology:
        g = Ontology()
        g.parse(path, format=format or rdflib.util.guess_format(path))
        return g

    def is_taxonomy(self) -> bool:
        """
        A taxonomy is an RDF graph consisting of raw subsumption relations ---
        rdfs:subClassOf statements.
        """
        return all(p == RDFS.subClassOf for p in self.predicates())
        # TODO also, no loops

    def core(self, dimensions: List[URIRef]) -> Ontology:
        """
        This method generates a taxonomy where nodes intersecting with more
        than one dimension (= not core) are removed. This is needed because APE
        should reason only within any of the dimensions.
        """

        result = Ontology()
        for (s, p, o) in self.triples((None, RDFS.subClassOf, None)):
            if self.dimensionality(s, dimensions) == 1:
                result.add((s, p, o))
        return result

    def subsumptions(self, root: URIRef) -> Ontology:
        """
        Take an arbitrary root and generate a new tree with only parent
        relations toward the root. Note that the relations might not be unique.
        """

        result = Ontology()

        def f(node, children):
            for child, grandchildren in children:
                result.add((child, RDFS.subClassOf, node))
                f(child, grandchildren)

        f(*rdflib.util.get_tree(self, root, RDFS.subClassOf))
        return result


def clean_owl_ontology(ontology: Ontology,
                       dimensions: List[URIRef]) -> Ontology:
    """
    This method takes some ontology and returns an OWL taxonomy. (consisting
    only of rdfs:subClassOf statements)
    """

    taxonomy = Ontology()

    # Only keep subclass nodes intersecting with exactly one dimension
    for (o, p, s) in itertools.chain(
            ontology.triples((None, RDFS.subClassOf, None)),
            ontology.triples((None, RDF.type, OWL.Class))
            ):
        if type(s) != BNode and type(o) != BNode \
                and s != o and s != OWL.Nothing and \
                ontology.dimensionality(o, dimensions) == 1:
            taxonomy.add((o, p, s))

    # Add common upper class for all data types
    taxonomy.add((CCD.Attribute, RDFS.subClassOf, CCD.DType))
    taxonomy.add((ADA.SpatialDataSet, RDFS.subClassOf, CCD.DType))
    taxonomy.add((ADA.Quality, RDFS.subClassOf, CCD.DType))

    return taxonomy


def extract_tool_ontology(tools: Ontology) -> Ontology:
    """
    Extracts a taxonomy of toolnames from the tool description.
    """

    taxonomy = Ontology()
    for (s, p, o) in tools.triples((None, TOOLS.implements, None)):
        taxonomy.add((o, RDFS.subClassOf, s))
        taxonomy.add((s, RDF.type, OWL.Class))
        taxonomy.add((o, RDF.type, OWL.Class))
        taxonomy.add((s, RDFS.subClassOf, TOOLS.Tool))
    return taxonomy
