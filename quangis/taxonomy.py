"""
Methods and datatypes to manipulate taxonomies.
"""

from __future__ import annotations

import logging
from rdflib import URIRef
from typing import List, Optional, Dict

from quangis import error
from quangis.ontology import Ontology
from quangis.namespace import RDFS
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

    def subsumes(self, superconcept: URIRef, concept: URIRef) -> bool:
        parent = self.parent(concept)
        return concept == superconcept or \
            bool(parent and self.subsumes(superconcept, parent))

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

                if not self.subsumes(old_parent, new_parent):
                    raise error.NonUniqueParents(child, new_parent, old_parent)

                if self.subsumes(child, new_parent):
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
