"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.
"""
# A primer: A type consists of type operators and type variables. Type
# operators encompass basic types, parameterized types and functions. When
# applying an argument of type A to a function of type B ** C, the algorithm
# tries to bind variables in such a way that A becomes equal to B. Constraints
# can be added to variables to make place further conditions on them;
# otherwise, variables are universally quantified. Constraints are enforced
# whenever a relevant variable is bound.
# When we bind a type to a type variable, binding happens on the type variable
# object itself. That is why we make fresh copies of generic type
# expressions before using them or adding constraints to them. This means that
# pointers are somewhat interwoven --- keep this in mind.
# To understand the module, I recommend you start by reading the methods of the
# TypeTerm class.
from __future__ import annotations

from abc import ABC
from functools import partial
from itertools import chain
from collections import defaultdict
from typing import Dict, Optional, Iterable, Union, List, Callable, Set

from quangis import error


class Variables(defaultdict):
    """
    For convenient notation, we provide a dispenser for type variables. Instead
    of writing x = TypeVar() to introduce type variable x every time, we can
    just instantiate a var = Variables() object and get var.x, var.y on the
    fly. To get a wildcard variable, use 'var._'; the assumption is that a
    wildcard will never be used anywhere else, so it will return a new type
    variable every time.
    """

    def __init__(self):
        super().__init__(TypeVar)

    def __getattr__(self, key):
        if key == '_':
            return TypeVar(wildcard=True)
        return self[key]


class Definition(object):
    """
    This class defines a function: it knows its general type and constraints,
    and can generate fresh instances.
    """

    def __init__(
            self,
            name: str,
            t: TypeTerm,
            *args: Union[Constraint, int]):
        """
        Define a function type. Additional arguments are distinguished by their
        type. This helps simplify notation: we won't have to painstakingly
        write out every definition, and instead just provide a tuple of
        relevant information.
        """

        constraints = set()
        number_of_data_arguments = 0

        for arg in args:
            if isinstance(arg, Constraint):
                constraints.add(arg)
            elif isinstance(arg, int):
                number_of_data_arguments = arg
            else:
                raise ValueError(f"cannot use extra {type(arg)} in Definition")

        #bound_variables = set(t.variables())
        #for constraint in constraints:
        #    if not all(
        #            var.wildcard or var in bound_variables
        #            for param in constraint.params
        #            for var in param.variables()):
        #        raise ValueError(
        #            "all variables in a constraint must be bound by "
        #            "an occurrence in the accompanying type signature")

        self.name = name
        self.type = t
        self.type.constraints = constraints
        self.data = number_of_data_arguments

    def instance(self) -> TypeTerm:
        ctx: Dict[TypeVar, TypeVar] = {}
        return self.type.fresh(ctx)

    def __str__(self) -> str:
        return f"{self.name} : {self.type}"


class TypeTerm(ABC):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    def __init__(self):
        self.constraints = NotImplemented

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: TypeTerm) -> bool:
        return value == self or (
            isinstance(self, TypeOperator) and
            any(value in t for t in self.params))

    def __str__(self) -> str:
        res = ""
        if isinstance(self, TypeOperator):
            if self.name == 'function':
                res = f"({self.params[0]} ** {self.params[1]})"
            elif self.params:
                res = f'{self.name}({", ".join(str(t) for t in self.params)})'
            else:
                res = self.name
        elif isinstance(self, TypeVar):
            res = "_" if self.wildcard else f"x{self.id}"

        if self.constraints:
            return f"{res}, {', '.join(str(c) for c in self.constraints)}"
        return res

    def __pow__(self, other: TypeTerm) -> TypeOperator:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return TypeOperator('function', self, other)

    def is_resolved(self) -> bool:
        """
        Test if every variable in this type is resolved.
        """
        return all(var.bound is None for var in self.variables())

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain all type variables currently in the type expression.
        """
        if isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.params)):
                yield v
        else:
            a = self.resolve(full=False)
            if isinstance(a, TypeVar):
                yield a
            else:
                for v in a.variables():
                    yield v

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeTerm:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            new: TypeTerm = TypeOperator(
                self.name,
                *(t.fresh(ctx) for t in self.params),
                supertype=self.supertype)
            new.constraints = set(c.fresh(ctx) for c in self.constraints)
            return new
        elif isinstance(self, TypeVar):
            assert self.is_resolved(), \
                "Cannot create a copy of a type with bound variables"
            if self in ctx:
                return ctx[self]
            else:
                new = TypeVar(wildcard=self.wildcard)
                new.constraints = set(c.fresh(ctx) for c in self.constraints)
                ctx[self] = new
                return new
        raise ValueError(f"{self} is of type {type(self)}; neither a type nor a type variable")

    def unify(self, other: TypeTerm) -> None:
        """
        Bind variables such that both types become the same. Binding is a
        side-effect; use resolve() to consolidate the bindings.
        """
        a = self.resolve(full=False)
        b = other.resolve(full=False)
        if a is not b:
            if isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
                if a.match(b):
                    for x, y in zip(a.params, b.params):
                        x.unify(y)
                else:
                    raise error.TypeMismatch(a, b)
            else:
                if (isinstance(a, TypeOperator) and b in a) or \
                        (isinstance(b, TypeOperator) and a in b):
                    raise error.RecursiveType(a, b)
                elif isinstance(a, TypeVar) and a != b:
                    a.bind(b)
                elif isinstance(b, TypeVar) and a != b:
                    b.bind(a)

    def apply(self, arg: TypeTerm) -> TypeTerm:
        """
        Apply an argument to a function type to get its resolved output type.
        """
        if isinstance(self, TypeOperator) and self.name == 'function':
            input_type, output_type = self.params
            arg.unify(input_type)
            result = output_type.resolve()
            result.constraints = set.union(arg.constraints, self.constraints)
            return result
        else:
            raise error.NonFunctionApplication(self, arg)

    def resolve(self, full: bool = True) -> TypeTerm:
        """
        A `full` resolve obtains a version of this type with all bound
        variables replaced with their bindings. Otherwise, just resolve the
        current variable.
        """
        # do sth with constraints?
        if isinstance(self, TypeVar) and self.bound:
            return self.bound.resolve(full)
        elif full and isinstance(self, TypeOperator):
            return TypeOperator(
                self.name,
                *(t.resolve(full) for t in self.params),
                supertype=self.supertype)
        return self

    def compatible(
            self,
            other: TypeTerm,
            allow_subtype: bool = False) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo variables. Subtypes may be allowed on the self side.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other, allow_subtype) and all(
                s.compatible(t, allow_subtype)
                for s, t in zip(self.params, other.params))
        return True

    def consistent(self, other: TypeTerm) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo wildcards.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other) and all(
                s.consistent(t) for s, t in zip(self.params, other.params))
        return self == other or \
            (isinstance(self, TypeVar) and self.wildcard) or \
            (isinstance(other, TypeVar) and other.wildcard)


class TypeOperator(TypeTerm):
    """
    n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            *params: TypeTerm,
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.supertype = supertype
        self.params: List[TypeTerm] = list(params)
        self.constraints = set()

        assert all(not param.constraints for param in self.params), \
            "constraints may only appear on the top level"

        if self.name == 'function' and self.arity != 2:
            raise ValueError("functions must have 2 argument types")
        if self.supertype and (self.params or self.supertype.params):
            raise ValueError("only nullary types may have supertypes")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.match(other, allow_subtype=False) and \
               all(s == t for s, t in zip(self.params, other.params))
        else:
            return False

    def match(self, other: TypeOperator, allow_subtype: bool = False) -> bool:
        """
        Check if the top-level type operator matches another.
        """
        return (
            (self.name == other.name and self.arity == other.arity) or
            (allow_subtype and bool(
                self.supertype and self.supertype.match(other, allow_subtype)
            ))
        )

    @property
    def arity(self) -> int:
        return len(self.params)

    @staticmethod
    def parameterized(
            name: str,
            arity: int = 0) -> Callable[..., TypeOperator]:
        """
        Allowing us to define parameterized types in an intuitive way, while
        optionally fixing the arity of the operator.
        """
        if arity > 0:
            def f(*params):
                if len(params) != arity:
                    raise TypeError(
                        f"type operator {name} has arity {arity}, "
                        f"but was given {len(params)} parameter(s)")
                return TypeOperator(name, *params)
            return f
        else:
            return partial(TypeOperator)


class TypeVar(TypeTerm):
    """
    Type variable. Note that any bindings and constraints are bound to the
    actual object instance, so make a fresh copy before applying them if the
    variable is not supposed to be a concrete instance.
    """

    counter = 0

    def __init__(self, wildcard: bool = False):
        cls = type(self)
        self.id = cls.counter
        self.bound: Optional[TypeTerm] = None
        self.wildcard = wildcard
        self.constraints = set()
        cls.counter += 1

    def bind(self, binding: TypeTerm) -> None:
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        # Once a variable has been bound, its constraints must carry over to
        # the variables in its binding. Consider a variable x that is
        # constrained to T(A) or T(B); and it has now been bound to T(z). This
        # can work, but only if binding z will still trigger the check that the
        # initial constraint still holds.
        #for var in binding.variables():
        #    var.constraints = set.union(var.constraints, self.constraints)

        self.bound = binding

        #for constraint in self.constraints:
        #    constraint.enforce()


class Typeclass(object):
    def __init__(
            self,
            name: str,
            *params: TypeVar,
            dependent: Iterable[TypeVar] = ()):
        self.name = name
        self.params = list(params)
        self.dependent = list(dependent)
        self.instances: List[List[TypeOperator]] = []

    def is_dependent(self, idx: int) -> bool:
        return self.params[idx] in self.dependent

    def instance(
            self,
            *params: TypeOperator,
            constraints: Iterable[Constraint] = ()):
        self.instances.append(list(params))

    def __call__(self, *params: TypeTerm) -> Constraint:
        return Constraint(self, *params)

    @staticmethod
    def member(var: TypeVar, options: Iterable[TypeOperator]):
        """
        Ad-hoc extensional typeclass constructor.
        """
        typeclass = Typeclass("Membership")
        for option in options:
            typeclass.instance(option)
        return Constraint(typeclass, var)


class Constraint(object):
    def __init__(self, typeclass: Typeclass, *params: TypeTerm):
        self.typeclass = typeclass
        self.params = list(params)

        #if not all(
        #        isinstance(t, TypeOperator)
        #        and all(isinstance(p, TypeVar) for p in t.params)
        #        for t in self.params):
            #raise TypeError

    def __str__(self):
        return \
            f"{self.typeclass.name}({', '.join(str(p) for p in self.params)})"

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Constraint:
        return Constraint(
            self.typeclass,
            *[t.fresh(ctx) for t in self.params])

    def find(self) -> List[TypeTerm]:
        """
        Find the instance corresponding to this constraint. None if a single
        instance cannot yet be determined.
        """
        # is there a corresponding instance?
        for instance in self.typeclass.instances:
            if all(
                    (isinstance(cparam, TypeOperator) and
                        iparam.consistent(cparam.resolve(False))) or
                    self.typeclass.is_dependent(i)
                    for i, (iparam, cparam) in
                    enumerate(zip(instance, self.params))):
                return instance
        raise error.ViolatedConstraint(self)

    def enforce(self) -> None:

        return None
