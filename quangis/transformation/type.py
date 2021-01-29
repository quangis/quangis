"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.

Be warned: This module abuses overloading of Python's standard operators.
"""
from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Dict, Optional, Iterable, Union, List, Callable, Tuple

from quangis import error


class Definition(object):
    """
    This class defines a function: it knows its general type and constraints,
    plus additional information that may be used by some parser, and can
    generate fresh instances of the function.
    """

    def __init__(
            self,
            name: str,
            type: AlgebraType,
            *constraints: Constraint,
            data: int = 0):
        self.name = name
        self.type = type
        self.constraints = list(constraints)
        self.data = data

    def instance(self) -> AlgebraType:
        ctx: Dict[TypeVar, TypeVar] = {}
        t = self.type.fresh(ctx)
        for constraint in self.constraints:
            new_constraint = constraint.fresh(ctx)
            for var in new_constraint.subject.variables():
                var.constraints.add(new_constraint)
        return t

    @staticmethod
    def from_tuple(
            name: str,
            values: Union[AlgebraType, tuple]) -> Definition:
        """
        This method is an alternative way of defining: it allows us to simply
        write a tuple of relevant information. This can simplify notation.
        """
        if isinstance(values, AlgebraType):
            return Definition(name, values)
        else:
            t = values[0]
            constraints: List[Constraint] = []
            data = 0
            for v in values[1:]:
                if isinstance(v, Constraint):
                    constraints.append(v)
                elif isinstance(v, int):
                    data = v
                else:
                    raise ValueError(f"cannot use type {type(v)} in Definition")
            return Definition(name, t, *constraints, data=data)

    def __str__(self) -> str:
        return (
            f"{self.name} : {self.type}{', ' if self.constraints else ''}"
            f"{', '.join(str(c) for c in self.constraints)}"
        )


class TypeDefiner(ABCMeta):
    """
    Allowing us to write type definitions in an intuitive way, such as
    TypeOperator.Int() for basic types or TypeOperator.Tuple for parameterized
    types.
    """
    # TODO add a parameter that fixes the arity of the operator or perhaps even
    # constrains it arguments

    def __getattr__(self, key: str) -> Callable[..., TypeOperator]:
        return partial(TypeOperator, key)


class AlgebraType(ABC, metaclass=TypeDefiner):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __contains__(self, value: AlgebraType) -> bool:
        return NotImplemented

    @abstractmethod
    def instantiate(self) -> AlgebraType:
        return NotImplemented

    def __pow__(self, other: AlgebraType) -> TypeOperator:
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

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain all type variables left in the type expression.
        """
        if isinstance(self, TypeVar):
            yield self
        elif isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.types)):
                yield v

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            new = TypeOperator(self.name, *(t.fresh(ctx) for t in self.types))
            new.supertype = self.supertype
            return new
        elif isinstance(self, TypeVar):
            if self.bound:
                raise error.AlreadyBound(self)
            elif self in ctx:
                return ctx[self]
            else:
                new2 = TypeVar()
                for tc in self.constraints:
                    new2.constraints.add(tc)
                ctx[self] = new2
                return new2
        raise ValueError(f"{self} is neither a type nor a type variable")

    def unify(self: AlgebraType, other: AlgebraType) -> None:
        """
        Bind variables such that both types become the same. Note that subtypes
        on the "self" side are tolerated: that is, if self is a subtype of
        other, then they are considered the same, but not vice versa.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):

            if self.match(other):
                for x, y in zip(self.types, other.types):
                    x.unify(y)
            else:
                raise error.TypeMismatch(self, other)
        else:
            if isinstance(self, TypeVar):
                if self != other and self in other:
                    raise error.RecursiveType(self, other)
                a = self.binding
                b = self.binding
                a.bind(b)
            elif isinstance(other, TypeVar):
                other.unify(self)

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its output type.
        """
        if isinstance(self, TypeOperator) and self.is_function():
            input_type, output_type = self.types
            arg.instantiate().unify(input_type.instantiate())
            return output_type.instantiate()
        else:
            raise error.NonFunctionApplication(self, arg)

    def is_function(self) -> bool:
        if isinstance(self, TypeOperator):
            return self.name == 'function'
        return False

    def compatible(self, other: AlgebraType) -> bool:
        """
        Test if a type is structurally equivalent to another, that is, if
        disregarding variables could lead to the same type. Like for unify,
        subtypes are tolerated here
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other) and \
                all(s.compatible(t) for s, t in zip(self.types, other.types))
        return True


class TypeOperator(AlgebraType):
    """
    n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            *types: AlgebraType,
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.types = list(types)
        self.supertype = supertype

        if self.name == 'function' and self.arity != 2:
            raise ValueError("functions must have 2 argument types")
        if self.supertype and (self.types or self.supertype.types):
            raise ValueError("only nullary types may have supertypes")

    def match(self, other: TypeOperator, allow_subtype: bool = True) -> bool:
        """
        Check if the top-level type operator matches another (modulo subtypes).
        """
        return (
            (self.name == other.name and self.arity == other.arity) or
            (allow_subtype and bool(
                self.supertype and self.supertype.match(other)
            ))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.match(other, allow_subtype=False) and \
               all(s == t for s, t in zip(self.types, other.types))
        else:
            return False

    def __contains__(self, value: AlgebraType) -> bool:
        return value == self or any(value in t for t in self.types)

    def __str__(self) -> str:
        if self.is_function():
            return f"({self.types[0]} -> {self.types[1]})"
        elif self.types:
            return f'{self.name}({", ".join(str(t) for t in self.types)})'
        else:
            return self.name

    def instantiate(self) -> AlgebraType:
        self.types = [t.instantiate() for t in self.types]
        return self

    @property
    def arity(self) -> int:
        return len(self.types)


class TypeVar(AlgebraType):

    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        self.bound = None
        self.constraints = set()
        cls.counter += 1

    def __str__(self) -> str:
        return f"x{self.id}"

    def __contains__(self, value: AlgebraType) -> bool:
        return self == value

    def bind(self, binding: AlgebraType):
        assert (not self.bound or binding == self.bound), \
            f"variable {self} is already bound to {self.bound}, cannot be bound to {binding}"

        self.bound = binding

        for constraint in self.constraints:
            constraint.enforce()

    @property
    def binding(self) -> AlgebraType:
        return self.bound or self

    def instantiate(self) -> AlgebraType:
        #for c in self.constraints:
        #    tc.instantiate()

        if self.bound:
            return self.bound
        else:
            return self


class Constraint(object):
    """
    A constraint is a ...
    To avoid recursive types, the typeclass may not contain any variable from
    the type.
    """
    # The way it works is: we assign the constraint to all variables occurring
    # in said constraint. We don't do this immediately, but only after the
    # user has refreshed the variables.

    def __init__(self, t: AlgebraType, *options: AlgebraType):
        self.subject = t
        self.options = list(options)

        for t in self.options:
            for v in self.subject.variables():
                if v in t:
                    raise error.RecursiveType(v, t)

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Constraint:
        return Constraint(
            self.subject.fresh(ctx), *(t.fresh(ctx) for t in self.options)
        )

    def variables(self) -> Iterable[TypeVar]:
        return chain(
            self.subject.variables(),
            *(t.variables() for t in self.options))

    def enforce(self):
        # should only be called once variables are non-generic

        # Check that at least one of the typeclasses matches
        subject = self.subject.instantiate()
        matches = [
            t for t in self.options if
            subject.compatible(t.instantiate())
        ]

        if len(matches) == 0:
            raise error.ViolatedConstraint(self)
        #elif len(matches) == 1:
        #    subject.unify(matches[0])

    @staticmethod
    def has(
            subject: AlgebraType,
            op: Callable[..., TypeOperator],
            target: AlgebraType,
            at: Optional[int] = None):
        """
        Produce a constraint holding that the subject must be a type operator
        `op` containing the target somewhere in its parameters.
        """
        options: List[AlgebraType] = []
        if not at or at == 1:
            options.extend((
                op(target),
                op(target, TypeVar()),
                op(target, TypeVar(), TypeVar())
            ))
        if not at or at == 2:
            options.extend((
                op(TypeVar(), target),
                op(TypeVar(), target, TypeVar())
            ))
        if not at or at == 3:
            options.append(
                op(TypeVar(), TypeVar(), target)
            )
        return Constraint(subject, *options)


