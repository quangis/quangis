import unittest

from quangis import error
from quangis.transformation.type import TypeOperator, Variables, Definition, Typeclass

var = Variables()
Any = TypeOperator("Any")
Int = TypeOperator("Int", supertype=Any)
Str = TypeOperator("Str", supertype=Any)
T = TypeOperator.parameterized("T", 1)

subtype = Typeclass("Subtype", var.x, var.y)
subtype.instance(Any, Any)
subtype.instance(Int, Any)
subtype.instance(Str, Any)


class TestType(unittest.TestCase):

    @staticmethod
    def define(x) -> Definition:
        """
        Obtain a test definition.
        """
        if isinstance(x, tuple):
            return Definition('test', *x)
        else:
            return Definition('test', x)

    def apply(self, f, *xs, result=None):
        """
        Test if the application of one (or more) arguments to a function gives
        the correct result.
        """

        f = self.define(f).instance()
        for i in range(0, len(xs)-1):
            x = self.define(xs[i]).instance()
            f = f.apply(x)
        x = self.define(xs[-1]).instance()

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, f.apply, x)
        else:
            self.assertEqual(f.apply(x), result)

    def test_apply_non_function(self):
        self.apply(Int, Int, result=error.NonFunctionApplication)

    def test_simple_function_match(self):
        self.apply(Int ** Str, Int, result=Str)

    def test_simple_function_mismatch(self):
        self.apply(Int ** Str, Str, result=error.TypeMismatch)

    def test_complex_function_match(self):
        self.apply(T(Int) ** T(Str), T(Int), result=T(Str))

    def test_variable(self):
        self.apply(var.x ** T(var.x), Int, result=T(Int))

    def test_compose(self):
        self.apply(
            ((var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)),
            Int ** Str,
            Str ** Int,
            result=Str ** Str)

    def test_simple_subtype_match(self):
        self.apply(
            (var.any ** Any, subtype(var.any, Any)),
            Int,
            result=Any)

    def test_simple_subtype_mismatch(self):
        self.apply(Int ** Any, Any, result=error.TypeMismatch)

    def test_complex_subtype_match(self):
        self.apply(
            ((var.any ** var.any) ** Any, subtype(var.any, Any)),
            Int ** Int, result=Any)

    def test_complex_subtype_mismatch(self):
        self.apply((Int ** Any) ** Any, Any ** Any, result=error.TypeMismatch)

    def test_variable_subtype_match(self):
        self.apply(
            ((var.x ** var.y) ** var.x, subtype(var.y, Any)),
            (Int ** Any), result=Int)

    def test_variable_subtype_mismatch(self):
        self.apply(
            ((var.x ** var.y) ** var.x, subtype(var.y, Int)),
            (Int ** Any), result=error.ViolatedConstraint)

    def test_simple_constraints_passed(self):
        self.apply(
            (var.x ** var.x, subtype(var.x, Any)),
            Int,
            result=Int
        )

    def test_simple_constraints_subtype_passed(self):
        self.apply(
            (var.x ** var.x, subtype(var.x, Any)),
            Int,
            result=Int
        )

    def test_simple_constraints_subtype_violated(self):
        self.apply(
            (var.x ** var.x, subtype(var.x, Int)),
            Any,
            result=error.ViolatedConstraint
        )

    @unittest.skip("nope")
    def test_compose_something(self):
        self.apply(
            (var.x ** var.x, subtype((var.x ** var.x), Any ** var._)),
            Int,
            result=Int)


if __name__ == '__main__':
    unittest.main()
