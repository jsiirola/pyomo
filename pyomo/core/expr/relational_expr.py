# -*- coding: utf-8 -*-
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import operator

from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
    native_numeric_types,
    check_if_numeric_type,
    value,
)

from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .visitor import polynomial_degree

from . import numeric_expr

# -------------------------------------------------------
#
# Expression classes
#
# -------------------------------------------------------


class RelationalExpression(ExpressionBase, BooleanValue):
    __slots__ = ('_args_',)

    EXPRESSION_SYSTEM = ExpressionType.RELATIONAL

    def __init__(self, args):
        self._args_ = args

    def __bool__(self):
        if self.is_constant():
            return bool(self())
        raise PyomoException(
            """
Cannot convert non-constant Pyomo expression (%s) to bool.
This error is usually caused by using a Var, unit, or mutable Param in a
Boolean context such as an "if" statement, or when checking container
membership or equality. For example,
    >>> m.x = Var()
    >>> if m.x >= 1:
    ...     pass
and
    >>> m.y = Var()
    >>> if m.y in [m.x, m.y]:
    ...     pass
would both cause this exception.""".strip()
            % (self,)
        )

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[: self.nargs()]

    @deprecated(
        "is_relational() is deprecated in favor of "
        "is_expression_type(ExpressionType.RELATIONAL)",
        version='6.4.3',
    )
    def is_relational(self):
        return self.is_expression_type(ExpressionType.RELATIONAL)

    def is_potentially_variable(self):
        return any(
            arg.__class__ not in native_numeric_types and arg.is_potentially_variable()
            for arg in self._args_
        )

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return polynomial_degree(self)

    def _compute_polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans

    def __eq__(self, other):
        """
        Equal to operator

        This method is called when Python processes the statement::

            self == other
        """
        return numeric_expr._equality_dispatcher[self.__class__, other.__class__](
            self, other
        )


class RangedExpression(RelationalExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    args:
        args (tuple): child nodes
        strict (tuple): flags that indicate whether the inequalities are strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    # Shared tuples for the most common RangedExpression objects encountered
    # in math programming.  Creating a single (shared) tuple saves memory
    STRICT = {
        False: (False, False),
        True: (True, True),
        (True, True): (True, True),
        (False, False): (False, False),
        (True, False): (True, False),
        (False, True): (False, True),
    }

    def __init__(self, args, strict):
        super(RangedExpression, self).__init__(args)
        self._strict = RangedExpression.STRICT[strict]

    def nargs(self):
        return 3

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def _apply_operation(self, result):
        _l, _b, _r = result
        if not self._strict[0]:
            if not self._strict[1]:
                return _l <= _b and _b <= _r
            else:
                return _l <= _b and _b < _r
        elif not self._strict[1]:
            return _l < _b and _b <= _r
        else:
            return _l < _b and _b < _r

    def _to_string(self, values, verbose, smap):
        return "%s  %s  %s  %s  %s" % (
            values[0],
            "<="[: 2 - self._strict[0]],
            values[1],
            "<="[: 2 - self._strict[1]],
            values[2],
        )

    @property
    def strict(self):
        return self._strict


class InequalityExpression(RelationalExpression):
    """
    Inequality expressions, which define less-than or
    less-than-or-equal relations::

        x < y
        x <= y

    args:
        args (tuple): child nodes
        strict (bool): a flag that indicates whether the inequality is strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super().__init__(args)
        self._strict = strict

    def nargs(self):
        return 2

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap):
        return "%s  %s  %s" % (values[0], "<="[: 2 - self._strict], values[1])

    @property
    def strict(self):
        return self._strict

    def __lt__(self, other):
        """
        Less than operator

        This method is called when Python processes statements of the form::

            self < other
            other > self
        """
        return numeric_expr._inequality_dispatcher[self.__class__, other.__class__](
            self, other, True
        )

    def __gt__(self, other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::

            self > other
            other < self
        """
        return numeric_expr._inequality_dispatcher[other.__class__, self.__class__](
            other, self, True
        )

    def __le__(self, other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::

            self <= other
            other >= self
        """
        return numeric_expr._inequality_dispatcher[self.__class__, other.__class__](
            self, other, False
        )

    def __ge__(self, other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::

            self >= other
            other <= self
        """
        return numeric_expr._inequality_dispatcher[other.__class__, self.__class__](
            other, self, False
        )


def inequality(lower=None, body=None, upper=None, strict=False):
    """
    A utility function that can be used to declare inequality and
    ranged inequality expressions.  The expression::

        inequality(2, model.x)

    is equivalent to the expression::

        2 <= model.x

    The expression::

        inequality(2, model.x, 3)

    is equivalent to the expression::

        2 <= model.x <= 3

    .. note:: This ranged inequality syntax is deprecated in Pyomo.
        This function provides a mechanism for expressing
        ranged inequalities without chained inequalities.

    args:
        lower: an expression defines a lower bound
        body: an expression defines the body of a ranged constraint
        upper: an expression defines an upper bound
        strict (bool): A boolean value that indicates whether the inequality
            is strict.  Default is :const:`False`.

    Returns:
        A relational expression.  The expression is an inequality
        if any of the values :attr:`lower`, :attr:`body` or
        :attr:`upper` is :const:`None`.  Otherwise, the expression
        is a ranged inequality.
    """
    if lower is None:
        if body is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return numeric_expr._inequality_dispatcher[body.__class__, upper.__class__](
            body, upper, strict
        )
    if body is None:
        if lower is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return numeric_expr._inequality_dispatcher[lower.__class__, upper.__class__](
            lower, upper, strict
        )
    if upper is None:
        return numeric_expr._inequality_dispatcher[lower.__class__, body.__class__](
            lower, body, strict
        )
    return RangedExpression((lower, body, upper), strict)


class EqualityExpression(RelationalExpression):
    """
    Equality expression::

        x == y
    """

    __slots__ = ()
    PRECEDENCE = 9

    def nargs(self):
        return 2

    def __bool__(self):
        lhs, rhs = self.args
        if lhs is rhs:
            return True
        return super().__bool__()

    def _apply_operation(self, result):
        _l, _r = result
        return _l == _r

    def _to_string(self, values, verbose, smap):
        return "%s  ==  %s" % (values[0], values[1])


class NotEqualExpression(RelationalExpression):
    """
    Not-equal expression::

        x != y
    """

    __slots__ = ()

    def nargs(self):
        return 2

    def __bool__(self):
        lhs, rhs = self.args
        if lhs is not rhs:
            return True
        return super().__bool__()

    def _apply_operation(self, result):
        _l, _r = result
        return _l != _r

    def _to_string(self, values, verbose, smap):
        return "%s  !=  %s" % (values[0], values[1])


def tuple_to_relational_expr(args):
    if len(args) == 2:
        return EqualityExpression(args)
    else:
        return inequality(*args)
