#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Expression', '_ExpressionData']

import sys
import logging
from weakref import ref as weakref_ref

from pyomo.common.deprecation import deprecated
from pyomo.common.timing import ConstructionTimer

from pyomo.core.expr import current as EXPR
from pyomo.core.base.component import ComponentData
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.indexed_component import (
    IndexedComponent,
    UnindexedComponent_set, )
from pyomo.core.base.misc import (apply_indexed_rule,
                                  tabular_writer)
from pyomo.core.base.numvalue import (NumericValue,
                                      as_numeric)
from pyomo.core.base.util import is_functor, disable_methods, Initializer

from six import iteritems

logger = logging.getLogger('pyomo.core')

_EXPRESSION_API = {'expr', 'set_value', 'is_constant', 'is_fixed'}

class _ExpressionData(NumericValue):
    """
    An object that defines a named expression.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __slots__ = ()

    #
    # NumericValue Interface
    #

    def is_named_expression_type(self):
        """A boolean indicating whether this in a named expression."""
        return True

    def is_expression_type(self):
        """A boolean indicating whether this in an expression."""
        return True

    # _ExpressionData should never return False because
    # they can store subexpressions that contain variables
    def is_potentially_variable(self):
        return True

    def polynomial_degree(self):
        """A tuple of subexpressions involved in this expressions operation."""
        return self.expr.polynomial_degree()

    #
    # ExpressionBase Interface (duck-typed)
    #

    def __call__(self, exception=True):
        """Compute the value of this expression."""
        if self.expr is None:
            return None
        return self.expr(exception=exception)

    def arg(self, index):
        if index < 0 or index >= 1:
            raise KeyError("Invalid index for expression argument: %d" % index)
        return self.expr

    @property
    def _args_(self):
        return (self.expr,)

    @property
    def args(self):
        return (self.expr,)

    def nargs(self):
        return 1

    def _precedence(self):
        return 0

    def _associativity(self):
        return 0

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "%s{%s}" % (str(self), values[0])
        if self.expr is None:
            return "%s{None}" % str(self)
        return values[0]

    def clone(self):
        """Return a clone of this expression (no-op)."""
        return self

    def _apply_operation(self, result):
        # This "expression" is a no-op wrapper, so just return the inner
        # result
        return result[0]

    def _is_fixed(self, values):
        return values[0]

    def _compute_polynomial_degree(self, result):
        return result[0]

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this expression."""
        raise NotImplementedError

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        raise NotImplementedError

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        raise NotImplementedError


class _GeneralExpressionDataImpl(_ExpressionData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __pickle_slots__ = ('_expr',)

    # any derived classes need to declare these as their slots,
    # but ignore them in their __getstate__ implementation
    __expression_slots__ = __pickle_slots__

    __slots__ = ()

    def __init__(self, expr=None):
        self._expr = as_numeric(expr) if (expr is not None) else None

    def create_node_with_local_data(self, values):
        """
        Construct a simple expression after constructing the 
        contained expression.
   
        This class provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.
        """
        obj = SimpleExpression()
        obj.construct()
        obj.expr = values[0]
        return obj

    def __getstate__(self):
        state = super(_GeneralExpressionDataImpl, self).__getstate__()
        for i in _GeneralExpressionDataImpl.__expression_slots__:
            state[i] = getattr(self, i)
        return state

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        return self._expr
    @expr.setter
    def expr(self, expr):
        self.set_value(expr)

    # for backwards compatibility reasons
    @property
    @deprecated("The .value property getter on _GeneralExpressionDataImpl "
                "is deprecated. Use the .expr property getter instead",
                version='4.2.10782')
    def value(self):
        return self._expr

    @value.setter
    @deprecated("The .value property setter on _GeneralExpressionDataImpl "
                "is deprecated. Use the set_value(expr) method instead",
                version='4.2.10782')
    def value(self, expr):
        self.set_value(expr)

    def set_value(self, expr):
        """Set the expression on this expression."""
        self._expr = as_numeric(expr) if (expr is not None) else None

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        # The underlying expression can always be changed
        # so this should never evaluate as constant
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        return self._expr.is_fixed()

class _GeneralExpressionData(_GeneralExpressionDataImpl,
                             ComponentData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr        The expression owned by this data.

    Private class attributes:
        _component  The expression component.
    """

    __slots__ = _GeneralExpressionDataImpl.__expression_slots__

    def __init__(self, expr=None, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        # Inlining ComponentData.__init__
        self._component = weakref_ref(component) if (component is not None) \
                          else None


@ModelComponentFactory.register("Named expressions that can be used in other expressions.")
class Expression(IndexedComponent):
    """
    A shared expression container, which may be defined over a index.

    Constructor Arguments:
        initialize  A Pyomo expression or dictionary of expressions
                        used to initialize this object.
        expr        A synonym for initialize.
        rule        A rule function used to initialize this object.
    """

    _ComponentDataClass = _GeneralExpressionData
    class Skip(object): pass

    def __new__(cls, *args, **kwds):
        if cls is not Expression:
            return super(Expression, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return super(Expression, cls).__new__(AbstractSimpleExpression)
        else:
            return super(Expression, cls).__new__(IndexedExpression)

    def __init__(self, *args, **kwds):
        _init = tuple( _arg for _arg in (
            kwds.pop('initialize', None),
            kwds.pop('rule', None),
            kwds.pop('expr', None) ) if _arg is not None )
        if len(_init) > 1:
            raise ValueError(
                "Duplicate initialization: Expression() only accepts "
                "one of 'initialize=', 'rule=', and 'expr='")
        elif _init:
            _init = _init[0]
        else:
            _init = None

        kwds.setdefault('ctype', Expression)
        IndexedComponent.__init__(self, *args, **kwds)

        self._init_expr = Initializer(_init)

    def _pprint(self):
        return (
            [('Size', len(self)),
             ('Index', None if (not self.is_indexed())
                  else self._index)
             ],
            self.iteritems(),
            ("Expression",),
            lambda k,v: \
               ["Undefined" if v.expr is None else v.expr]
            )

    def display(self, prefix="", ostream=None):
        """TODO"""
        if ostream is None:
            ostream = sys.stdout
        tab="    "
        ostream.write(prefix+self.local_name+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix+tab,
            ((k,v) for k,v in iteritems(self._data)),
            ( "Value", ),
            lambda k, v: \
               ["Undefined" if v.expr is None else v()])

    #
    # A utility to extract all index-value pairs defining this
    # expression, returned as a dictionary. useful in many contexts,
    # in which key iteration and repeated __getitem__ calls are too
    # expensive to extract the contents of an expression.
    #
    def extract_values(self):
        return {key:expression_data.expr
                for key, expression_data in iteritems(self)}

    #
    # takes as input a (index, value) dictionary for updating this
    # Expression.  if check=True, then both the index and value are
    # checked through the __getitem__ method of this class.
    #
    def store_values(self, new_values):
        for index, new_value in iteritems(new_values):
            self[index].set_value(new_value)

    #
    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    #
    def _getitem_when_not_present(self, index):
        # TBD: Is this desired behavior?  I can see implicitly setting
        # an Expression if it was not originally defined, but I am less
        # convinced that implicitly creating an Expression (like what
        # works with a Var) makes sense.  [JDS 25 Nov 17]
        if self._init_expr is None:
            _expr = None
        else:
            _block = self.parent_block()
            _expr = self._init_expr(_block, index)
        if _expr is Expression.Skip:
            if index is None and not self.is_indexed():
                raise ValueError(
                    "Expression.Skip can not be assigned to an "
                    "Expression that is not indexed: %s" % (self.name,))
            return

        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        obj.set_value(_expr)
        return obj

    def construct(self, data=None):
        """ Apply the rule to construct values in this set """
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Constructing Expression, name=%s, from data=%s"
                % (self.name, str(data)))
        self._constructed = True

        if data is not None:
            # Data supplied to construct() should override data provided
            # to the constructor
            tmp_init, self._init_expr = self._init_expr, Initializer(data)
        try:
            if self._init_expr is None:
                if not self.is_indexed():
                    # This ensures backwards compatibility by causing
                    # all scalar Expressions to be initialized (and
                    # potentially empty) after construct().
                    self._getitem_when_not_present(None)
            elif self._init_expr.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in self._init_expr.indices():
                    IndexedComponent.__getitem__(self, index)
            else:
                # Bypass the index validation and create the member directly
                for index in self.index_set():
                    self._getitem_when_not_present(index)
        finally:
            # Restore the original initializer (if overridden by data argument)
            if data is not None:
                self._init_expr = tmp_init
        timer.report()


class SimpleExpression(_GeneralExpressionData, Expression):

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, expr=None, component=self)
        Expression.__init__(self, *args, **kwds)

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be deprecated/removed?)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise KeyError(
                "SimpleExpression object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        if expr is Expression.Skip:
            raise ValueError(
                "Expression.Skip can not be assigned "
                "to an Expression that is not indexed: %s"
                % (self.name))
        self.set_value(expr)
        return self


@disable_methods(_EXPRESSION_API | {'add'})
class AbstractSimpleExpression(SimpleExpression):
    pass


class IndexedExpression(Expression):

    #
    # Leaving this method for backward compatibility reasons
    # (probably should be deprecated/removed?)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if expr is Expression.Skip:
            return None
        self[index] = expr
        return self[index]

