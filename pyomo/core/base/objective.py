#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('Objective',
           'simple_objective_rule',
           '_ObjectiveData',
           'minimize',
           'maximize',
           'simple_objectivelist_rule',
           'ObjectiveList')

import sys
import logging
from weakref import ref as weakref_ref

from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecated, RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.template_expr import (
    templatize_rule, resolve_template, TemplateExpressionError,
)
from pyomo.core.base.component import (
    ActiveComponentData, ModelComponentFactory,
)
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent, UnindexedComponent_set, rule_wrapper,
    _get_indexed_component_data_name,
)
from pyomo.core.base.expression import (_ExpressionData,
                                        _GeneralExpressionDataImpl)
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set
from pyomo.core.base.initializer import (
    Initializer, IndexedCallInitializer, CountedCallInitializer,
)
from pyomo.core.base import minimize, maximize

logger = logging.getLogger('pyomo.core')

ATTEMPT_TEMPLITIZATION = True

_rule_returned_none_error = """Objective '%s': rule returned None.

Objective rules must return either a valid expression, numeric value, or
Objective.Skip.  The most common cause of this error is forgetting to
include the "return" statement at the end of your rule.
"""

def simple_objective_rule(rule):
    """
    This is a decorator that translates None into Objective.Skip.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    @simple_objective_rule
    def O_rule(model, i, j):
        ...

    model.o = Objective(rule=simple_objective_rule(...))
    """
    return rule_wrapper(rule, {None: Objective.Skip})

def simple_objectivelist_rule(rule):
    """
    This is a decorator that translates None into ObjectiveList.End.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    @simple_objectivelist_rule
    def O_rule(model, i, j):
        ...

    model.o = ObjectiveList(expr=simple_objectivelist_rule(...))
    """
    return rule_wrapper(rule, {None: ObjectiveList.End})

#
# This class is a pure interface
#

class _ObjectiveData(_ExpressionData):
    """
    This class defines the data for a single objective.

    Public class attributes:
        expr            The Pyomo expression for this objective
        sense           The direction for this objective.
    """

    __slots__ = ()

    #
    # Interface
    #

    def is_minimizing(self):
        """Return True if this is a minimization objective."""
        return self.sense == minimize

    #
    # Abstract Interface
    #

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        raise NotImplementedError

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        raise NotImplementedError

class _GeneralObjectiveData(_GeneralExpressionDataImpl,
                            _ObjectiveData,
                            ActiveComponentData):
    """
    This class defines the data for a single objective.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr            The Pyomo expression stored in this objective.
        sense           The direction for this objective.
        component       The Objective object that owns this data.

    Public class attributes:
        expr            The Pyomo expression for this objective
        active          A boolean that is true if this objective is active
                            in the model.
        sense           The direction for this objective.

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __pickle_slots__ = ("_sense",)
    __slots__ = __pickle_slots__ + _GeneralExpressionDataImpl.__pickle_slots__

    def __init__(self, expr=None, sense=minimize, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        # Inlining ActiveComponentData.__init__
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True
        self._sense = sense

        if (self._sense != minimize) and \
           (self._sense != maximize):
            raise ValueError("Objective sense must be set to one of "
                             "'minimize' (%s) or 'maximize' (%s). Invalid "
                             "value: %s'" % (minimize, maximize, sense))

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = _GeneralExpressionDataImpl.__getstate__(self)
        for i in _GeneralObjectiveData.__pickle_slots__:
            state[i] = getattr(self,i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    #       we don't need to implement a specialized __setstate__
    #       method.

    def set_value(self, expr):
        if expr is None:
            raise ValueError(_rule_returned_none_error % (self.name,))
        return super().set_value(expr)

    #
    # Abstract Interface
    #

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        return self._sense
    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if sense in {minimize, maximize}:
            self._sense = sense
        else:
            raise ValueError("Objective sense must be set to one of "
                             "'minimize' (%s) or 'maximize' (%s). Invalid "
                             "value: %s'" % (minimize, maximize, sense))


class _TemplatedObjectiveCommonInfo(object):
    def __init__(self, template_expr, template_indices, linear_repn_template):
        self.template_expr = template_expr
        self.template_indices = template_indices
        self.linear_repn_template = linear_repn_template


class _TemplatedObjectiveMixin(object):
    __slots__ = ()

    def _resolve_template(self, exp):
        for idx, val in zip(self._expr[0].template_indices,
                            self._expr[1]):
            idx.set_value(val)
        return resolve_template(exp)

    @property
    def expr(self):
        return self._resolve_template(self._expr[0].template_expr)

    def canonical_form(self):
        if self._expr[0].linear_repn_template is None:
            self._build_linear_repn_template()
        for idx, val in zip(self._expr[0].template_indices,
                            self._expr[1]):
            idx.set_value(val)
        return self._expr[0].linear_repn_template.standardRepn()

    def _build_linear_repn_template(self):
        from pyomo.repn.linear_template_repn import (
            LinearTemplateRepnVisitor as visitor
        )
        self._expr[0].linear_repn_template = visitor(
            self._expr[0].template_indices).walk_expression(
                self._expr[0].template_expr)


class _TemplatedObjectiveData(_TemplatedObjectiveMixin,
                               _GeneralObjectiveData):
    __slots__ = ()

    def __init__(self):
        pass

    def initialize_template(self, component, template, indices, index):
        if index.__class__ is not tuple:
            index = (index,)
        super().__init__(component=component, expr=template)
        self._expr = (
            _TemplatedObjectiveCommonInfo(self._expr, indices, None),
            index,
        )
        return self

    def duplicate_template(self, index):
        if index.__class__ is not tuple:
            index = (index,)
        ans = _TemplatedObjectiveData()
        ans._component = self._component
        ans._active = self._active
        ans._expr = (self._expr[0], index)
        ans._sense = self._sense
        return ans

    def set_value(self, expr):
        # If someone is setting the expression, then this index is no
        # longer defined by the template: revert it back to a
        # _GeneralObjectiveData
        if self._expr is not None:
            self.__class__ = _GeneralObjectiveData
            self.set_value(expr)
        else:
            super().set_value(expr)

    def getname(self, fully_qualified=False, name_buffer=None,
                relative_to=None):
        try:
            return super().getname(fully_qualified, name_buffer, relative_to)
        except RuntimeError:
            pass
        return self.parent_component().getname(
            fully_qualified, name_buffer, relative_to) + "[{template}]"


@ModelComponentFactory.register("Expressions that are minimized or maximized.")
class Objective(ActiveIndexedComponent):
    """
    This modeling component defines an objective expression.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr
            A Pyomo expression for this objective
        rule
            A function that is used to construct objective expressions
        sense
            Indicate whether minimizing (the default) or maximizing
        doc
            A text string describing this component
        name
            A name for this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to construct
            a model instance
        rule
            The rule used to initialize the objective(s)
        sense
            The objective sense

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index
            The set of valid indices
        _implicit_subsets
            A tuple of set objects that represents the index set
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """

    _ComponentDataClass = _GeneralObjectiveData
    NoObjective = ActiveIndexedComponent.Skip

    def __new__(cls, *args, **kwds):
        if cls != Objective:
            return super(Objective, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return ScalarObjective.__new__(ScalarObjective)
        else:
            return IndexedObjective.__new__(IndexedObjective)

    def __init__(self, *args, **kwargs):
        _sense = kwargs.pop('sense', minimize)
        _init = tuple( _arg for _arg in (
            kwargs.pop('rule', None), kwargs.pop('expr', None)
        ) if _arg is not None )
        if len(_init) == 1:
            _init = _init[0]
        elif not _init:
            _init = None
        else:
            raise ValueError("Duplicate initialization: Objective() only "
                             "accepts one of 'rule=' and 'expr='")

        kwargs.setdefault('ctype', Objective)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

        self.rule = Initializer(_init)
        self._init_sense = Initializer(_sense)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing objective %s" % (self.name))

        rule = self.rule
        try:
            # We do not (currently) accept data for constructing Objectives
            index = None
            assert data is None

            if rule is None:
                # If there is no rule, then we are immediately done.
                return

            if rule.constant() and self.is_indexed():
                raise IndexError(
                    "Objective '%s': Cannot initialize multiple indices "
                    "of an objective with a single expression" %
                    (self.name,) )

            block = self.parent_block()
            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    ans = self.__setitem__(index, rule(block, index))
                    if ans is not None:
                        self[index].set_sense(self._init_sense(block, index))
                return
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                return
            elif ATTEMPT_TEMPLITIZATION:
                template = None
                try:
                    template, indices, count = templatize_rule(
                        block, rule, self.index_set())
                except:
                    # If anything fails, then (silently - for the
                    # moment) fall back on the normal objective
                    # construction procedure
                    self._data.clear()
                if template is not None:
                    if not count:
                        # constant expressions and scalar functions that
                        # do not loop over Pyomo Sets
                        for index in self.index_set():
                            ans = self._setitem_when_not_present(
                                index, template)
                            if ans is not None:
                                ans.set_sense(self._init_sense(block, index))
                    elif self.is_indexed():
                        index_iter = iter(self.index_set())
                        index = next(index_iter)
                        ref = _TemplatedObjectiveData().initialize_template(
                            self, template, indices, index)
                        ref.set_sense(self._init_sense(block, index))
                        self._data[index] = ref
                        for index in index_iter:
                            tmp = ref.duplicate_template(index)
                            self._data[index] = tmp
                            tmp.set_sense(self._init_sense(block, index))
                    else:
                        self._setitem_when_not_present(None, template)
                        self.__class__ = ScalarTemplatedObjective
                        self._expr = (
                            _TemplatedObjectiveCommonInfo(
                                self._expr, indices, None),
                            ()
                        )
                        self.set_sense(self._init_sense(block, index))
                    return
            # Bypass the index validation and create the member directly
            for index in self.index_set():
                ans = self._setitem_when_not_present(
                    index, rule(block, index))
                if ans is not None:
                    ans.set_sense(self._init_sense(block, index))
        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when generating expression for "
                "Objective %s with index %s:\n%s: %s"
                % (self.name,
                   str(index),
                   type(err).__name__,
                   err))
            raise
        finally:
            timer.report()

    def _getitem_when_not_present(self, index):
        if self.rule is None:
            raise KeyError(index)
        obj = self._setitem_when_not_present(
            index, self.rule(self.parent_block(), index))
        if obj is None:
            raise KeyError(index)
        else:
            obj.set_sense(self._init_sense(block, index))
        return obj

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active)
             ],
            self._data.items(),
            ( "Active","Sense","Expression"),
            lambda k, v: [ v.active,
                           ("minimize" if (v.sense == minimize) else "maximize"),
                           v.expr
                           ]
            )

    def display(self, prefix="", ostream=None):
        """Provide a verbose display of this object"""
        if not self.active:
            return
        tab = "    "
        if ostream is None:
            ostream = sys.stdout
        ostream.write(prefix+self.local_name+" : ")
        ostream.write(", ".join("%s=%s" % (k,v) for k,v in [
                    ("Size", len(self)),
                    ("Index", self._index if self.is_indexed() else None),
                    ("Active", self.active),
                    ] ))

        ostream.write("\n")
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in self._data.items() if v.active),
                        ( "Active","Value" ),
                        lambda k, v: [ v.active, value(v), ] )


class ScalarObjective(_GeneralObjectiveData, Objective):
    """
    ScalarObjective is the implementation representing a single,
    non-indexed objective.
    """

    def __init__(self, *args, **kwd):
        _GeneralObjectiveData.__init__(self, expr=None, component=self)
        Objective.__init__(self, *args, **kwd)

    #
    # Since this class derives from Component and
    # Component.__getstate__ just packs up the entire __dict__ into
    # the state dict, we do not need to define the __getstate__ or
    # __setstate__ methods.  We just defer to the super() get/set
    # state.  Since all of our get/set state methods rely on super()
    # to traverse the MRO, this will automatically pick up both the
    # Component and Data base classes.
    #

    #
    # Override abstract interface methods to first check for
    # construction
    #

    @property
    def expr(self):
        """Access the expression of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the expression of ScalarObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralObjectiveData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @expr.setter
    def expr(self, expr):
        """Set the expression of this objective."""
        self.set_value(expr)

    # for backwards compatibility reasons
    @property
    @deprecated("The .value property getter on ScalarObjective is deprecated. "
                "Use the .expr property getter instead", version='4.3.11323')
    def value(self):
        return self.expr

    @value.setter
    @deprecated("The .value property setter on ScalarObjective is deprecated. "
                "Use the set_value(expr) method instead", version='4.3.11323')
    def value(self, expr):
        self.set_value(expr)

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the sense of ScalarObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralObjectiveData.sense.fget(self)
        raise ValueError(
            "Accessing the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    #
    # Singleton objectives are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Objective.Skip are managed. But after that they will behave
    # like _ObjectiveData objects where set_value does not handle
    # Objective.Skip but expects a valid expression or None
    #

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression of this objective."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of objective '%s' "
                "before the Objective has been constructed (there "
                "is currently no object to set)."
                % (self.name))
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralObjectiveData.set_sense(self, sense)
        raise ValueError(
            "Setting the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise ValueError(
                "ScalarObjective object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        self.set_value(expr)
        return self


class SimpleObjective(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarObjective
    __renamed__version__ = '6.0'


class ScalarTemplatedObjective(_TemplatedObjectiveMixin, ScalarObjective):

    def set_value(self, expr):
        # If someone is setting the expression, then this index is no
        # longer defined by the template: revert it back to a
        # _GeneralObjectiveData
        if self._expr is not None:
            self.__class__ = ScalarObjective
        # Explicitly call the base class method, as changing the
        # __class__ breaks super()
        ScalarObjective.set_value(self, expr)


class IndexedObjective(Objective):

    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add an objective with a given index."""
        return self.__setitem__(index, expr)


@ModelComponentFactory.register("A list of objective expressions.")
class ObjectiveList(IndexedObjective):
    """
    An objective component that represents a list of objectives.
    Objectives can be indexed by their index, but when they are added
    an index value is not specified.
    """

    class End(object): pass

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError(
                "ObjectiveList does not accept the 'expr' keyword")
        _rule = kwargs.pop('rule', None)

        args = (Set(dimen=1),)
        super().__init__(*args, **kwargs)

        self.rule = Initializer(_rule, allow_generators=True)
        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self.rule is not None and type(self.rule) is IndexedCallInitializer:
            self.rule = CountedCallInitializer(self, self.rule)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if self._constructed:
            return
        self._constructed=True

        if is_debug_set(logger):
            logger.debug("Constructing objective list %s"
                         % (self.name))

        self.index_set().construct()

        if self.rule is not None:
            _rule = self.rule(self.parent_block(), ())
            for cc in iter(_rule):
                if cc is ObjectiveList.End:
                    break
                if cc is Objective.Skip:
                    continue
                self.add(cc, sense=self._init_sense)

    def add(self, expr, sense=minimize):
        """Add an objective to the list."""
        next_idx = len(self._index) + 1
        self._index.add(next_idx)
        ans = self.__setitem__(next_idx, expr)
        if ans is not None:
            if sense not in {minimize, maximize}:
                sense = sense(self.parent_block(), next_idx)
            ans.set_sense(sense)
        return ans

