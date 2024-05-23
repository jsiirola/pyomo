#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import MouseTrap
from pyomo.common.dependencies import scipy
from pyomo.core import (
    ConcreteModel,
    Block,
    Var,
    Constraint,
    Objective,
    TransformationFactory,
    NonNegativeReals,
    NonPositiveReals,
    maximize,
    minimize,
    Reals,
)
from pyomo.opt import WriterFactory


# ESJ: TODO: copied from FME basically, should centralize.
def var_list(x):
    if x.ctype is Var:
        if not x.is_indexed():
            return ComponentSet([x])
        ans = ComponentSet()
        for j in x.index_set():
            ans.add(x[j])
        return ans
    elif hasattr(x, '__iter__'):
        ans = ComponentSet()
        for i in x:
            ans.update(vars_to_eliminate_list(i))
        return ans
    else:
        raise ValueError("Expected Var or list of Vars.\n\tReceived %s" % type(x))


class _LPDualData(AutoSlots.Mixin):
    __slots__ = ('primal_var', 'dual_var', 'primal_constraint', 'dual_constraint')
    def __init__(self):
        self.primal_var = {}
        self.dual_var = {}
        self.primal_constraint = ComponentMap()
        self.dual_constraint = ComponentMap()


Block.register_private_data_initializer(_LPDualData)


@TransformationFactory.register(
    'core.lp_dual', 'Generate the linear programming dual of the given model'
)
class LinearProgrammingDual(object):
    CONFIG = ConfigDict("core.lp_dual")
    CONFIG.declare(
        'parameterize_wrt',
        ConfigValue(
            default=None,
            domain=var_list,
            description="Vars to treat as data for the purposes of taking the dual",
            doc="""
            Optional list of Vars to be treated as data while taking the LP dual.

            For example, if this is the dual of the inner problem in a multilevel
            optimization problem, then the outer problem's Vars would be specified
            in this list since they are not variables from the perspective of the 
            inner problem.
            """,
        ),
    )

    def apply_to(self, model, **options):
        raise MouseTrap(
            "The 'core.lp_dual' transformation does not currently implement "
            "apply_to since it is a bit ambiguous what it means to take a dual "
            "in place. Please use 'create_using' and do what you wish with the "
            "returned model."
        )

    def create_using(self, model, ostream=None, **kwds):
        """Take linear programming dual of a model

        Returns
        -------
        ConcreteModel containing linear programming dual

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to take the dual of

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        if config.parameterize_wrt is None:
            return self._take_dual(model)

        return self._take_parameterized_dual(model, config.parameterize_wrt)

    def _take_dual(self, model):
        std_form = WriterFactory('compile_standard_form').write(
            model, mixed_form=True, set_sense=None
        )
        if len(std_form.objectives) != 1:
            raise ValueError(
                "Model '%s' has n o objective or multiple active objectives. Cannot "
                "take dual with more than one objective!" % model.name
            )
        primal_sense = std_form.objectives[0].sense

        dual = ConcreteModel(name="%s dual" % model.name)
        A_transpose = scipy.sparse.csc_matrix.transpose(std_form.A)
        rows = range(A_transpose.shape[0])
        cols = range(A_transpose.shape[1])
        dual.x = Var(cols, domain=NonNegativeReals)
        trans_info = model.private_data()
        for j, (primal_cons, ineq) in enumerate(std_form.rows):
            if primal_sense is minimize and ineq == 1:
                dual.x[j].domain = NonPositiveReals
            elif primal_sense is maximize and ineq == -1:
                dual.x[j].domain = NonPositiveReals
            if ineq == 0:
                # equality
                dual.x[j].domain = Reals
            trans_info.primal_constraint[dual.x[j]] = primal_cons
            trans_info.dual_var[primal_cons] = dual.x[j]
            
        dual.constraints = Constraint(rows)
        for i, primal in enumerate(std_form.columns):
            if primal_sense is minimize:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = (
                        sum(A_transpose[i, j] * dual.x[j] for j in cols)
                        >= std_form.c[0, i]
                    )
                elif primal.domain is NonPositiveReals:
                    dual.constraints[i] = (
                        sum(A_transpose[i, j] * dual.x[j] for j in cols)
                        <= std_form.c[0, i]
                    )
            else:
                if primal.domain is NonNegativeReals:
                    dual.constraints[i] = (
                        sum(A_transpose[i, j] * dual.x[j] for j in cols)
                        <= std_form.c[0, i]
                    )
                elif primal.domain is NonPositiveReals:
                    dual.constraints[i] = (
                        sum(A_transpose[i, j] * dual.x[j] for j in cols)
                        >= std_form.c[0, i]
                    )
            if primal.domain is Reals:
                dual.constraints[i] = (
                    sum(A_transpose[i, j] * dual.x[j] for j in cols) == std_form.c[0, i]
                )
            trans_info.dual_constraint[primal] = dual.constraints[i]
            trans_info.primal_var[dual.constraints[i]] = primal

        dual.obj = Objective(
            expr=sum(std_form.rhs[j] * dual.x[j] for j in cols), sense=-primal_sense
        )

        return dual

    def _take_parameterized_dual(self, model, wrt):
        pass

    def get_primal_constraint(self, model, dual_var):
        primal_constraint = model.private_data().primal_constraint
        if dual_var in primal_constraint:
            return primal_constraint[dual_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a dual variable on model '%s'"
                % (dual_var.name, model.name)
            )

    def get_dual_constraint(self, model, primal_var):
        dual_constraint = model.private_data().dual_constraint
        if primal_var in dual_constraint:
            return dual_constraint[primal_var]
        else:
            raise ValueError(
                "It does not appear that Var '%s' is a primal variable from model '%s'"
                % (primal_var.name, model.name)
            )

    def get_primal_var(self, model, dual_constraint):
        primal_var = model.private_data().primal_var
        if dual_constraint in primal_var:
            return primal_var[dual_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a dual constraint on "
                "model '%s'" % (dual_constraint.name, model.name))

    def get_dual_var(self, model, primal_constraint):
        dual_var = model.private_data().dual_var
        if primal_constraint in dual_var:
            return dual_var[primal_constraint]
        else:
            raise ValueError(
                "It does not appear that Constraint '%s' is a primal constraint from "
                "model '%s'" % (primal_constraint.name, model.name))
