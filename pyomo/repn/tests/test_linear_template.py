#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.repn.linear_template import LinearTemplateRepnVisitor
from pyomo.repn.util import TemplateVarRecorder

import pyomo.core.base.constraint as constraint
import pyomo.core.base.objective as objective

from pyomo.environ import ConcreteModel, Var, Param


class VisitorConfig(dict):
    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}
        self.sorter = None
        self.var_recorder = TemplateVarRecorder(self.var_map, self.sorter)
        super().__init__(
            subexpression_cache=self.subexpr, var_recorder=self.var_recorder
        )

    def order_quadratic(self, quad):
        return {
            (
                (vid1, vid2)
                if self.var_order[vid1] <= self.var_order[vid2]
                else (vid2, vid1)
            ): val
            for (vid1, vid2), val in quad.items()
        }


class TestLinearTemplate(unittest.TestCase):
    templatize_stack = []

    def push_templatization(self, mode):
        self.templatize_stack.append(
            (constraint.TEMPLATIZE_CONSTRAINTS, objective.TEMPLATIZE_OBJECTIVES)
        )
        constraint.TEMPLATIZE_CONSTRAINTS = mode
        objective.TEMPLATIZE_OBJECTIVES = mode

    def pop_templatization(self):
        constraint.TEMPLATIZE_CONSTRAINTS, objective.TEMPLATIZE_OBJECTIVES = (
            self.templatize_stack.pop()
        )

    def setUp(self):
        super().setUp()
        self.push_templatization(True)

    def tearDown(self):
        self.pop_templatization()

    def test_repn_to_string(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        m.p = Param(range(3), initialize={0: 5}, mutable=True, default=1)

        e = m.p[0] * m.x[0] + m.x[1] + 10

        cfg = VisitorConfig()
        repn = LinearTemplateRepnVisitor(**cfg).walk_expression(e)
        self.assertEqual(
            str(repn),
            "LinearTemplateRepn(mult=1, const=10, linear={0: 5, 1: 1}, "
            "linear_sum=[], nonlinear=None)",
        )

        @m.Objective()
        def obj(m):
            return sum(m.p[i] * m.x[i] for i in range(3))

        e = m.obj.template_expr()[0]
        cfg = VisitorConfig()
        visitor = LinearTemplateRepnVisitor(**cfg)
        repn = visitor.walk_expression(e)
        self.assertEqual(
            str(repn),
            "LinearTemplateRepn(mult=1, const=0, linear={}, "
            "linear_sum=[LinearTemplateRepn(mult=p[_1], const=0, "
            "linear={%s: 1}, linear_sum=[], nonlinear=None), "
            "[(_1)], [(0, 1, 2)]], nonlinear=None)" % (list(visitor.expr_cache)[-1],),
        )
