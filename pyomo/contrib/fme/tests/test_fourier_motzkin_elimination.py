#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
from pyomo.core import (Var, Constraint, Param, ConcreteModel, NonNegativeReals,
                        Binary, value, Block)
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.current import log
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.kernel.component_set import ComponentSet

class TestFourierMotzkinElimination(unittest.TestCase):
    @staticmethod
    def makeModel():
        """
        This is a single-level reformulation of a bilevel model.
        We project out the dual variables to recover the reformulation in 
        the original space.
        """
        m = ConcreteModel()
        m.x = Var(bounds=(0,2))
        m.y = Var(domain=NonNegativeReals)
        m.lamb = Var([1, 2], domain=NonNegativeReals)
        m.M = Param([1, 2], mutable=True, default=100)
        m.u = Var([1, 2], domain=Binary)

        m.primal1 = Constraint(expr=m.x - 0.01*m.y <= 1)
        m.dual1 = Constraint(expr=1 - m.lamb[1] - 0.01*m.lamb[2] == 0)

        @m.Constraint([1, 2])
        def bound_lambdas(m, i):
            return m.lamb[i] <= m.u[i]*m.M[i]

        m.bound_y = Constraint(expr=m.y <= 1000*(1 - m.u[1]))
        m.dual2 = Constraint(expr=-m.x + 0.01*m.y + 1 <= (1 - m.u[2])*1000)

        return m

    def test_no_vars_specified(self):
        m = self.makeModel()
        self.assertRaisesRegexp(
            RuntimeError,
            "The Fourier-Motzkin Elimination transformation "
            "requires the argument vars_to_eliminate, a "
            "list of Vars to be projected out of the model.",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m)

    def check_projected_constraints(self, m):
        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        # x - 0.01y <= 1
        cons = constraints[1]
        self.assertEqual(value(cons.lower), -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.x)
        self.assertEqual(coefs[0], -1)
        self.assertIs(linear_vars[1], m.y)
        self.assertEqual(coefs[1], 0.01)

        # y <= 1000*(1 - u_1)
        cons = constraints[2]
        self.assertEqual(value(cons.lower), -1000)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[0], m.y)
        self.assertEqual(coefs[0], -1)
        self.assertIs(linear_vars[1], m.u[1])
        self.assertEqual(coefs[1], -1000)

        # -x + 0.01y + 1 <= 1000*(1 - u_2)
        cons = constraints[3]
        self.assertEqual(value(cons.lower), -999)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 3)
        self.assertIs(linear_vars[0], m.x)
        self.assertEqual(coefs[0], 1)
        self.assertIs(linear_vars[1], m.y)
        self.assertEqual(coefs[1], -0.01)
        self.assertIs(linear_vars[2], m.u[2])
        self.assertEqual(coefs[2], -1000)

        # u_2 + 100u_1 >= 1
        cons = constraints[6]
        self.assertEqual(value(cons.lower), 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        linear_vars = body.linear_vars
        coefs = body.linear_coefs
        self.assertEqual(len(linear_vars), 2)
        self.assertIs(linear_vars[1], m.u[2])
        self.assertEqual(coefs[1], 1)
        self.assertIs(linear_vars[0], m.u[1])
        self.assertEqual(coefs[0], 100)

    def test_transformed_constraints_indexed_var_arg(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb)

        # we get some trivial constraints too, but let's check that the ones
        # that should be there really are
        self.check_projected_constraints(m)

    def test_transformed_constraints_varData_list_arg(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb[1], m.lamb[2]])

        self.check_projected_constraints(m)

    def test_transformed_constraints_indexedVar_list(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = [m.lamb])

        self.check_projected_constraints(m)

    def test_original_constraints_deactivated(self):
        m = self.makeModel()
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to( 
            m,
            vars_to_eliminate = m.lamb)
        
        self.assertFalse(m.primal1.active)
        self.assertFalse(m.dual1.active)
        self.assertFalse(m.dual2.active)
        self.assertFalse(m.bound_lambdas[1].active)
        self.assertFalse(m.bound_lambdas[2].active)
        self.assertFalse(m.bound_y.active)

    def test_infeasible_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.cons1 = Constraint(expr=m.x >= 6)
        m.cons2 = Constraint(expr=m.x <= 2)

        self.assertRaisesRegexp(
            RuntimeError,
            "Fourier-Motzkin found the model is infeasible!",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_infeasible_model_no_var_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons1 = Constraint(expr=m.x >= 6)
        m.cons2 = Constraint(expr=m.x <= 2)

        self.assertRaisesRegexp(
            RuntimeError,
            "Fourier-Motzkin found the model is infeasible!",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)
        
    def test_nonlinear_error(self):
        m = ConcreteModel()
        m.x = Var()
        m.cons = Constraint(expr=m.x**2 >= 2)
        m.cons2 = Constraint(expr=m.x<= 10)

        self.assertRaisesRegexp(
            RuntimeError,
            "Variable x appears in a nonlinear "
            "constraint. The Fourier-Motzkin "
            "Elimination transformation can only "
            "be used to eliminate variables "
            "which only appear linearly.",
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_components_we_do_not_understand_error(self):
        m = self.makeModel()
        m.disj = Disjunction(expr=[m.x == 0, m.y >= 2])

        self.assertRaisesRegexp(
            RuntimeError,
            "Found active component %s of type %s. The "
            "Fourier-Motzkin Elimination transformation can only "
            "handle purely algebraic models. That is, only "
            "Sets, Params, Vars, Constraints, Expressions, Blocks, "
            "and Objectives may be active on the model." % (m.disj.name, 
                                                            m.disj.type()),
            TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to,
            m, 
            vars_to_eliminate=m.x)

    def test_combine_three_inequalities_and_flatten_blocks(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.b = Block()
        m.b.c = Constraint(expr=m.x >= 2)
        m.c = Constraint(expr=m.y <= m.x)
        m.b.b2 = Block()
        m.b.b2.c = Constraint(expr=m.y >= 4)
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(
            m, vars_to_eliminate=m.y)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        self.assertEqual(len(constraints), 2)
        cons = constraints[1]
        self.assertEqual(value(cons.lower), 2)
        self.assertIsNone(cons.upper)
        self.assertIs(cons.body, m.x)
        
        cons = constraints[2]
        self.assertEqual(value(cons.lower), 4)
        self.assertIsNone(cons.upper)
        self.assertIs(cons.body, m.x)

    def test_project_disaggregated_vars(self):
        """This is a little bit more of an integration test with GDP, 
        but also an example of why FME is 'useful.' We will give a GDP, 
        take chull relaxation, and then project out the disaggregated 
        variables."""

        m = ConcreteModel()
        m.p = Var([1, 2], bounds=(0, 10))
        m.time1 = Disjunction(expr=[m.p[1] >= 1, m.p[1] == 0])

        m.on = Disjunct()
        m.on.above_min = Constraint(expr=m.p[2] >= 1)
        m.on.ramping = Constraint(expr=m.p[2] - m.p[1] <= 3)
        m.on.on_before = Constraint(expr=m.p[1] >= 1)

        m.startup = Disjunct()
        m.startup.startup_limit = Constraint(expr=(1, m.p[2], 2))
        m.startup.off_before = Constraint(expr=m.p[1] == 0)

        m.off = Disjunct()
        m.off.off = Constraint(expr=m.p[2] == 0)
        m.time2 = Disjunction(expr=[m.on, m.startup, m.off])

        TransformationFactory('gdp.chull').apply_to(m)
        relaxationBlocks = m._pyomo_gdp_chull_relaxation.relaxedDisjuncts
        disaggregatedVars = ComponentSet([relaxationBlocks[0].component("p[1]"),
                                          relaxationBlocks[1].component("p[1]"),
                                          relaxationBlocks[2].component("p[1]"),
                                          relaxationBlocks[2].component("p[2]"),
                                          relaxationBlocks[3].component("p[1]"),
                                          relaxationBlocks[3].component("p[2]"),
                                          relaxationBlocks[4].component("p[1]"),
                                          relaxationBlocks[4].component("p[2]")])
        TransformationFactory('contrib.fourier_motzkin_elimination').apply_to(
            m, vars_to_eliminate=disaggregatedVars)

        constraints = m._pyomo_contrib_fme_transformation.projected_constraints
        # we of course get tremendous amounts of garbage, but we make sure that
        # what should be here is:

        # p[1] >= on.ind_var
        cons = constraints[22]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.p[1])
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)

        # p[1] <= 10*on.ind_var + 10*off.ind_var
        cons = constraints[20]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], 10)
        self.assertIs(body.linear_vars[1], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[1], 10)
        self.assertIs(body.linear_vars[2], m.p[1])
        self.assertEqual(body.linear_coefs[2], -1)

        # p[1] >= time1_disjuncts[0].ind_var
        cons = constraints[58]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[0], m.p[1])
        self.assertEqual(body.linear_coefs[0], 1)

        # p[1] <= 10*time1_disjuncts[0].ind_var
        cons = constraints[61]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[0], 10)
        self.assertIs(body.linear_vars[1], m.p[1])
        self.assertEqual(body.linear_coefs[1], -1)

        # p[2] - p[1] <= 3*on.ind_var + 2*startup.ind_var
        cons = constraints[56]
        self.assertEqual(value(cons.lower), 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 4)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[3], m.p[2])
        self.assertEqual(body.linear_coefs[3], -1)
        self.assertIs(body.linear_vars[0], m.p[1])
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[1], 3)
        self.assertIs(body.linear_vars[2], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[2], 2)

        # p[2] >= on.ind_var + startup.ind_var
        cons = constraints[38]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.p[2])
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[2], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[2], -1)

        # p[2] <= 10*on.ind_var + 2*startup.ind_var
        cons = constraints[32]
        self.assertEqual(cons.lower, 0)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], 10)
        self.assertIs(body.linear_vars[1], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[1], 2)
        self.assertIs(body.linear_vars[2], m.p[2])
        self.assertEqual(body.linear_coefs[2], -1)

        # 1 <= time1_disjuncts[0].ind_var + time_1.disjuncts[1].ind_var
        cons = constraints[1]
        self.assertEqual(cons.lower, 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[1].indicator_var)
        self.assertEqual(body.linear_coefs[1], 1)

        # 1 >= time1_disjuncts[0].ind_var + time_1.disjuncts[1].ind_var
        cons = constraints[2]
        self.assertEqual(cons.lower, -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 2)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.time1_disjuncts[0].indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.time1_disjuncts[1].indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)

        # 1 <= on.ind_var + startup.ind_var + off.ind_var
        cons = constraints[3]
        self.assertEqual(cons.lower, 1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], 1)
        self.assertIs(body.linear_vars[1], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[1], 1)
        self.assertIs(body.linear_vars[2], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[2], 1)

        # 1 >= on.ind_var + startup.ind_var + off.ind_var
        cons = constraints[4]
        self.assertEqual(cons.lower, -1)
        self.assertIsNone(cons.upper)
        body = generate_standard_repn(cons.body)
        self.assertEqual(body.constant, 0)
        self.assertEqual(len(body.linear_vars), 3)
        self.assertTrue(body.is_linear())
        self.assertIs(body.linear_vars[0], m.on.indicator_var)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[1], m.startup.indicator_var)
        self.assertEqual(body.linear_coefs[1], -1)
        self.assertIs(body.linear_vars[2], m.off.indicator_var)
        self.assertEqual(body.linear_coefs[2], -1)

    def test_model_with_unrelated_nonlinear_expressions(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], bounds=(0,3))
        m.y = Var()
        m.z = Var()

        @m.Constraint([1,2])
        def cons(m, i):
            return m.x[i] <= m.y**i

        m.cons2 = Constraint(expr=m.x[1] >= m.y)
        m.cons3 = Constraint(expr=m.x[2] >= m.z - 3)
        # This is vacuous, but I just want something that's not quadratic
        m.cons4 = Constraint(expr=m.x[3] <= log(m.y + 1))

        TransformationFactory('contrib.fourier_motzkin_elimination').\
            apply_to(m, vars_to_eliminate=m.x)
        constraints = m._pyomo_contrib_fme_transformation.projected_constraints

        # 0 <= y <= 3
        cons = constraints[6]
        self.assertEqual(cons.lower, 0)
        self.assertIs(cons.body, m.y)
        cons = constraints[5]
        self.assertEqual(cons.lower, -3)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertIs(body.linear_vars[0], m.y)
        self.assertEqual(body.linear_coefs[0], -1)

        # z <= y**2 + 3
        cons = constraints[4]
        self.assertEqual(cons.lower, -3)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_quadratic())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertIs(body.linear_vars[0], m.z)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertEqual(len(body.quadratic_vars), 1)
        self.assertEqual(body.quadratic_coefs[0], 1)
        self.assertIs(body.quadratic_vars[0][0], m.y)
        self.assertIs(body.quadratic_vars[0][1], m.y)

        # z <= 6
        cons = constraints[2]
        self.assertEqual(cons.lower, -6)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_linear())
        self.assertEqual(len(body.linear_vars), 1)
        self.assertEqual(body.linear_coefs[0], -1)
        self.assertIs(body.linear_vars[0], m.z)

        # 0 <= ln(y+ 1)
        cons = constraints[1]
        self.assertEqual(cons.lower, 0)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_nonlinear())
        self.assertFalse(body.is_quadratic())
        self.assertEqual(len(body.linear_vars), 0)
        self.assertEqual(body.nonlinear_expr.name, 'log')
        self.assertEqual(len(body.nonlinear_expr.args[0].args), 2)
        self.assertIs(body.nonlinear_expr.args[0].args[0], m.y)
        self.assertEqual(body.nonlinear_expr.args[0].args[1], 1)

        # 0 <= y**2
        cons = constraints[3]
        self.assertEqual(cons.lower, 0)
        body = generate_standard_repn(cons.body)
        self.assertTrue(body.is_quadratic())
        self.assertEqual(len(body.quadratic_vars), 1)
        self.assertEqual(body.quadratic_coefs[0], 1)
        self.assertIs(body.quadratic_vars[0][0], m.y)
        self.assertIs(body.quadratic_vars[0][1], m.y)

        # check constraints valid for a selection of points (this is nonconvex,
        # but anyway...)
        pts = [#(sqrt(3), 6), Not numerically stable enough for this test
               (1, 4), (3, 6), (3, 0), (0, 0), (2,6)]
        for pt in pts:
            m.y.fix(pt[0])
            m.z.fix(pt[1])
            for i in constraints:
                self.assertLessEqual(value(constraints[i].lower),
                                     value(constraints[i].body))
