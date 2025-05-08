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
import pyomo.contrib.solver.solvers.gurobi_direct
import pyomo.core.base.constraint
import pyomo.core.base.objective
import pyomo.environ
import logging
LOGGER = logging.getLogger(__name__)

opt = pyomo.contrib.solver.solvers.gurobi_direct.GurobiDirect()
if not opt.available():
    raise unittest.SkipTest
import gurobipy

SOLVER_NAME="gurobi_direct_v2"

def get_solver(solver_name=SOLVER_NAME):
    return pyomo.environ.SolverFactory(solver_name)

def create_model(
    *,
    templatize_constraints=True,
    templatize_objectives=True,
    solver_name=SOLVER_NAME,
    debug_file=None):

    pyomo.core.base.constraint.TEMPLATIZE_CONSTRAINTS = templatize_constraints
    pyomo.core.base.objective.TEMPLATIZE_OBJECTIVES = templatize_objectives
    if debug_file:
        pyomo.contrib.solver.solvers.gurobi_direct.DEBUG_LP_FILE=debug_file

    LOGGER.info(f"TEMPLATIZE_CONSTRAINTS = {templatize_constraints}")
    LOGGER.info(f"TEMPLATIZE_OBJECTIVES = {templatize_objectives}")
    LOGGER.info(f"DEBUG_LP_FILE = {pyomo.contrib.solver.solvers.gurobi_direct.DEBUG_LP_FILE}")

    return pyomo.environ.ConcreteModel(), get_solver(solver_name)


class TestGurobiQuadraticTemplate(unittest.TestCase):

    def test_verify_using_templatized_constraints_and_objectives(self):
        self.assertTrue(pyomo.core.base.constraint.TEMPLATIZE_CONSTRAINTS)
        self.assertTrue(pyomo.core.base.objective.TEMPLATIZE_OBJECTIVES)

    def test_quadratic_objective_decorator(self): pass
    def test_quadratic_objective_nondecorator(self): pass
    def test_linear_objective(self): pass


    def test_linear_constraint(self): pass # make sure linear still works

    def test_quadratic_constraint_nondecorator(self):
        model, opt = create_model()
        rng=list(range(2))
        model.x = pyomo.environ.Var(rng, bounds=(0,6))
        model.obj = pyomo.environ.Objective(expr=model.x[0] + model.x[1], sense=pyomo.environ.maximize)
        model.c1 = pyomo.environ.Constraint(expr=model.x[1]<= (model.x[0]-2) * (model.x[0]-5))
        opt.solve(model, tee=False)
        self.assertAlmostEqual(model.x[0].value, 6.0, delta=0.1)
        self.assertAlmostEqual(model.x[1].value, 4.0, delta=0.1)

    def test_quadratic_constraint_binomial_expansion(self):
        """
        FIX: This case of binomial expansion does not appear to work.
        """
        self.skipTest("This case does not work")
        model, opt = create_model()
        rng=list(range(2))
        model.x = pyomo.environ.Var(rng, bounds=(0,6))
        model.obj = pyomo.environ.Objective(expr=model.x[0] + model.x[1], sense=pyomo.environ.maximize)

        @model.Constraint([tuple(rng)])
        def c1(model, i,j):
            return model.x[j] <= (model.x[i]-2) * (model.x[i]-5)
        opt.solve(model, tee=False)

        self.assertAlmostEqual(model.x[0].value, 6.0, delta=0.1)
        self.assertAlmostEqual(model.x[1].value, 4.0, delta=0.1)

    def test_quadratic_constraint_decorator_1(self):
        model, opt = create_model()
        rng=list(range(2))
        model.x = pyomo.environ.Var(rng, bounds=(0,6))
        model.obj = pyomo.environ.Objective(expr=model.x[0] + model.x[1], sense=pyomo.environ.maximize)
        # model.c1 = pyomo.environ.Constraint(expr=model.x[1]<= (model.x[0]-2) * (model.x[0]-5))
        @model.Constraint([tuple(rng)])
        def c1(model, i,j):
            return model.x[j] - model.x[i]**2 + 7*model.x[i] <= 10
        opt.solve(model, tee=False)
        self.assertAlmostEqual(model.x[0].value, 6.0, delta=0.1)
        self.assertAlmostEqual(model.x[1].value, 4.0, delta=0.1)



    def test_quadratic_constraint_decorator_2(self):pass