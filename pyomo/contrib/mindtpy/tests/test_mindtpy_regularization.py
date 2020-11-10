# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
from math import fabs
import pyomo.core.base.symbolic
import pyutilib.th as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple
from pyomo.opt import TerminationCondition
from pyomo.contrib.mindtpy.tests.MINLP4_simple import SimpleMINLP4

required_solvers = ('ipopt', 'cplex')
# required_solvers = ('gams', 'gams')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 "Required subsolvers %s are not available"
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 "Symbolic differentiation is not available")
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_OA_8PP_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                init_strategy='rNLP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_init_max_binary_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(max_binary, L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                init_strategy='max_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_MINLP_simple_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving MINLP_simple problem with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    def test_OA_MINLP2_simple_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving MINLP2_simple problem with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_OA_MINLP3_simple_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving MINLP3_simple problem with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_OA_Proposal_level_L1(self):
        # A little difference from the proposal slides
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                init_strategy='initial_binary',
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_Proposal_with_int_cuts_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(no-good cuts, L1)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                integer_to_binary=True
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_ConstraintQualificationExample_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n Solving Constraint Qualification Example with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_ConstraintQualificationExample_integer_cut_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with Level-based Outer Approximation(no-good cuts, L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_OnlineDocExample_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_OnlineDocExample_L_infinity_norm_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm="L_infinity"
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_MINLP4_simple_level_L1(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP4()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L1)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L1",
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                init_strategy='initial_binary',
                                level_coef=0.4
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.obj.expr), -56.981, places=2)

    def test_OA_8PP_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                init_strategy='rNLP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_init_max_binary_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(max_binary, L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                init_strategy='max_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_MINLP_simple_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving MINLP_simple problem with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    def test_OA_MINLP2_simple_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving MINLP2_simple problem with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_OA_MINLP3_simple_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving MINLP3_simple problem with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_OA_Proposal_level_L2(self):
        # A little difference from the proposal slides
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                init_strategy='initial_binary',
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_Proposal_with_int_cuts_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(no-good cuts, L2)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                integer_to_binary=True
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_ConstraintQualificationExample_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n Solving Constraint Qualification Example with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_ConstraintQualificationExample_integer_cut_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with Level-based Outer Approximation(no-good cuts, L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_OnlineDocExample_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_OnlineDocExample_L_infinity_norm_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm="L_infinity"
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_MINLP4_simple_level_L2(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP4()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L2)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L2",
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                init_strategy='initial_binary',
                                level_coef=0.4
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.obj.expr), -56.981, places=2)

    def test_OA_8PP_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                init_strategy='rNLP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_init_max_binary_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem with Level-based Outer Approximation(max_binary, L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                init_strategy='max_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_MINLP_simple_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving MINLP_simple problem with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    def test_OA_MINLP2_simple_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving MINLP2_simple problem with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_OA_MINLP3_simple_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving MINLP3_simple problem with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_OA_Proposal_level_L_infinity(self):
        # A little difference from the proposal slides
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                init_strategy='initial_binary',
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_Proposal_with_int_cuts_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Level-based Outer Approximation(no-good cuts, L_infinity)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                integer_to_binary=True
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_ConstraintQualificationExample_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n Solving Constraint Qualification Example with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_ConstraintQualificationExample_integer_cut_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with Level-based Outer Approximation(no-good cuts, L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_no_good_cuts=True,
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_OnlineDocExample_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_OnlineDocExample_L_infinity_norm_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm="L_infinity"
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_MINLP4_simple_level_L_infinity(self):
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP4()
            print('\n Solving Online Doc Example with Level-based Outer Approximation(L_infinity)')
            results = opt.solve(model, strategy='OA',
                                add_regularization="level_L_infinity",
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                init_strategy='initial_binary',
                                level_coef=0.4
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.obj.expr), -56.981, places=2)


if __name__ == "__main__":
    unittest.main()
