#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import json
import pyutilib.th as unittest
from pyutilib.services import TempfileManager

from pyomo.environ import (
    ConcreteModel, Var, Param, Expression, Constraint, Objective, 
    exp, sin, cos,
)
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base import CNameLabeler
from pyomo.opt import WriterFactory

class TestMathOptFormat(unittest.TestCase):
    def test_generate_Linear_MOF(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1,2])
        m.p = Param(initialize=5, mutable=True)

        MOF = WriterFactory('mathoptformat')
        SM = SymbolMap(CNameLabeler())

        json_data = MOF.generate_MOF_function(
            m.p + m.x + m.p*m.y[1] - (m.y[1] + 2*m.y[2]), SM )
        self.assertEqual(
            json_data,
            {'constant': 5,
             'head': 'ScalarAffineFunction',
             'terms': [{'coefficient': 1,
                        'head': 'ScalarAffineTerm',
                        'variable_index': 'x'},
                       {'coefficient': 4,
                        'head': 'ScalarAffineTerm',
                        'variable_index': 'y[1]'},
                       {'coefficient': -2,
                        'head': 'ScalarAffineTerm',
                        'variable_index': 'y[2]'}]}
        )

    def test_generate_Quadratic_MOF(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1,2])
        m.p = Param(initialize=5, mutable=True)

        MOF = WriterFactory('mathoptformat')
        SM = SymbolMap(CNameLabeler())

        json_data = MOF.generate_MOF_function(
            m.p + m.x + 2*m.y[1]*m.x - (m.p*m.y[1]**2 + 2*m.y[2]*m.x), SM )
        self.assertEqual(
            json_data,
            {'head': 'ScalarQuadraticFunction',
             'constant': 5,
             'affine_terms': [
                 {'coefficient': 1,
                  'head': 'ScalarAffineTerm',
                  'variable_index': 'x'}],
             'quadratic_terms': [
                 {'coefficient': 2,
                  'head': 'ScalarQuadraticTerm',
                  'variable_index_1': 'x',
                  'variable_index_2': 'y[1]'},
                 {'coefficient': -5,
                  'head': 'ScalarQuadraticTerm',
                  'variable_index_1': 'y[1]',
                  'variable_index_2': 'y[1]'},
                 {'coefficient': -2,
                  'head': 'ScalarQuadraticTerm',
                  'variable_index_1': 'x',
                  'variable_index_2': 'y[2]'}]}
        )

    def test_generate_NL_MOF(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1,2])
        m.p = Param(initialize=5, mutable=True)
        m.e = Expression(expr=sin(m.y[1]**2 + 2*m.y[2]/m.x))

        MOF = WriterFactory('mathoptformat')
        SM = SymbolMap(CNameLabeler())

        json_data = MOF.generate_MOF_function(
            m.p + exp(m.x) + cos(m.p-5) + m.y[1]*m.x - m.e,
            SM )

        self.assertEqual(
            json_data,
            {'head': 'nonlinear',
             'root': {'head': '+',
                      'args': [{'head': 'real',
                                'value': 5},
                               {'index': 7,
                                'head': 'node'},
                               {'head': 'real',
                                'value': 1},
                               {'index': 8,
                                'head': 'node'},
                               {'index': 9,
                                'head': 'node'}]},
             'node_list': [{'head': '*',
                            'args': [{'head': 'real',
                                      'value': 2},
                                     {'head': 'variable',
                                      'name': 'y[2]'}]},
                           {'head': '/',
                            'args': [{'head': 'real',
                                      'value': 1},
                                     {'head': 'variable',
                                      'name': 'x'}]},
                           {'head': '^',
                            'args': [{'head': 'variable',
                                      'name': 'y[1]'},
                                     {'head': 'real',
                                      'value': 2}]},
                           {'head': '*',
                            'args': [{'index': 1,
                                      'head': 'node'},
                                     {'index': 2,
                                      'head': 'node'}]},
                           {'head': '+',
                            'args': [{'index': 3,
                                      'head': 'node'},
                                     {'index': 4,
                                      'head': 'node'}]},
                           {'head': 'sin',
                            'args': [{'index': 5,
                                      'head': 'node'}]},
                           {'head': 'exp',
                            'args': [{'head': 'variable',
                                      'name': 'x'}]},
                           {'head': '*',
                            'args': [{'head': 'variable',
                                      'name': 'y[1]'},
                                     {'head': 'variable',
                                      'name': 'x'}]},
                           {'head': '*',
                            'args': [{'head': 'real',
                                      'value': -1},
                                     {'index': 6,
                                      'head': 'node'}]}]}
        )

    
    def test_generate_repn_write_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.o = Objective(expr=m.x+m.y)
        m.c1 = Constraint(expr= m.x >= 0)
        m.c2 = Constraint(expr= m.y <= 5)
        m.c3 = Constraint(expr= m.x == 4)
        m.c4 = Constraint(expr= 1 <= m.y <= 5)

        ref_JSON = {
            'objectives': [
                {'function': {
                    'terms': [
                        {'coefficient': 1,
                         'head': 'ScalarAffineTerm',
                         'variable_index': 'x'},
                        {'coefficient': 1,
                         'head': 'ScalarAffineTerm',
                         'variable_index': 'y'}],
                    'head': 'ScalarAffineFunction',
                    'constant': 0},
                 'sense': 'minimize'}],
            'version': 1,
            'variables': [{'name': 'x'},
                          {'name': 'y'}],
            'constraints': [
                {'function': {'terms': [{'coefficient': 1,
                                         'head': 'ScalarAffineTerm',
                                         'variable_index': 'x'}],
                              'head': 'ScalarAffineFunction',
                              'constant': 0},
                 'set': {'head': 'GreaterThan',
                         'lower': 0.0},
                 'name': 'c1'},
                {'function': {'terms': [{'coefficient': 1,
                                         'head': 'ScalarAffineTerm',
                                         'variable_index': 'y'}],
                              'head': 'ScalarAffineFunction',
                              'constant': 0},
                 'set': {'upper': 5.0,
                         'head': 'LessThan'},
                 'name': 'c2'},
                {'function': {'terms': [{'coefficient': 1,
                                         'head': 'ScalarAffineTerm',
                                         'variable_index': 'x'}],
                              'head': 'ScalarAffineFunction',
                              'constant': 0},
                 'set': {'head': 'EqualTo',
                         'value': 4.0},
                 'name': 'c3'},
                {'function': {'terms': [{'coefficient': 1,
                                         'head': 'ScalarAffineTerm',
                                         'variable_index': 'y'}],
                              'head': 'ScalarAffineFunction',
                              'constant': 0},
                 'set': {'upper': 5.0,
                         'head': 'Interval',
                         'lower': 1.0},
                 'name': 'c4'}
            ]
        }
    
        MOF = WriterFactory('mathoptformat')
        SM = SymbolMap(CNameLabeler())

        json1 = MOF.generate_repn(m, SM, CNameLabeler())
        self.assertEqual(json1, ref_JSON)

        try:
            TempfileManager.push()
            tmpfile = TempfileManager.create_tempfile(suffix='.mof.json')
            MOF(m, tmpfile)
            with open(tmpfile) as FILE:
                json2 = json.load(FILE)
        finally:
            TempfileManager.pop(remove=True)

        self.assertEqual(json2, ref_JSON)

