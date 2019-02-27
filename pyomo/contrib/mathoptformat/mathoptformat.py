#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Problem Writer for MathOptFormat JSON files
#

try:
    import ujson as json
except:
    import json

import logging
from six import itervalues

from pyutilib.misc import PauseGC

from pyomo.core.expr.numvalue import (
    value, polynomial_degree, native_numeric_types
)
from pyomo.core.expr.symbol_map import SymbolMap 
from pyomo.core.expr.current import (
    StreamBasedExpressionVisitor,
    AbsExpression, LinearExpression, NegationExpression, NPV_AbsExpression,
    NPV_ExternalFunctionExpression, NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression, PowExpression, ProductExpression,
    ReciprocalExpression, StreamBasedExpressionVisitor, SumExpression,
    UnaryFunctionExpression,
)
from pyomo.core.expr.expr_pyomo5 import nonpyomo_leaf_types
from pyomo.core.base import (
    CNameLabeler, Objective, Constraint, Var, minimize, maximize
)
from pyomo.core.base.expression import _ExpressionData
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn import generate_standard_repn

NPV_expressions = (
    NPV_AbsExpression, NPV_ExternalFunctionExpression,
    NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression
)

class MathOptFormatWalker(StreamBasedExpressionVisitor):
    def __init__(self, symbol_map):
        super(MathOptFormatWalker, self).__init__()
        self.symbol_map = symbol_map
        self.node_list = None

    def walk_expression(self, expr):
        self.node_list = []
        return super(MathOptFormatWalker, self).walk_expression(expr)

    def beforeChild(self, node, child):
        #
        # Don't replace native types
        #
        if type(child) in native_numeric_types:
            return False, {'head': "real", "value": child}
        #
        # We will descend into all expressions...
        #
        if child.is_expression_type():
            return True, None
        #
        # Replace pyomo variables with sympy variables
        #
        if not child.is_fixed():
            return False, {"head": "variable",
                           "name": str(self.symbol_map.getSymbol(child))}
        #
        # Everything else is a constant...
        #
        return False, {"head": "real", "value": value(child)}

    def exitNode(self, node, values):
        if isinstance(node, _ExpressionData):
            return values[0]

        for i,v in enumerate(values):
            if v['head'] not in {'real','variable'}:
                self.node_list.append(v)
                values[i] = {"head": "node", "index": len(self.node_list)}

        if type(node) in nonpyomo_leaf_types:
            return {"head": "real", "value": value(node)},
        elif not node.is_expression_type() and not node.is_fixed():
            return {"head": "variable",
                    "name": str(self.symbol_map.getSymbol(node))},
        elif isinstance(node, NPV_expressions):
            return {"head": "real", "value": value(node)}
        elif isinstance(node, ProductExpression):
            head = "*"
        elif isinstance(node, SumExpression):
            head = "+"
        elif isinstance(node, PowExpression):
            head = "^"
        elif isinstance(node, ReciprocalExpression):
            head = "/"
            values.insert(0, {'head': "real", "value": 1})
        elif isinstance(node, NegationExpression):
            head = "*"
            values.insert(0, {'head': "real", "value": -1})
        elif isinstance(node, AbsExpression):
            raise NotImplementedError(
                'MathOptFormat does not aupport abs()')
        elif isinstance(node, LinearExpression):
            assert not values
            for c,v in zip(node.linear_coefs, linear_ars):
                self.node_list.append(
                    {'head': '*',
                     'args': [
                         {"head": "real", "value": c},
                         {"head": "variable",
                          "name": str(self.symbol_map.getSymbol(v))},
                     ]})
                values.append({"head": "node", "index": len(self.node_list)})
            if node.constant:
                values.append({"head": "real", "value": node.constant})
            head = '+'
        elif isinstance(node, UnaryFunctionExpression):
            if node.name in {'exp','log','log10','ceil','floor','sqrt',
                             'sin','asin','sinh','asinh',
                             'cos','acos','cosh','acosh',
                             'tan','atan','tanh','atanh'}:
                head = node.name
            else:
                raise NotImplementedError("Unknown unary function: %s"
                                          % (node.name,))
        else:
            raise RuntimeError("Unhandled expression type: %s" % (type(node)))
        return {
            'head': head,
            'args': values
        }


@WriterFactory.register('mathoptformat',
                        'Write the model in MathOptFormat JSON format')
class MathOptFormatWriter(AbstractProblemWriter):
    def __init__(self):
        AbstractProblemWriter.__init__(self, 'MathOptFormat')

    def __call__(self, model, output_filename, 
                 solver_capability=None, io_options=None):
        assert not io_options

        labeler = CNameLabeler()
        symbol_map = SymbolMap(labeler)

        with PauseGC() as pgc:
            model_repn = self.generate_repn(model, symbol_map, labeler)
        with open(output_filename, 'w') as FILE:
            # Note: dumps() is significantly faster than dump().
            #       See the (wontfix) bug report here:
            #       https://bugs.python.org/issue12134
            #json.dump(model_repn, FILE)
            FILE.write(json.dumps(model_repn))

    def generate_repn(self, model, symbol_map, labeler):
        ans = {'version': 1}
        obj = ans['objectives'] = []
        for o in model.component_data_objects(Objective):
            obj.append({
                'sense': 'minimize' if o.sense == minimize else 'maximize',
                'function': self.generate_MOF_function(o.expr, symbol_map),
            })
        constr = ans['constraints'] = []
        for c in model.component_data_objects(Constraint):
            tmp = {
                'name': str(symbol_map.getSymbol(c, labeler)),
                'function': self.generate_MOF_function(c.body, symbol_map),
            }
            lb = value(c.lower)
            ub = value(c.upper)
            if lb is None:
                tmp['set'] = {'head': 'LessThan', 'upper': ub}
            elif ub is None:
                tmp['set'] = {'head': 'GreaterThan', 'lower': lb}
            elif lb == ub:
                tmp['set'] = {'head': 'EqualTo', 'value': lb}
            else:
                tmp['set'] = {'head': 'Interval', 'lower': lb, 'upper': ub}
            constr.append(tmp)
        ans['variables'] = [
            {'name': str(n)} for n in 
            sorted( n for n in itervalues(symbol_map.byObject)
                    if symbol_map.bySymbol[n]().type() is Var )
        ]
        return ans

    def generate_MOF_function(self, expr, symbol_map):
        repn = generate_standard_repn(expr)
        if repn.nonlinear_expr is not None:
            MOFwalker = MathOptFormatWalker(symbol_map)
            ans = {
                'head': 'nonlinear',
                'root': MOFwalker.walk_expression(expr),
            }
            # node_list is calculated as a side-effect of the walker
            ans['node_list'] = MOFwalker.node_list
            return ans
        elif repn.quadratic_coefs:
            return {
                "head": "ScalarQuadraticFunction",
                "constant": value(repn.constant),
                "affine_terms": [
                    {
                        "head": "ScalarAffineTerm",
                        "coefficient": value(repn.linear_coefs[i]),
                        "variable_index": symbol_map.getSymbol(v),
                    } for i,v in enumerate(repn.linear_vars)
                ],
                "quadratic_terms": [
                    {
                        "head": "ScalarQuadraticTerm",
                        "coefficient": value(repn.quadratic_coefs[i]),
                        "variable_index_1": symbol_map.getSymbol(v[0]),
                        "variable_index_2": symbol_map.getSymbol(v[1]),
                    } for i,v in enumerate(repn.quadratic_vars)
                ],
            }
        else:
            return {
                "head": "ScalarAffineFunction",
                "constant": value(repn.constant),
                "terms": [
                    {
                        "head": "ScalarAffineTerm",
                        "coefficient": value(c),
                        "variable_index": symbol_map.getSymbol(v),
                    } for c,v in zip(repn.linear_coefs, repn.linear_vars)
                ],
            }

