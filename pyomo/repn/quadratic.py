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

import copy

from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    Expr_ifExpression,
    LinearExpression,
    MonomialTermExpression,
    mutable_expression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.base.expression import Expression
from . import linear
from . import linear_template
from . import util
from .linear import _merge_dict, to_expression
import pyomo.core.expr as expr
from pyomo.core.expr import ExpressionType
from pyomo.common.numeric_types import native_types

import logging

code_type = copy.deepcopy.__class__

logger = logging.getLogger(__name__)

_CONSTANT = linear.ExprType.CONSTANT
_LINEAR = linear.ExprType.LINEAR
_GENERAL = linear.ExprType.GENERAL
_QUADRATIC = linear.ExprType.QUADRATIC


class QuadraticRepn(object):
    __slots__ = ("multiplier", "constant", "linear", "quadratic", "nonlinear")

    def __init__(self):
        self.multiplier = 1
        self.constant = 0
        self.linear = {}
        self.quadratic = {}
        self.nonlinear = None

    def __str__(self):
        return (
            f"QuadraticRepn(mult={self.multiplier}, const={self.constant}, "
            f"linear={self.linear}, quadratic={self.quadratic}, "
            f"nonlinear={self.nonlinear})"
        )

    def __repr__(self):
        return str(self)

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.quadratic:
            return _QUADRATIC, self
        elif self.linear:
            return _LINEAR, self
        else:
            return _CONSTANT, self.multiplier * self.constant

    def duplicate(self):
        logger.debug(f"DUPLICATE: {self}")
        ans = self.__class__.__new__(self.__class__)
        ans.multiplier = self.multiplier
        ans.constant = self.constant
        ans.linear = dict(self.linear)
        if self.quadratic:
            ans.quadratic = dict(self.quadratic)
        else:
            ans.quadratic = None
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(self, visitor):
        var_map = visitor.var_map
        logger.debug(f"var_map: {var_map}")
        logger.debug(
            f"self: linear={self.linear}, quadratic={self.quadratic}, nonlinear={self.nonlinear}"
        )
        logger.debug(f"constant: {self.constant}, multiplier: {self.multiplier}")
        if self.nonlinear is not None:
            # We want to start with the nonlinear term (and use
            # assignment) in case the term is a non-numeric node (like a
            # relational expression)
            ans = self.nonlinear
        else:
            ans = 0
        if self.quadratic:
            with mutable_expression() as e:
                for (x1, x2), coef in self.quadratic.items():
                    if x1 == x2:
                        e += coef * var_map[x1] ** 2
                    else:
                        e += coef * (var_map[x1] * var_map[x2])
            ans += e
        if self.linear:
            var_map = visitor.var_map
            with mutable_expression() as e:
                for vid, coef in self.linear.items():
                    if coef:
                        e += coef * var_map[vid]
            if e.nargs() > 1:
                ans += e
            elif e.nargs() == 1:
                ans += e.arg(0)
        if self.constant:
            ans += self.constant
        if self.multiplier != 1:
            ans *= self.multiplier
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a QuadraticRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        # Note that self.multiplier will always be 1 (we only call append()
        # within a sum, so there is no opportunity for self.multiplier to
        # change). Omitting the assertion for efficiency.
        # assert self.multiplier == 1
        _type, other = other
        if _type is _CONSTANT:
            self.constant += other
            return

        mult = other.multiplier
        if not mult:
            # 0 * other, so there is nothing to add/change about
            # self.  We can just exit now.
            return
        if other.constant:
            self.constant += mult * other.constant
        if other.linear:
            _merge_dict(self.linear, mult, other.linear)
        if other.quadratic:
            if not self.quadratic:
                self.quadratic = {}
            _merge_dict(self.quadratic, mult, other.quadratic)
        if other.nonlinear is not None:
            if mult != 1:
                nl = mult * other.nonlinear
            else:
                nl = other.nonlinear
            if self.nonlinear is None:
                self.nonlinear = nl
            else:
                self.nonlinear += nl


def _mul_linear_linear(visitor, linear1, linear2):
    logger.debug(f"linear1: {linear1}, linear2: {linear2}")
    quadratic = {}
    for vid1, coef1 in linear1.items():
        for vid2, coef2 in linear2.items():
            key = (min(vid1, vid2), max(vid2, vid1))
            quadratic[key] = quadratic.get(key, 0) + (coef1 * coef2)

    return quadratic


def _handle_product_linear_linear(visitor, node, arg1, arg2):
    logger.debug(f"{node}, {arg1}, {arg2}")
    _, arg1 = arg1
    _, arg2 = arg2
    # Quadratic first, because we will update linear in a minute
    arg1.quadratic = _mul_linear_linear(visitor, arg1.linear, arg2.linear)
    # Linear second, as this relies on knowing the original constants
    if not arg2.constant:
        arg1.linear = {}
    elif arg2.constant != 1:
        c = arg2.constant
        _linear = arg1.linear
        for vid, coef in _linear.items():
            _linear[vid] = c * coef
    if arg1.constant:
        _merge_dict(arg1.linear, arg1.constant, arg2.linear)
    # Finally, the constant and multipliers
    arg1.constant *= arg2.constant
    arg1.multiplier *= arg2.multiplier
    return _QUADRATIC, arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans

    # We are multiplying (A + Bx + Cx^2 + D(x)) * (A + Bx + Cx^2 + Dx))
    _, x1 = arg1
    _, x2 = arg2
    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1
    # x1.const * x2.const [AA]
    ans.constant = x1.constant * x2.constant
    # linear & quadratic terms
    if x2.constant:
        # [BA], [CA]
        c = x2.constant
        if c == 1:
            ans.linear = dict(x1.linear)
            if x1.quadratic:
                ans.quadratic = dict(x1.quadratic)
        else:
            ans.linear = {vid: c * coef for vid, coef in x1.linear.items()}
            if x1.quadratic:
                ans.quadratic = {k: c * coef for k, coef in x1.quadratic.items()}
    if x1.constant:
        # [AB]
        _merge_dict(ans.linear, x1.constant, x2.linear)
        # [AC]
        if x2.quadratic:
            if ans.quadratic:
                _merge_dict(ans.quadratic, x1.constant, x2.quadratic)
            elif x1.constant == 1:
                ans.quadratic = dict(x2.quadratic)
            else:
                c = x1.constant
                ans.quadratic = {k: c * coef for k, coef in x2.quadratic.items()}
    # [BB]
    if x1.linear and x2.linear:
        quad = _mul_linear_linear(visitor, x1.linear, x2.linear)
        if ans.quadratic:
            _merge_dict(ans.quadratic, 1, quad)
        else:
            ans.quadratic = quad
    # [DA] + [DB] + [DC] + [DD]
    ans.nonlinear = 0
    if x1.nonlinear is not None:
        ans.nonlinear += x1.nonlinear * x2.to_expression(visitor)
    x1.nonlinear = None
    x2.constant = 0
    x1_c = x1.constant
    x1.constant = 0
    x1_lin = x1.linear
    x1.linear = {}
    # [CB] + [CC] + [CD]
    if x1.quadratic:
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
        x1.quadratic = None
    x2.linear = {}
    # [BC] + [BD]
    if x1_lin and (x2.nonlinear is not None or x2.quadratic):
        x1.linear = x1_lin
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
    # [AD]
    if x1_c and x2.nonlinear is not None:
        ans.nonlinear += x1_c * x2.nonlinear
    return _GENERAL, ans


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    linear.define_exit_node_handlers(_exit_node_handlers)
    #
    # NEGATION
    #
    _exit_node_handlers[NegationExpression][(_QUADRATIC,)] = linear._handle_negation_ANY
    #
    # PRODUCT
    #
    _exit_node_handlers[ProductExpression].update(
        {
            None: _handle_product_nonlinear,
            (_CONSTANT, _QUADRATIC): linear._handle_product_constant_ANY,
            (_QUADRATIC, _CONSTANT): linear._handle_product_ANY_constant,
            # Replace handler from the linear walker
            (_LINEAR, _LINEAR): _handle_product_linear_linear,
        }
    )
    #
    # DIVISION
    #
    _exit_node_handlers[DivisionExpression].update(
        {(_QUADRATIC, _CONSTANT): linear._handle_division_ANY_constant}
    )
    #
    # EXPONENTIATION
    #
    _exit_node_handlers[PowExpression].update(
        {(_QUADRATIC, _CONSTANT): linear._handle_pow_ANY_constant}
    )
    #
    # ABS and UNARY handlers
    #
    # (no changes needed)
    #
    # NAMED EXPRESSION handlers
    #
    # (no changes needed)
    #
    # EXPR_IF handlers
    #
    # Note: it is easier to just recreate the entire data structure, rather
    # than update it
    _exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, i, _QUADRATIC): linear._handle_expr_if_const
            for i in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL)
        }
    )
    _exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, _QUADRATIC, i): linear._handle_expr_if_const
            for i in (_CONSTANT, _LINEAR, _GENERAL)
        }
    )
    #
    # RELATIONAL handlers
    #
    # (no changes needed)

    _exit_node_handlers[expr.GetItemExpression] = {
        None: linear_template._handle_getitem
    }
    _exit_node_handlers[expr.TemplateSumExpression] = {
        None: linear_template._handle_templatesum
    }

    return _exit_node_handlers


class QuadraticRepnVisitor(linear.LinearRepnVisitor):
    Result = QuadraticRepn
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        util.initialize_exit_node_dispatcher(define_exit_node_handlers())
    )
    max_exponential_expansion = 2

    ## handle quadratics, then let LinearRepnVisitor handle the rest
    def finalizeResult(self, result):
        ans = result[1]
        if (
            ans.__class__ is self.Result
            and ans.multiplier
            and ans.multiplier != 1
            and ans.quadratic
        ):
            mult = ans.multiplier
            quadratic = ans.quadratic
            zeros = []
            for vid, coef in quadratic.items():
                if coef:
                    quadratic[vid] = coef * mult
                else:
                    zeros.append(vid)
            for vid in zeros:
                del quadratic[vid]

        return super().finalizeResult(result)


class QuadraticTemplateRepn(QuadraticRepn):
    # __slots__ = ("linear_sum",)

    def __init__(self):
        super().__init__()
        self.linear_sum = []

    @classmethod
    def _resolve_symbols(
        cls, k, ans, expr_cache, smap, remove_fixed_vars, check_duplicates, var_map
    ):

        if isinstance(k, tuple):
            return (
                "("
                + ",".join(
                    [
                        cls._resolve_symbols(
                            _k,
                            ans,
                            expr_cache,
                            smap,
                            remove_fixed_vars,
                            check_duplicates,
                            var_map,
                        )
                        for _k in k
                    ]
                )
                + ")"
            )

        logger.debug(f"k={k} ({type(k)})")
        logger.debug(
            f"expr_cache: {expr_cache}, smap: {smap.bySymbol} {smap.byObject}, "
            f"var_map: {[(k,v._index,v._value,v._component()) for k, v in var_map.items()]}"
        )

        if k in expr_cache or k in var_map:
            if k in expr_cache:
                k = expr_cache[k]
            else:
                symbol_obj_id = id(var_map[k]._component())
                if symbol_obj_id in smap.byObject:
                    k = f"{smap.byObject[symbol_obj_id]}[{var_map[k]._index}]"

            logger.debug(f"k={k} ({type(k)})")

            if k.__class__ not in native_types and k.is_expression_type():
                ans.append("v = " + k.to_string(smap=smap))
                k = "v"
                if remove_fixed_vars:
                    ans.append("if v.__class__ is tuple:")
                    ans.append("    const += v[0] * {coef}")
                    ans.append("    v = None")
                    ans.append("else:")
                    indent = "    "  # FIX
                elif not check_duplicates:
                    k = ans.pop()[4:]

        return k

    def compile(
        self,
        env,
        smap,
        expr_cache,
        args,
        remove_fixed_vars=False,
        check_duplicates=False,
        *,
        var_map={},
    ):
        ans, constant = self._build_evaluator(
            smap, expr_cache, 1, 1, remove_fixed_vars, check_duplicates, var_map=var_map
        )
        if not ans:
            return constant
        indent = "\n    "
        if not constant and ans and ans[0].startswith("const +="):
            # Convert initial "const +=" to "const ="
            ans[0] = "".join(ans[0].split("+", 1))
        else:
            ans.insert(0, "const = " + repr(constant))
        fcn_body = indent.join(ans[1:])
        if "const" not in fcn_body:
            # No constants in the expression.  Move the initial const
            # term to the return value and avoid declaring the local
            # variable
            ans = ["return " + ans[0].split("=", 1)[1]]
            if fcn_body:
                ans.insert(0, fcn_body)
        else:
            ans = [ans[0], fcn_body, "return const"]
        if check_duplicates:
            ans.insert(0, f"def build_expr(linear, quadratic, {', '.join(args)}):")
        else:
            ans.insert(
                0,
                f"def build_expr(linear_indices, linear_data, quadratic_indices, quadratic_data, {', '.join(args)}):",
            )
        ans = indent.join(ans)
        import textwrap

        logger.debug(
            f"EXECUTING:\n\n{textwrap.indent(ans, '  ')}\n* compile RETURNING: build_expr\n"
        )
        # build the function in the env namespace, then remove and
        # return the compiled function.  The function's globals will
        # still be bound to env
        exec(ans, env)
        return env.pop("build_expr")

    def _build_evaluator(
        self,
        smap,
        expr_cache,
        multiplier,
        repetitions,
        remove_fixed_vars,
        check_duplicates,
        *,
        var_map=None,
    ):
        logger.debug(
            f"smap:{smap}, expr_cache:{expr_cache}, multiplier:{multiplier}, repetitions:{repetitions}, "
            f"remove_fixed_vars:{remove_fixed_vars}, check_duplicates:{check_duplicates}\n\n"
        )
        ans = []
        multiplier *= self.multiplier
        constant = self.constant
        if constant.__class__ not in native_types or constant:
            constant *= multiplier
            if not repetitions or (
                constant.__class__ not in native_types and constant.is_expression_type()
            ):
                ans.append("const += " + constant.to_string(smap=smap))
                constant = 0
            else:
                constant *= repetitions

        for term_type in ["linear", "quadratic"]:
            for k, coef in list(getattr(self, term_type).items()):
                logger.debug(f"{term_type}: ({k} ({k.__class__.__name__}): {coef})")
                coef *= multiplier
                if coef.__class__ not in native_types and coef.is_expression_type():
                    coef = coef.to_string(smap=smap)
                elif coef:
                    coef = repr(coef)
                else:
                    continue

                indent = ""

                k = self.__class__._resolve_symbols(
                    k,
                    ans,
                    expr_cache,
                    smap,
                    remove_fixed_vars,
                    check_duplicates,
                    var_map,
                )

                if check_duplicates:
                    ans.append(indent + f"if {k} in {term_type}:")
                    ans.append(indent + f"    {term_type}[{k}] += {coef}")
                    ans.append(indent + "else:")
                    ans.append(indent + f"    {term_type}[{k}] = {coef}")
                else:
                    ans.append(indent + f"{term_type}_indices.append({k})")
                    ans.append(indent + f"{term_type}_data.append({coef})")

        for subrepn, subindices, subsets in getattr(self, "linear_sum", []):
            ans.extend(
                "    " * i
                + f"for {','.join(smap.getSymbol(i) for i in _idx)} in "
                + (
                    _set.to_string(smap=smap)
                    if _set.is_expression_type()
                    else smap.getSymbol(_set)
                )
                + ":"
                for i, (_idx, _set) in enumerate(zip(subindices, subsets))
            )
            try:
                subrep = 1
                for _set in subsets:
                    subrep *= len(_set)
            except:
                subrep = 0
            subans, subconst = subrepn._build_evaluator(
                smap,
                expr_cache,
                multiplier,
                repetitions * subrep,
                remove_fixed_vars,
                check_duplicates,
            )
            indent = "    " * (len(subsets))
            ans.extend(indent + line for line in subans)
            constant += subconst
        return ans, constant


class QuadraticTemplateRepnVisitor(linear_template.LinearTemplateRepnVisitor):
    Result = QuadraticTemplateRepn
    max_exponential_expansion = 2
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        util.initialize_exit_node_dispatcher(define_exit_node_handlers())
    )

    ## handle quadratics, then let LinearRepnVisitor handle the rest.
    ## duplicate of QuadraticRepnVisitor, but not directly inheritable because of ambiguous
    ## multi-class inheritance (QuadaraticRepnVisitor vs. LinearTemplateRepnVisitor)
    def finalizeResult(self, result):
        ans = result[1]
        if (
            ans.__class__ is self.Result
            and ans.multiplier
            and ans.multiplier != 1
            and ans.quadratic
        ):
            mult = ans.multiplier
            quadratic = ans.quadratic
            zeros = []
            for vid, coef in quadratic.items():
                if coef:
                    quadratic[vid] = coef * mult
                else:
                    zeros.append(vid)
            for vid in zeros:
                del quadratic[vid]

        return super().finalizeResult(result)

    def expand_expression(self, obj, template_info):
        env = self.env
        logger.debug(f"obj={type(obj)}")
        logger.debug(
            f"template_info={type(template_info)}, {[type(ti) for ti in template_info]}"
        )
        logger.debug(f"id(template_info)={id(template_info)}")
        try:
            # attempt to look up already-constructed template
            body, lb, ub = self.expanded_templates[id(template_info)]
        except KeyError:
            # create a new expanded template
            logger.debug(f"create new expanded template")
            smap = self.symbolmap
            expr, indices = template_info
            args = [smap.getSymbol(i) for i in indices]
            if expr.is_expression_type(ExpressionType.RELATIONAL):
                logger.debug("expression_type = RELATIONAL")

                lb, body, ub = obj.to_bounded_expression()
                if body is not None:
                    body = self.walk_expression(body).compile(
                        env, smap, self.expr_cache, args, False, var_map=self.var_map
                    )
                if lb is not None:
                    lb = self.walk_expression(lb).compile(
                        env, smap, self.expr_cache, args, True, var_map=self.var_map
                    )
                if ub is not None:
                    ub = self.walk_expression(ub).compile(
                        env, smap, self.expr_cache, args, True, var_map=self.var_map
                    )

            elif expr is not None:
                lb = ub = None
                body = self.walk_expression(expr).compile(
                    env, smap, self.expr_cache, args, False, var_map=self.var_map
                )
            else:
                body = lb = ub = None
            self.expanded_templates[id(template_info)] = body, lb, ub
            logger.debug(
                f"SET: {template_info} self.expanded_templates[{id(template_info)}] = {body}, {lb}, {ub}"
            )

        linear_indices = []
        linear_data = []
        quadratic_indices = []
        quadratic_data = []
        call_args = (linear_indices, linear_data, quadratic_indices, quadratic_data)

        index = obj.index()
        if index.__class__ is not tuple:
            if index is None and not obj.parent_component().is_indexed():
                index = ()
            else:
                index = (index,)
        if lb.__class__ is code_type:
            lb = lb(*call_args, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed lower bound")
        if ub.__class__ is code_type:
            ub = ub(*call_args, *index)
            if linear_indices:
                raise RuntimeError(f"Constraint {obj} has non-fixed upper bound")

        return (
            body(*call_args, *index),
            linear_indices,
            linear_data,
            lb,
            ub,
            quadratic_indices,
            quadratic_data,
        )
