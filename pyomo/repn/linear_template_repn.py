from __future__ import division

from pyomo.common.errors import DeveloperError
from pyomo.core.expr import (
    native_types, value, is_fixed,
)
from pyomo.core.expr.current import (
    ExpressionBase, ProductExpression, DivisionExpression,
    MonomialTermExpression, SumExpressionBase, LinearExpression,
    NegationExpression, 
    GetItemExpression, TemplateSumExpression,
    SymbolMap, identify_variables, resolve_template,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.base.label import NumericLabeler
from pyomo.core.base import Param, Var

from pyomo.repn.standard_repn import StandardRepn


_CONSTANT = 1
_MONOMIAL = 2
_LINEAR = 4
_GENERAL = 8

REMOVE_ZERO_COEF = False
INLINE = True

class CompiledGetItem(ExpressionBase):
    def __init__(self, expr, context):
        self.smap = SymbolMap(NumericLabeler('x'))
        self._expr_str = expr.to_string(smap=self.smap, compute_values=False)
        self._context = context
        self._locals = {key: obj() for key, obj in self.smap.bySymbol.items()}
        self._locals.update(context)
        self._eval = eval('lambda: %s' % (self._expr_str,), self._locals, {})

    def nargs(self):
        return 0

    _args_ = ()

    def __str__(self):
        return "CompiledGetitem(%s)" % (self._expr_str,)

    def _apply_operation(self, result):
        ans = self._resolve_template(result)
        if ans.__class__ in native_types:
            return ans
        if ans.is_fixed():
            return value(ans)
        else:
            return ans

    def _resolve_template(self, args=None):
        for k, v in self._context.items():
            self._locals[k] = v()
        return self._eval()


class linearTemplateRepn(object):
    __slots__ = ('linear', 'const', 'sum_tmpl')

    def __init__(self):
        self.linear = []
        self.sum_tmpl = []
        self.const = 0

    def merge(self, other):
        self.const += other.const
        self.linear.extend(other.linear)
        self.sum_tmpl.extend(other.sum_tmpl)

    def distribute_multiplicand(self, mult):
        self.const *= mult
        # for i in range(len(self.linear)):
        #     self.linear[i] *= mult
        self.linear = list((c*mult, v) for c,v in self.linear)
        for st in self.sum_tmpl:
            st.distribute_multiplicand(mult)

    def distribute_divisor(self, div):
        self.const /= div
        #for i in range(len(self.linear)):
        #    self.linear[i] /= div
        self.linear = list((c/div, v) for c,v in self.linear)
        for st in self.sum_tmpl:
            st.distribute_divisor(div)

    def toLinearExpr(self):
        ans = LinearExpression()
        ans.constant = self.const
        ans.linear_coefs, ans.linear_vars = zip(*self.linear)
        if REMOVE_ZERO_COEF:
            try:
                i = 0
                while 1:
                    i = ans.linear_coefs.index(0,i)
                    ans.linear_coefs.pop(i)
                    ans.linear_vars.pop(i)
            except ValueError:
                pass
        return ans

    def fromLinearExpr(self, linear):
        self.const += value(linear.constant)
        self.linear.extend(zip(linear.linear_coefs, linear.linear_vars))

    def _generateTemplateSums(self):
        yield self
        for sum_template in self.sum_tmpl:
            args = sum_template.args
            for i in range(len(args)):
                yield from args[i]._generateTemplateSums()

    def standardRepn(self):
        ans = StandardRepn()
        ans.linear_vars = _vars = []
        ans.linear_coefs = _coefs = []
        for ltr in self._generateTemplateSums():
            for c, v in ltr.linear:
                if hasattr(v, '_resolve_template'):
                    _vars.append(v._resolve_template())
                else:
                    _vars.append(v)
                if hasattr(c, '_resolve_template'):
                    _coefs.append(c._resolve_template())
                elif c.__class__ in native_types:
                    _coefs.append(c)
                else:
                    _coefs.append(c())
            if ltr.const.__class__ in native_types:
                ans.constant += ltr.const
            else:
                ans.constant += ltr.const()
        ans.constant = ans.constant
        return ans


class LinearTemplateRepnVisitor(StreamBasedExpressionVisitor):

    def __init__(self, template_indices):
        super().__init__()
        self.template_context = [{str(t): t for t in template_indices}]

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, None

    def beforeChild(self, node, child, child_idx):
        child_type = child.__class__
        if child_type in native_types:
            return False, (_CONSTANT, child)
        if not child.is_expression_type():
            if child.is_fixed():
                return False, (_CONSTANT, value(child))
            else:
                return False, (_MONOMIAL, 1, child)

        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #

        if child_type is MonomialTermExpression:
            arg1, arg2 = child._args_
            # if arg1.__class__ not in native_types:
            #     arg1 = value(arg1)
            if arg2.is_fixed():
                return False, (_CONSTANT, arg1 * arg2) #value(arg2))
            else:
                return False, (_MONOMIAL, arg1, arg2)

        if child_type is LinearExpression:
            # Because we are going to modify the LinearExpression in this
            # walker, we need to make a copy of the LinearExpression from
            # the original expression tree.
            linear = linearTemplateRepn()
            linear.fromLinearExpr(child)
            return False, (_LINEAR, linear)

        if child_type is GetItemExpression:
            _compiled = CompiledGetItem(child, self.template_context[-1])
            if child.arg(0).ctype is Param:
                return False, (_CONSTANT, _compiled)
            elif child.arg(0).ctype is Var:
                return False, (_MONOMIAL, 1, _compiled)
            else:
                raise DeveloperError(
                    "We do not (yet) support standard_repn templates "
                    "of expressions containing GetItemExpressions of "
                    "ctypes other than Var and Param")

        return True, None

    def enterNode(self, node):
        if node.__class__ is TemplateSumExpression:
            new_context = dict(self.template_context[-1])
            self.template_context.append(new_context)
            for t in node.index_templates():
                new_context[str(t)] = t
            return node._local_args_, linearTemplateRepn()
        elif isinstance(node, SumExpressionBase):
            return node.args, linearTemplateRepn()
        else:
            return node.args, []

    def acceptChildResult(self, node, data, child_result, child_idx):
        if data.__class__ is list:
            # General expression... cache the child result until the end
            data.append(child_result)
        else:
            # Linear Expression
            child_type = child_result[0]
            if child_type is _MONOMIAL:
                data.linear.append(child_result[1:])
            elif child_type is _CONSTANT:
                data.const += child_result[1]
            elif child_type is _LINEAR:
                data.merge(child_result[1])
            elif child_type is _GENERAL:
                if data.const or data.linear or data.sum_tmpl:
                    if not data.linear and not data.sum_tmpl:
                        return [(_CONSTANT, const), child_result]
                    return [(_LINEAR, data), child_result]
                else:
                    return [child_result]
        return data

    def exitNode(self, node, data):
        if data.__class__ is linearTemplateRepn:
            if node.__class__ is TemplateSumExpression:
                tmp = node.create_node_with_local_data((data,))
                data = linearTemplateRepn()
                data.sum_tmpl.append(tmp)
            return (_LINEAR, data)
        #
        # General (nonlinear) expression...
        #
        # If all the arguments are constants, then evaluate this node
        # if all(_[0] is _CONSTANT for _ in data):
        #     return (
        #         _CONSTANT,
        #         node._apply_operation(tuple(_[1] for _ in data))
        #     )
        # We need special handling for Product/Division expressions
        if len(data) == 1:
            arg1 = data[0]
            if isinstance(node, NegationExpression):
                if arg1[0] is _MONOMIAL:
                    return (_MONOMIAL, -1*arg1[1], arg1[2])
                elif arg1[0] is _LINEAR:
                    arg1[1].distribute_multiplicand(-1)
                    return arg1
                elif arg1[0] is _CONSTANT:
                    return (_CONSTANT, -1*arg1[1])

        elif len(data) == 2:
            if isinstance(node, ProductExpression):
                # If there is a constant argument, ensure that it is the
                # first argument.  As we are "compiling" the expression,
                # it is less critical that we preserve the argument
                # order (but of course still need to be deterministic)
                if data[1][0] is _CONSTANT:
                    arg2, arg1 = data
                else:
                    arg1, arg2 = data
                if arg1[0] is _CONSTANT:
                    if arg2[0] is _MONOMIAL:
                        return (_MONOMIAL, arg1[1]*arg2[1], arg2[2])
                    elif arg2[0] is _LINEAR:
                        arg2[1].distribute_multiplicand(arg1[1])
                        return arg2
                    elif arg2[0] is _CONSTANT:
                        return (_CONSTANT, arg1[1]*arg2[1])
            elif isinstance(node, DivisionExpression):
                arg1, arg2 = data
                if arg2[0] is _CONSTANT:
                    div = arg2[1]
                    if arg1[0] is _MONOMIAL:
                        return (_MONOMIAL, (arg1[1]/div, arg1[2])
                        )
                    elif arg1[0] is _LINEAR:
                        arg1[1].distribute_divisor(arg2[1])
                        return arg1
                    elif arg1[0] is _CONSTANT:
                        return (_CONSTANT, arg1[1]/arg2[1])

        # We need to convert data to valid expression objects
        args = tuple( _[1]*_[2] if _[0] is _MONOMIAL
                      else _[1].toLinearExpr() if _[0] is _LINEAR
                      else _[1] for _ in data)
        if all(_[0] is _CONSTANT for _ in data):
            return node._apply_operation(args)
        return (_GENERAL, node.create_node_with_local_data(args))

    def finalizeResult(self, result):
        result_type = result[0]
        if result_type is _LINEAR:
            return result[1]
        elif result_type is _GENERAL:
            #print("TODO: Separate Linear and Nonlinear terms")
            raise RuntimeError(
                "LinearTemplateRepnVisitor only supports linear "
                "expressions; found '%s' (type: %s)"
                % (result[1], result[1].__class__.__name__))
        elif result_type is _MONOMIAL:
            ans = linearTemplateRepn()
            if result[1]:
                ans.linear.append(result[1:])
        elif result_type is _CONSTANT:
            ans = linearTemplateRepn()
            ans.const = result[1]
        else:
            raise DeveloperError("unknown result type")
        return ans
