from pyomo.contrib.appsi.base import PersistentBase
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat, NonNegativeInt
from .cmodel import cmodel, cmodel_available
from typing import List
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData, minimize, maximize
from pyomo.core.base.block import _BlockData


class IntervalConfig(ConfigDict):
    """
    Attributes
    ----------
    feasibility_tol: float
    integer_tol: float
    improvement_tol: float
    max_iter: int
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super(IntervalConfig, self).__init__(description=description,
                                           doc=doc,
                                           implicit=implicit,
                                           implicit_domain=implicit_domain,
                                           visibility=visibility)

        self.feasibility_tol: float = self.declare('feasibility_tol',
                                                   ConfigValue(domain=NonNegativeFloat, default=1e-8))
        self.integer_tol: float = self.declare('integer_tol',
                                               ConfigValue(domain=NonNegativeFloat, default=1e-5))
        self.improvement_tol: float = self.declare('improvement_tol',
                                                   ConfigValue(domain=NonNegativeFloat, default=1e-4))
        self.max_iter: int = self.declare('max_iter',
                                          ConfigValue(domain=NonNegativeInt, default=10))


class IntervalTightener(PersistentBase):
    def __init__(self):
        super(IntervalTightener, self).__init__()
        self._config = IntervalConfig()
        self._cmodel = None
        self._var_map = dict()
        self._con_map = dict()
        self._param_map = dict()
        self._rvar_map = dict()
        self._rcon_map = dict()
        self._pyomo_expr_types = cmodel.PyomoExprTypes()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val: IntervalConfig):
        self._config = val

    def set_instance(self, model):
        saved_config = self.config
        saved_update_config = self.update_config
        self.__init__()
        self.config = saved_config
        self.update_config = saved_update_config
        self._model = model
        self._cmodel = cmodel.Model()
        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    def _add_variables(self, variables: List[_GeneralVarData]):
        cmodel.process_pyomo_vars(self._pyomo_expr_types, variables, self._var_map, self._param_map,
                                  self._vars, self._rvar_map, False, None, None, False)

    def _add_params(self, params: List[_ParamData]):
        cparams = cmodel.create_params(len(params))
        for ndx, p in enumerate(params):
            cp = cparams[ndx]
            cp.value = p.value
            self._param_map[id(p)] = cp

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        cmodel.process_constraints(self._cmodel, self._pyomo_expr_types, cons, self._var_map, self._param_map,
                                   self._active_constraints, self._con_map, self._rcon_map)

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('IntervalTightener does not support SOS constraints')

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        for c in cons:
            cc = self._con_map.pop(c)
            self._cmodel.remove_constraint(cc)
            del self._rcon_map[cc]

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) != 0:
            raise NotImplementedError('IntervalTightener does not support SOS constraints')

    def _remove_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            cvar = self._var_map.pop(id(v))
            del self._rvar_map[cvar]

    def _remove_params(self, params: List[_ParamData]):
        for p in params:
            del self._param_map[id(p)]

    def _update_variables(self, variables: List[_GeneralVarData]):
        cmodel.process_pyomo_vars(self._pyomo_expr_types, variables, self._var_map, self._param_map,
                                  self._vars, self._rvar_map, False, None, None, True)

    def update_params(self):
        for p_id, p in self._params.items():
            cp = self._param_map[p_id]
            cp.value = p.value

    def _set_objective(self, obj: _GeneralObjectiveData):
        if obj is None:
            ce = cmodel.Constant(0)
            sense = 0
        else:
            ce = cmodel.appsi_expr_from_pyomo_expr(obj.expr, self._var_map, self._param_map, self._pyomo_expr_types)
            if obj.sense is minimize:
                sense = 0
            else:
                sense = 1
        cobj = cmodel.Objective(ce)
        cobj.sense = sense
        self._cmodel.objective = cobj

    def perform_fbbt(self, model: _BlockData):
        if model is not self._model:
            self.set_instance(model)
        else:
            self.update()
        n_iter = self._cmodel.perform_fbbt(self.config.feasibility_tol, self.config.integer_tol,
                                           self.config.improvement_tol, self.config.max_iter)
        for cv, v in self._rvar_map.items():
            v.setlb(cv.get_lb())
            v.setub(cv.get_ub())
        return n_iter

    def perform_fbbt_with_seed(self, model: _BlockData, seed_var: _GeneralVarData):
        if model is not self._model:
            self.set_instance(model)
        else:
            self.update()
        n_iter = self._cmodel.perform_fbbt_with_seed(self._var_map[id(seed_var)], self.config.feasibility_tol,
                                                     self.config.integer_tol, self.config.improvement_tol,
                                                     self.config.max_iter)
        for cv, v in self._rvar_map.items():
            v.setlb(cv.get_lb())
            v.setub(cv.get_ub())
        return n_iter
