"""
Class for implementing and running the Stage 1 MILP for each individual prosumer.
"""
import re

from pulp import *
from loguru import logger
from .helpers.helpers4milp import *


class MILP:
	def __init__(self, backpack, solver='CBC', timeout=1E9, mipgap=0.01):

		# Indices and sets
		self._horizon = backpack.get('horizon')  # operation period (hours)

		# Parameters
		self._delta_t = backpack.get('delta_t') / 60  # length of time intervals (hours)
		self._pcc_limit = backpack.get('pcc_limit')  # power limit for transactions at the Point of Common Coupling (kW)
		self._buy_price = backpack.get('buy_price')  # forecasted prices (€/kWh)
		self._sell_tariff = backpack.get('sell_tariff')  # forecasted tariffs (€/kWh)
		self._int_co2 = backpack.get('int_co2')  # forecasted grid energy mix CO2 intensity (g/kWh)
		self._p_load = backpack.get('p_load')  # forecasted average power consumption (kW)
		self._p_gen_pv = backpack.get('p_gen_pv')  # forecasted average photovoltaic power generation (kW)
		self._p_gen_wind = backpack.get('p_gen_wind')  # forecasted average wind power generation (kW)
		self._p_nominal = backpack.get('p_nominal')  # nominal load
		self._p_min = backpack.get('p_min')  # minimum load

		# Cooling
		self._t_amb = backpack.get('t_ambient')  # ambient temperature
		self._t_0 = [12] * 24  # chiller evaporator temperature

		# Battery storage
		self._batteries = backpack.get('batteries')
		self._nbat = len(self._batteries)
		self._b_capacity = [b.get('b_capacity') for b in self._batteries]   # the battery's nominal capacity (kWh)
		self._b_init_energy = [b.get('b_init_energy') for b in self._batteries]  # the battery's initial energy content (kWh)
		self._b_soc_min = [b.get('b_soc_min') for b in self._batteries]  # the battery's minimum state of charge (%)
		self._b_soc_max = [b.get('b_soc_max') for b in self._batteries]  # the battery's maximum state of charge (%)
		self._b_pch_max = [b.get('b_pch_max') for b in self._batteries]  # the battery's maximum charge power (kW)
		self._b_pdis_max = [b.get('b_pdis_max') for b in self._batteries]  # the battery's maximum discharge power (kW)
		self._b_ch_eff = [b.get('b_ch_eff') for b in self._batteries]  # the battery's constant charge efficiency
		self._b_dis_eff = [b.get('b_dis_eff') for b in self._batteries]  # the battery's constant discharge efficiency

		# MILP variables
		self.milp = None  # for storing the MILP formulation
		self.solver = solver  # solver chosen for the MILP
		self.timeout = timeout  # solvers temporal limit to find optimal solution (s)
		self.mipgap = mipgap  # controls the solvers tolerance; intolerant [0 - 1] fully permissive
		self.status = None  # stores the status of the MILP's solution
		self.obj_value = None  # stores the MILP's numeric solution
		self.prosumer_id = backpack.get('prosumer_id')  # identification of the prosumer for which te MILP will run (cliente)
		self.time_intervals = None  # for number of time intervals per horizon
		self.time_series = None  # for a range of time intervals
		self.obj_function = backpack.get('obj_function')  # chosen objective function

	def __define_milp(self):
		"""
		Method to define the first stage MILP problem.
		:return:
		"""

		# Define a minimization MILP
		self.milp = LpProblem(f'stage1_{self.prosumer_id}', LpMinimize)

		# Additional temporal variables
		self.time_intervals = time_intervals(self._horizon, self._delta_t)
		self.time_series = range(self.time_intervals)

		# Initialize the decision variables
		b_energy = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # the battery's current stored energy at time step t (kWh)
		b_soc = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # the battery's current state of charge at time step t (%)
		b_pch = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # the battery's charge power set point at time step t (kW)
		b_pdis = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # the battery's discharge power set point at time step t (kW)
		b_pbin = [zero_lists(self.time_intervals) for _ in range(self._nbat)] # binary variable (non charge/discharge simultaneity)
		pcc_pin = zero_lists(self.time_intervals)  # power imported from the grid at time step t (kW)
		pcc_pout = zero_lists(self.time_intervals)  # power exported to the grid at time step t (kW)
		pcc_pbin = zero_lists(self.time_intervals)  # binary variable (non import/export simultaneity)

		v_load = zero_lists(self.time_intervals)  # power for adjusted load (kW)
		v_computing = zero_lists(self.time_intervals)  # power for adjusted load (kW)
		v_cooling = zero_lists(self.time_intervals)  # power for adjusted load (kW)
		p_total = zero_lists(self.time_intervals)  # power for adjusted load (kW)
		p_cooling = zero_lists(self.time_intervals)  # power for adjusted load (kW)
		p_gen_pv_load = zero_lists(self.time_intervals)  # power from PV to fulfill load (kW)
		p_gen_pv_bat = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # power from PV to charge batteries at time step t (kW)
		p_gen_wind_load = zero_lists(self.time_intervals)  # power from Wind to fulfill load at time step t (kW)
		p_gen_wind_bat = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # power from Wind to charge batteries at time step t (kW)
		b_pdis_load = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # power discharged from batteries to fulfill load at time step t (kW)
		b_pdis_grid = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # power discharged from batteries to grid export at time step t (kW)
		b_pch_grid = [zero_lists(self.time_intervals) for _ in range(self._nbat)]  # power imported from grid to charge batteries at time step t (kW)
		pcc_pin_load = zero_lists(self.time_intervals)  # power imported from grid to fulfill load at time step t (kW)
		pcc_pout_pv = zero_lists(self.time_intervals)  # power from PV exported to grid at time step t (kW)
		pcc_pout_wind = zero_lists(self.time_intervals)  # power from Wind exported to grid at time step t (kW)

		# Define the decision variables as puLP objets
		for t in self.time_series:
			increment = f'{t:03d}'
			pcc_pin[t] = LpVariable('pcc_pin_' + increment, 0)
			pcc_pout[t] = LpVariable('pcc_pout_' + increment, 0)
			pcc_pbin[t] = LpVariable('pcc_pbin_' + increment, cat=LpBinary)
			v_load[t] = LpVariable('v_load_' + increment, 0)
			v_computing[t] = LpVariable('v_computing_' + increment, 0)
			v_cooling[t] = LpVariable('v_cooling_' + increment, 0)
			p_total[t] = LpVariable('p_total_' + increment, 0)
			p_cooling[t] = LpVariable('p_cooling_' + increment, 0)
			p_gen_pv_load[t] = LpVariable('p_gen_pv_load_' + increment, 0)
			p_gen_wind_load[t] = LpVariable('p_gen_wind_load_' + increment, 0)
			pcc_pin_load[t] = LpVariable('pcc_pin_load_' + increment, 0)
			pcc_pout_pv[t] = LpVariable('pcc_pout_pv_' + increment, 0)
			pcc_pout_wind[t] = LpVariable('pcc_pout_wind_' + increment, 0)

			for b, bat in enumerate(self._batteries):
				name = bat.get('b_name')
				increment = f'{name}_{t:03d}'
				b_energy[b][t] = LpVariable('b_energy_' + increment, 0)
				b_soc[b][t] = LpVariable('b_soc_' + increment, 0)
				b_pch[b][t] = LpVariable('b_pch_' + increment, 0)
				b_pdis[b][t] = LpVariable('b_pdis_' + increment, 0)
				b_pbin[b][t] = LpVariable('b_pbin_' + increment, cat=LpBinary)
				p_gen_pv_bat[b][t] = LpVariable('p_gen_pv_bat_' + increment, 0)
				p_gen_wind_bat[b][t] = LpVariable('p_gen_wind_bat_' + increment, 0)
				b_pdis_load[b][t] = LpVariable('b_pdis_load_' + increment, 0)
				b_pdis_grid[b][t] = LpVariable('b_pdis_grid_' + increment, 0)
				b_pch_grid[b][t] = LpVariable('b_pch_grid_' + increment, 0)

		# Eq. 1: Objective Function
		if self.obj_function == 'min_costs':  # Minimize operational costs
			objective = lpSum(self._buy_price[t] * pcc_pin[t] - self._sell_tariff[t] * pcc_pout[t] for t in self.time_series)
		elif self.obj_function == 'min_co2':  # Minimize CO2
			objective = lpSum(self._int_co2[t] * pcc_pin[t] for t in self.time_series)
		elif self.obj_function == 'max_ssr':  # Maximize self-sufficiency
			objective = lpSum(pcc_pin[t] for t in self.time_series)

		self.milp += objective, 'Objective Function'

		# Constraints
		for t in self.time_series:
			increment = f'{t:03d}'

			# Sum computing and cooling power
			#COP = self._t_0[t] / (self._t_amb[t] - self._t_0[t])
			COP = 4
			self.milp += p_cooling[t] == (1/COP) * self._p_load[t], 'CoolingLoad_' + increment
			self.milp += p_total[t] == self._p_load[t] + p_cooling[t], 'TotalLoad_' + increment

			# Sum of forecasted load is equal to the adjusted load
			#self.milp += v_load[t] == p_total[t]  # No load shaping
			self.milp += lpSum(v_load[t] for t in self.time_series) == lpSum(p_total[t] for t in self.time_series)
			#self.milp += lpSum(v_load[t] for t in range(0, t + 1)) <= lpSum(p_total[t] for t in range(0, t + 1))

			# Load is fulfilled by PV, Wind, Battery Discharge and Grid (x)
			self.milp += v_load[t] == p_gen_pv_load[t] + p_gen_wind_load[t] + lpSum([b_pdis_load[b][t] for b in range(self._nbat)]) + pcc_pin_load[t], 'EquilibriumLoad_' + increment
			self.milp += v_computing[t] <= self._p_nominal, 'NominalLoad_' + increment

			# Minimum load to fulfill
			self.milp += v_computing[t] >= self._p_min[t], 'MinimumLoad_' + increment

			# Total load is composed by computing and cooling load
			self.milp += v_load[t] == v_computing[t] + v_cooling[t], 'OptimalLoad_' + increment
			self.milp += v_cooling[t] == (1/COP) * v_computing[t], 'OptimalCooling_' + increment

			# PV generation can be used to fulfill the load, to grid feed-in or battery charging
			self.milp += self._p_gen_pv[t] == p_gen_pv_load[t] + pcc_pout_pv[t] + lpSum([p_gen_pv_bat[b][t] for b in range(self._nbat)]), 'EquilibriumPV_' + increment

			# Wind generation can be used to fulfill the load, to grid feed-in or battery charging
			self.milp += self._p_gen_wind[t] == p_gen_wind_load[t] + pcc_pout_wind[t] + lpSum([p_gen_wind_bat[b][t] for b in range(self._nbat)]), 'EquilibriumWind_' + increment

			# Grid Export comes from PV/wind generation or battery discharging
			self.milp += pcc_pout[t] == pcc_pout_pv[t] + pcc_pout_wind[t] + lpSum([b_pdis_grid[b][t] for b in range(self._nbat)]), 'EquilibriumOut_' + increment

			# Grid Import is used to fulfill the load or battery charging
			self.milp += pcc_pin[t] == pcc_pin_load[t] + lpSum([b_pch_grid[b][t] for b in range(self._nbat)]), 'EquilibriumIn_' + increment

			# Prevent importing and exporting power simultaneously
			self.milp += pcc_pin[t] <= self._pcc_limit * pcc_pbin[t], 'PCC_abs_limit_' + increment
			self.milp += pcc_pout[t] <= self._pcc_limit * (1 - pcc_pbin[t]), 'PCC_inj_limit_' + increment

			for b in range(self._nbat):
				increment = f'{b:02d}_{t:03d}'

				# Battery Charging comes from PV/Wind generation or grid
				self.milp += b_pch[b][t] == p_gen_pv_bat[b][t] + p_gen_wind_bat[b][t] + b_pch_grid[b][t], 'EquilibriumCharge_' + increment

				# Battery Discharging is used to fulfill the load or to grid feed-in
				self.milp += b_pdis[b][t] == b_pdis_load[b][t] + b_pdis_grid[b][t], 'EquilibriumDischarge_' + increment

				if self._b_capacity[b] > 0:
					# Prevent charging and discharging power from batteries simultaneously. Charge and discharge rate limits
					self.milp += b_pch[b][t] <= self._b_pch_max[b] * b_pbin[b][t], 'Battery_charge_rate_limit_' + increment
					self.milp += b_pdis[b][t] <= s
					
					
					
					
					elf._b_pdis_max[b] * (1 - b_pbin[b][t]), 'Battery_discharge_rate_limit' + increment

					# Battery energy update
					energy_update = (b_pch[b][t] * self._b_ch_eff[b] - b_pdis[b][t] * 1 / self._b_dis_eff[b]) * self._delta_t
					if t == 0:
						self.milp += b_energy[b][t] == self._b_init_energy[b] + energy_update, 'Initial_SOC_update_' + increment

					else:
						self.milp += b_energy[b][t] == b_energy[b][t - 1] + energy_update * self._delta_t, 'SOC_update_' + increment

					# State of charge limits
					self.milp += b_soc[b][t] == b_energy[b][t] * 100 / self._b_capacity[b], 'Energy_to_SOC_' + increment
					self.milp += self._b_soc_min[b] <= b_soc[b][t], 'Minimum_SOC_' + increment
					self.milp += b_soc[b][t] <= self._b_soc_max[b], 'Maximum_SOC_' + increment

				else:
					# Auxiliary equations for prosumer without BESS assets
					self.milp += b_pch[b][t] == 0, 'Battery_charge_rate_limit_' + increment
					self.milp += b_pdis[b][t] == 0, 'Battery_discharge_rate_limit_' + increment

		# Write MILP to .lp file
		dir_name = os.path.abspath(os.path.join(__file__, '..'))
		lp_file = os.path.join(dir_name, f'Stage1_{self.prosumer_id}.lp')
		self.milp.writeLP(lp_file)

		# Set the solver to be called
		if self.solver == 'CBC':
			self.milp.setSolver(pulp.PULP_CBC_CMD(msg=False, timeLimit=self.timeout, gapRel=self.mipgap))
		else:
			raise ValueError

		return

	def solve_milp(self):
		"""
		Function that heads the definition and solution of the MILP.
		:return:
		"""
		# Define the MILP
		self.__define_milp()

		# Solve the MILP
		try:
			self.milp.solve()
			status = LpStatus[self.milp.status]
			opt_value = value(self.milp.objective)

		except Exception as e:
			logger.warning(f'Solver raised an error: \'{e}\'. Considering problem as "Infeasible".')
			status = 'Infeasible'
			opt_value = None

		self.status = status
		self.obj_value = opt_value

		# Case when no objective value is found since all data is 0 (for testing purposes)
		if self.status == 'Optimal' and self.obj_value is None:
			self.obj_value = 0

		return

	def generate_outputs(self):
		"""
		Function for generating the outputs of optimization, namely the battery's set points.
		:return:
		"""

		outputs = {}

		# -- Verification added to avoid raising error whenever encountering a puLP solver error with CBC
		if self.obj_value is not None:
			outputs['prosumer_id'] = self.prosumer_id
			outputs['obj_value'] = self.obj_value
			outputs['milp_status'] = self.status

			for b, bat in enumerate(self._batteries):
				name = bat.get('b_name')
				outputs['b_energy_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_soc_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_pch_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_pdis_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['p_gen_wind_bat_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_pdis_load_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['p_gen_pv_bat_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_pdis_grid_' + f'{name}'] = zero_lists(self.time_intervals)
				outputs['b_pch_grid_' + f'{name}'] = zero_lists(self.time_intervals)

			outputs['pcc_pin'] = zero_lists(self.time_intervals)
			outputs['pcc_pout'] = zero_lists(self.time_intervals)
			outputs['v_load'] = zero_lists(self.time_intervals)
			outputs['p_total'] = zero_lists(self.time_intervals)
			outputs['p_cooling'] = zero_lists(self.time_intervals)
			outputs['v_computing'] = zero_lists(self.time_intervals)
			outputs['v_cooling'] = zero_lists(self.time_intervals)
			outputs['p_gen_pv_load'] = zero_lists(self.time_intervals)
			outputs['p_gen_wind_load'] = zero_lists(self.time_intervals)
			outputs['pcc_pin_load'] = zero_lists(self.time_intervals)
			outputs['pcc_pout_pv'] = zero_lists(self.time_intervals)
			outputs['pcc_pout_wind'] = zero_lists(self.time_intervals)

		for v in self.milp.variables():
			step_nr = None
			if not re.fullmatch('dummy', v.name[:-4]):
				step_nr = int(v.name[-3:])

			# Battery-related variables
			for b, bat in enumerate(self._batteries):
				name = bat.get('b_name')
				if re.fullmatch('b_energy_' + f'{name}', v.name[:-4]):
					outputs['b_energy_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_soc_' + f'{name}', v.name[:-4]):
					outputs['b_soc_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_pch_' + f'{name}', v.name[:-4]):
					outputs['b_pch_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_pdis_' + f'{name}', v.name[:-4]):
					outputs['b_pdis_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_pdis_load_' + f'{name}', v.name[:-4]):
					outputs['b_pdis_load_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('p_gen_wind_bat_' + f'{name}', v.name[:-4]):
					outputs['p_gen_wind_bat_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_pdis_grid_' + f'{name}', v.name[:-4]):
					outputs['b_pdis_grid_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('b_pch_grid_' + f'{name}', v.name[:-4]):
					outputs['b_pch_grid_' + f'{name}'][step_nr] = v.varValue
				elif re.fullmatch('p_gen_pv_bat_' + f'{name}', v.name[:-4]):
					outputs['p_gen_pv_bat_' + f'{name}'][step_nr] = v.varValue

			if re.fullmatch('pcc_pin', v.name[:-4]):
				outputs['pcc_pin'][step_nr] = v.varValue
			elif re.fullmatch('v_load', v.name[:-4]):
				outputs['v_load'][step_nr] = v.varValue
			elif re.fullmatch('p_total', v.name[:-4]):
				outputs['p_total'][step_nr] = v.varValue
			elif re.fullmatch('v_computing', v.name[:-4]):
				outputs['v_computing'][step_nr] = v.varValue
			elif re.fullmatch('v_cooling', v.name[:-4]):
				outputs['v_cooling'][step_nr] = v.varValue
			elif re.fullmatch('p_cooling', v.name[:-4]):
				outputs['p_cooling'][step_nr] = v.varValue
			elif re.fullmatch('pcc_pout', v.name[:-4]):
				outputs['pcc_pout'][step_nr] = v.varValue
			elif re.fullmatch('p_gen_pv_load', v.name[:-4]):
				outputs['p_gen_pv_load'][step_nr] = v.varValue
			elif re.fullmatch('p_gen_wind_load', v.name[:-4]):
				outputs['p_gen_wind_load'][step_nr] = v.varValue
			elif re.fullmatch('pcc_pin_load', v.name[:-4]):
				outputs['pcc_pin_load'][step_nr] = v.varValue
			elif re.fullmatch('pcc_pout_pv', v.name[:-4]):
				outputs['pcc_pout_pv'][step_nr] = v.varValue
			elif re.fullmatch('pcc_pout_wind', v.name[:-4]):
				outputs['pcc_pout_wind'][step_nr] = v.varValue

		return outputs


if __name__ == '__main__':
	parameters = {
		'horizon': 24,
		'delta_t': 60,
		'pcc_limit': 1E9,
		'buy_price': [0] * 24,
		'sell_tariff': [0] * 24,
		'p_load': [0] * 24,
		'p_gen_pv': [0] * 24,
		'p_gen_wind': [0] * 24,
		'int_co2': [0] * 24,
		'b_capacity': 51.8,
		'b_init_energy': 5,
		'b_soc_min': 0,
		'b_soc_max': 100,
		'b_pch_max': 10,
		'b_pdis_max': 10,
		'b_ch_eff': 1.0,
		'b_dis_eff': 1.0,
		't_ambient' : 300.15,
		'prosumer_id': 'Prosumer#1'
	}

	milp = MILP(backpack=parameters, solver='CBC', timeout=1E9, mipgap=0.01)
	milp.solve_milp()
	milp.generate_outputs()
 