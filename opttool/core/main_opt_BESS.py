"""
Main procedure for computing the energy optimization of the HPC
"""
import datetime as dt
import json
import os
import time
from loguru import logger
from module.MILP_new_BESS_load_thermal import MILP
from module.helpers.helpers4milp import zero_lists
from module.helpers.otimize import optimize_epso
from tests.plot_results_opt import plot_results


def default_outputs_opt(time_intervals, prosumer):
	"""
	Returns a default outputs' structure when stage 2 cannot be performed due to no-optimal stage 1 results
	:param time_intervals: number of optimization steps
	:type time_intervals: int
	:param set_prosumers: list with prosumers' id
	:type set_prosumers: list
	:return: dictionary with idle set points for all prosumers
	:rtype: dict
	"""

	# creates a dictionary called outputs and initialize its values using zero_lists (which returns a list of 0)

	outputs = {}
	outputs['prosumer_id'] = prosumer
	outputs['obj_value'] = 0.0
	outputs['milp_status'] = 'Not Solved'
	outputs['b_energy'] = zero_lists(time_intervals)
	outputs['b_soc'] = zero_lists(time_intervals)
	outputs['b_pch'] = zero_lists(time_intervals)
	outputs['b_pdis'] = zero_lists(time_intervals)
	# outputs['b_pbin'] = zero_lists(time_intervals)
	outputs['pcc_pin'] = zero_lists(time_intervals)
	outputs['pcc_pout'] = zero_lists(time_intervals)
	# outputs['pcc_pbin'] = zero_lists(time_intervals)

	return outputs


def set_and_run(data):
	"""
	All-in-one function, parallelization friendly, for setting up, solving and generating the output of MILP.
	:param data: dictionary with the data required for MILP
	:type data: dict
	:return: dictionary with the outputs of MILP
	:rtype: dict
	"""
	milp = MILP(backpack=data, solver='CBC', timeout=1E9, mipgap=0.01)
	milp.solve_milp()

	return milp.generate_outputs()

def set_and_run_epso(data):

	optimize_epso(data)

	outputs={}
	return outputs

def optimization(parameters):
	"""
	Main function for performing the MILP procedure.
	:param parameters: dictionary with the list of data dictionaries of the prosumer, ready for MILP
	:type parameters: dict
	:return: dictionary with relevant MILP outputs
	:rtype: dict
	"""

	# Paths
	ROOT_PATH = os.path.abspath(os.path.join(__file__, '..'))
	LOGS_PATH = os.path.join(ROOT_PATH, 'logs', 'optimization.log')

	# Logger setup
	logger_format = '{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name: ^20} | {function: ^20} | {line: >4} | {message}'
	logger.add(LOGS_PATH, format=logger_format, rotation="25 MB", level='DEBUG', backtrace=True)

	# First messages
	#logger.info('*' * 100)
	#logger.info(f'MILP procedure START: {dt.datetime.now().isoformat(" ", "seconds")}')

	# Add step and horizon information to each prosumer
	horizon = parameters.get('horizon')  # optimization horizon in hours
	step = parameters.get('delta_t')  # optimization step in hours

	prosumer = parameters.get('prosumers')[0]
	prosumer['horizon'] = horizon
	prosumer['delta_t'] = step

	# Set objective function
	prosumer['obj_function'] = parameters.get('obj_function')  # options: min_co2, min_costs

	# Run optimization
	logger.info(f' -- MILP START: {dt.datetime.now().isoformat(" ", "seconds")}')
	t1 = time.time()
	outputs = set_and_run_epso(prosumer)
	logger.info(f' -- MILP END: {dt.datetime.now().isoformat(" ", "seconds")} ({time.time() - t1:.3f}s)')

	# Check if MILP was optimally solved...
	stage1_all_ok = True if outputs.get('milp_status') == 'Optimal' else False

	# ... if not, return error
	if not stage1_all_ok:
		nr_steps = int(horizon * 60 / step)
		logger.error(f'MILP was not optimally solved.')
		logger.error(f'Returning idle set points.')
		logger.info('*' * 100)
		outputs = default_outputs_opt(nr_steps, prosumer.get('prosumer_id'))

	# Save JSON files of all outputs
	now = time.strftime('%d%b%y__%H_%M_%S', time.localtime(time.time()))

	with open(os.path.abspath(os.path.join(RESULTS_PATH, f'opt_{now}.json')), 'w') as outfile:
		json.dump(outputs, outfile, indent=5, sort_keys=True)

	# Return outputs of MILP
	return outputs


if __name__ == '__main__':

	INPUT_PATH = os.path.abspath(os.path.join(__file__, '..', 'examples/input_parameters.json'))
	with open(INPUT_PATH) as json_file:
		params = json.load(json_file)

	outputs = optimization(params)
	print(outputs)

	# -- Uncomment for saving a plot with the results
	plot_results(params, outputs)







