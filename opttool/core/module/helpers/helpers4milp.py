import itertools
import math
import numpy as np


def zero_lists(*m_dims): # m_dims specifies the dimension of the nested list
	"""
	Creates nested lists with depth given by the number of m_dims passed. Bottom lists are filled with None.
	:param m_dims:
	:return:
	"""
	master_list = np.zeros(m_dims) # initializes master_list variable  as a array of 0 with the dimension given by (m_dims) 
								   # which means the nº of 0 in the array will be the dimension of m_dms

	# Note: it is necessary to convert to list when storing LpVariables.
	
	return master_list.tolist() # .tolist converts the array to a list


def time_intervals(horizon, delta_t, f='int'):
	"""
	OBJECTIVE : Retrieves the number of time steps within an horizon, provided the duration of those steps.
	:param horizon: -> lenght of the time period being considered
	:param delta_t: -> duration of each time step
	:param f: -> explica a função de arredondamento usada para calcular o number of time steps
	:return:
	"""
	if f == 'ceil':   # se f estiver definido para 'ceil', a função math.ceil arredonda f para o int ACIMA + próximo
		return math.ceil(horizon / delta_t)

	elif f == 'floor':  # se f estiver definido para 'floor', a função math.floor arredonda f para o int ABAIXO + próximo
		return math.floor(horizon / delta_t)

	else:  # função arredonda o resutado para o int + próximo (indiferente se acima ou abaixo)
		return int(horizon / delta_t)


# m_dim is the length of the list to be created as values of the dictionary
# designations is a variable-length parameter that contains the keys of the dictionary

def dict_zero_lists(m_dim, *designations):
	"""
	Returns a dictionary with the same keys as in designations and whose values are lists.
	If more than one key is passed, the dictionary will have nested keys.
	:type m_dim: int
	:type designations: list of str
	:rtype: dict of lists
	"""
	nested_dict_of_lists = {}

	# creates all possible combinations of keys in 'designations'
	for tuple_key in itertools.product(*designations):
		first = 0
		last = len(tuple_key) - 1

		for position, value in enumerate(tuple_key):
			if position == first:
				aux = nested_dict_of_lists

			if position == last:
				aux[value] = zero_lists(m_dim)
				continue

			if not aux.get(value):
				aux[value] = {}

			aux = aux[value]

	return nested_dict_of_lists


def dict_lists(prosumers, values_id):
	"""
	Returns a dictionary with the prosumers' IDs as keys and the specified lists of data as values.
	:param prosumers: dictionary with data from all prosumers
	:type prosumers: dict
	:param values_id: prosumers data that will be stored as the dictionary's value
	:type values_id: str
	:return: dictionary whose keys are prosumer IDs and the values are the specified lists in values_id
	:rtype: dict
	"""

	return {key: value.get(values_id) for key, value in prosumers.items()}


def iterator(*args):
	"""
	Function to return an iterator with the combinations of the lists or ranges passed in the arguments.
	:param args: lists or range with the value to iterate over
	:type args: Union[list, range]
	:return: iterator
	:rtype: itertools.product
	"""
	return itertools.product(*args)
