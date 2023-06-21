import matplotlib as plt
import matplotlib.colors as clrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import warnings


def color_fader(c1, c2, mix=0):
	clr1 = np.array(clrs.to_rgb(c1))
	clr2 = np.array(clrs.to_rgb(c2))
	return clrs.to_hex((1-mix) * clr1 + mix * clr2)


def plot_results(prosumers, outputs):
	"""
	Function to plot the relevant results from the optimization problem resolution
	:param prosumers: dictionary with the list of data dictionaries used as input for the two stage MILP procedure
	:type prosumers:dict
	:param outputs: list with the first stage output dictionaries of each prosumer
	:type outputs: list
	:return: None
	:rtype: None
	"""
	# Silence warnings (for setting colors)
	warnings.filterwarnings("ignore")

	# Initialize auxiliary variables
	horizon = prosumers.get('horizon')
	data = prosumers.get('prosumers')
	time_series = range(horizon)
	offset_series = np.arange(1/3, horizon + 1/3, 1)
	offset_series_2 = np.arange(2/3, horizon + 2/3, 1)
	nr_prosumers = 1
	name_prosumers = [prosumer.get('prosumer_id') for prosumer in data]

	# Calculate individual net load
	df_data = pd.DataFrame(data)
	map_net_load = map(lambda load, gen_pv, gen_wind: [lo - ph - wi for lo, ph, wi in zip(load, gen_pv, gen_wind)], df_data['p_load'], df_data['p_gen_pv'], df_data['p_gen_wind'])
	df_data['net_load'] = pd.Series(map_net_load)

	# Matplotlib settings
	matplotlib.rcParams.update({'font.size': 15})

	# **************************************************************************************************************
	#        PLOTS
	# **************************************************************************************************************
	fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[30, 2.5 + 2.5 * nr_prosumers * 2], sharex=True)
	vertical = 0  # Vertical relative position of the plot

	name = name_prosumers[0]
	# **************************************************************************************************************
	#        PLOT - Indivdual net loads and prices
	# **************************************************************************************************************
	ax = axes[vertical]

	# 1) Net load
	prosumer_data = df_data[df_data['prosumer_id'] == name]
	prosumer_net_load = pd.Series(prosumer_data['net_load'].iloc[0])

	prosumer_net_load.plot(y=f'{name} net load', kind='bar', width=1.0, align='edge', edgecolor='tomato',
						   color='goldenrod', alpha=0.7, ax=ax)
	ax.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='goldenrod', edgecolor='tomato', alpha=0.7, label=f'{name} net load'))

	# 2) Second axis for prices and tariffs
	ax2 = ax.twinx()

	prosumer_tariffs = pd.Series(prosumer_data['sell_tariff'].iloc[0])
	prosumer_prices = pd.Series(prosumer_data['buy_price'].iloc[0])

	ax2.scatter(offset_series, prosumer_prices, label='Market prices', color='darkred', alpha=0.9, marker=7, s=100.0)
	ax2.scatter(offset_series, prosumer_tariffs, label='Feedin tariffs', color='darkgreen', alpha=0.9, marker=6, s=100.0)

	fp_handles, _ = ax2.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax.set_ylabel('kW')
	ax2.set_ylabel('€/kWh')
	ax.grid(which='major', axis='x', linestyle='--')

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Indivdual BESS schedules
	# **************************************************************************************************************
	vertical += 1
	ax2 = axes[vertical]

	prosumer_objval1 = outputs.get('obj_value')
	prosumer_b_energy1 = pd.Series(outputs.get('b_energy'))
	prosumer_b_energy1.index = time_series
	prosumer_b_pch = pd.Series(outputs.get('b_pch'))
	prosumer_b_pdis = pd.Series(outputs.get('b_pdis'))

	ax2.bar(time_series, prosumer_b_energy1, width=1/3, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax2.bar(offset_series, prosumer_b_pch, width=1/3, align='edge', edgecolor='silver', color='forestgreen',
			alpha=1.0)
	ax2.bar(offset_series_2, prosumer_b_pdis, width=1/3, align='edge', edgecolor='silver', color='darkred',
			alpha=1.0)

	ax2.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} BESS energy -- Obj val: {round(prosumer_objval1, 2)}'))
	handles.append(mpatches.Patch(color='forestgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} BESS Charge'))
	handles.append(mpatches.Patch(color='darkred', edgecolor='silver', alpha=1.0,
								  label=f'{name} BESS Discharge'))

	# -- Tweak plot parameters
	ax2.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax2.set_ylabel('kW')
	ax2.grid(which='major', axis='x', linestyle='--')

	box = ax2.get_position()
	ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Renewable Generation and Load
	# **************************************************************************************************************
	vertical += 1
	ax3 = axes[vertical]

	load = pd.Series(prosumer_data['p_load'].iloc[0])
	pv = pd.Series(prosumer_data['p_gen_pv'].iloc[0])
	wind = pd.Series(prosumer_data['p_gen_wind'].iloc[0])


	ax3.bar(time_series, pv, width=1/3, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax3.bar(offset_series, wind, width=1/3, align='edge', edgecolor='silver', color='forestgreen',
			alpha=1.0)
	ax3.bar(offset_series_2, load, width=1/3, align='edge', edgecolor='silver', color='darkred',
			alpha=1.0)

	ax3.axhline(color='darkred')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV generation'))
	handles.append(mpatches.Patch(color='forestgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind generation'))
	handles.append(mpatches.Patch(color='darkred', edgecolor='silver', alpha=1.0,
								  label=f'{name} Load'))

	# -- Tweak plot parameters
	ax3.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax3.set_ylabel('kW')
	ax3.grid(which='major', axis='x', linestyle='--')

	box = ax3.get_position()
	ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])


	# **************************************************************************************************************
	#        PLOT - PCC Exchange
	# **************************************************************************************************************
	vertical += 1
	ax4 = axes[vertical]

	prosumer_pcc_pin = pd.Series(outputs.get('pcc_pin'))
	prosumer_pcc_pout = pd.Series(outputs.get('pcc_pout'))

	ax4.bar(time_series, prosumer_pcc_pin, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax4.bar(offset_series, prosumer_pcc_pout, width=1 / 2, align='edge', edgecolor='silver', color='forestgreen',
			alpha=1.0)

	ax4.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} PCC Import'))
	handles.append(mpatches.Patch(color='forestgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} PCC Export'))

	ax5 = ax4.twinx()

	co2_intensity = pd.Series(prosumer_data['int_co2'].iloc[0])
	ax5.scatter(offset_series, co2_intensity, label='CO2 intensity (g/kWh)', color='darkred', alpha=0.9, marker=7, s=100.0)

	fp_handles, _ = ax4.get_legend_handles_labels()
	handles.extend(fp_handles)
	handles.append(mpatches.Patch(color='darkred', edgecolor='silver', alpha=1.0,
								  label='CO2 intensity (g/kWh)'))

	# -- Tweak plot parameters
	ax4.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax4.set_ylabel('kW')
	ax4.grid(which='major', axis='x', linestyle='--')
	ax5.set_ylabel('g/kWh')

	box = ax4.get_position()
	ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	plt.xlim(0, horizon)
	now = time.strftime('%d%b%y__%H_%M_%S', time.localtime(time.time()))
	plt.savefig(os.path.abspath(os.path.join(__file__, '..', '..', 'results', f'{now}.png')))
