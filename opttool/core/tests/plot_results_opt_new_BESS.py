import matplotlib
import matplotlib.colors as clrs
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
	offset_series = np.arange(1/2, horizon + 1/2, 1)
	name_prosumers = [prosumer.get('prosumer_id') for prosumer in data]
	nbat = len(prosumers.get('prosumers')[0].get('batteries'))
	nbat_list = range(1, nbat + 1)

	# Calculate individual net load
	df_data = pd.DataFrame(data)

	# Matplotlib settings
	matplotlib.rcParams.update({'font.size': 8})

	# **************************************************************************************************************
	#        PLOTS
	# **************************************************************************************************************
	fig, axes = plt.subplots(nrows=9, ncols=1, figsize=[30, 10], sharex=True)
	vertical = 0  # Vertical relative position of the plot

	name = name_prosumers[0]
	prosumer_data = df_data[df_data['prosumer_id'] == name]
	prosumer_b_pch = pd.Series(outputs.get('b_pch'))
	prosumer_b_pdis = pd.Series(outputs.get('b_pdis'))
	load = pd.Series(prosumer_data['p_load'].iloc[0])
	pv = pd.Series(prosumer_data['p_gen_pv'].iloc[0])
	wind = pd.Series(prosumer_data['p_gen_wind'].iloc[0])

	# **************************************************************************************************************
	#        PLOT - Indivdual net loads and prices
	# **************************************************************************************************************
	ax = axes[vertical]

	co2_intensity = pd.Series(prosumer_data['int_co2'].iloc[0])
	co2_intensity.plot(ax=ax, x=time_series, y=co2_intensity, style='.--', label='CO2 intensity (g/kWh)', color='k', alpha=0.9, marker='s')

	ax.axhline(color='k')

	handles = list()
	handles.append(mlines.Line2D(xdata=[],ydata=[], color='k', alpha=0.7, label=f'CO2 intensity (g/kWh)', marker='s'))

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
	ax.set_ylabel('g/kWh')
	ax2.set_ylabel('€/kWh')
	ax.grid(which='major', axis='x', linestyle='--')

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


	# **************************************************************************************************************
	#        PLOT - Load Balance
	# **************************************************************************************************************
	vertical += 1
	ax6 = axes[vertical]

	b_pdis_load = pd.Series(outputs.get('b_pdis_load'))
	p_gen_pv_load = pd.Series(outputs.get('p_gen_pv_load'))
	p_gen_wind_load = pd.Series(outputs.get('p_gen_wind_load'))
	pcc_pin_load = pd.Series(outputs.get('pcc_pin_load'))

	ax6.bar(time_series, load, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax6.bar(offset_series, p_gen_pv_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')
	ax6.bar(offset_series, p_gen_wind_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_pv_load, color='g')
	sum = p_gen_pv_load + p_gen_wind_load

	k = 0
	for b in nbat_list:
		dis_load = pd.Series(outputs.get('b_pdis_load_' + f'{b:02d}'))
		ax6.bar(offset_series, dis_load, width=1 / 2, align='edge', edgecolor='silver',
				alpha=1.0, bottom=sum + k, color='y')
		k += dis_load
	ax6.bar(offset_series, pcc_pin_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_pv_load+p_gen_wind_load+k, color='k')

	ax6.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Load'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Load'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Load'))
	handles.append(mpatches.Patch(color='b', edgecolor='silver', alpha=1.0,
								  label=f'{name} Bat_Load'))
	handles.append(mpatches.Patch(color='k', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid_Load'))

	fp_handles, _ = ax6.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax6.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax6.set_ylabel('kW')
	ax6.grid(which='major', axis='x', linestyle='--')

	box = ax6.get_position()
	ax6.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - PV Balance
	# **************************************************************************************************************
	vertical += 1
	ax7 = axes[vertical]

	pcc_pout_pv = pd.Series(outputs.get('pcc_pout_pv'))
	p_gen_pv_load = pd.Series(outputs.get('p_gen_pv_load'))
	p_gen_pv_bat = pd.Series(outputs.get('p_gen_pv_bat'))

	ax7.bar(time_series, pv, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax7.bar(offset_series, p_gen_pv_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')

	k = 0
	for b in nbat_list:
		pv_bat = pd.Series(outputs.get('p_gen_pv_bat_' + f'{b:02d}'))
		ax7.bar(offset_series, pv_bat, width=1 / 2, align='edge', edgecolor='silver',
				alpha=1.0, bottom=p_gen_pv_load + k, color='y')
		k += pv_bat
	ax7.bar(offset_series, pcc_pout_pv, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_pv_load+k, color='b')

	ax7.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Load'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Bat'))
	handles.append(mpatches.Patch(color='b', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Grid'))

	fp_handles, _ = ax7.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax7.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax7.set_ylabel('kW')
	ax7.grid(which='major', axis='x', linestyle='--')

	box = ax7.get_position()
	ax7.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Wind Balance
	# **************************************************************************************************************
	vertical += 1
	ax8 = axes[vertical]

	pcc_pout_wind = pd.Series(outputs.get('pcc_pout_wind'))
	p_gen_wind_load = pd.Series(outputs.get('p_gen_wind_load'))
	p_gen_wind_bat = pd.Series(outputs.get('p_gen_wind_bat'))

	ax8.bar(time_series, wind, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax8.bar(offset_series, p_gen_wind_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')

	k = 0
	for b in nbat_list:
		wind_bat = pd.Series(outputs.get('p_gen_wind_bat_' + f'{b:02d}'))
		ax8.bar(offset_series, wind_bat, width=1 / 2, align='edge', edgecolor='silver',
				alpha=1.0, bottom=p_gen_wind_load + k, color='y')
		k += wind_bat
	ax8.bar(offset_series, pcc_pout_wind, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_wind_load + k, color='b')

	ax8.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Load'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Bat'))
	handles.append(mpatches.Patch(color='b', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Grid'))

	fp_handles, _ = ax8.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax8.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax8.set_ylabel('kW')
	ax8.grid(which='major', axis='x', linestyle='--')

	box = ax8.get_position()
	ax8.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Battery Energy
	# **************************************************************************************************************
	vertical += 1
	ax9 = axes[vertical]

	b_energy = pd.Series(outputs.get('b_energy'))

	k = 0
	for b in nbat_list:
		b_energy = pd.Series(outputs.get('b_energy_' + f'{b:02d}'))
		ax9.bar(offset_series, b_energy, width=1 / 2, align='edge', edgecolor='silver',
				alpha=1.0, bottom=k)
		k += b_energy

	ax9.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Battery Energy'))

	fp_handles, _ = ax9.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax9.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax9.set_ylabel('kWh')
	ax9.grid(which='major', axis='x', linestyle='--')

	box = ax9.get_position()
	ax9.set_position([box.x0, box.y0, box.width * 0.8, box.height])


	"""	# **************************************************************************************************************
	#        PLOT - Battery Charge Balance
	# **************************************************************************************************************
	vertical += 1
	ax9 = axes[vertical]

	b_pch_grid = pd.Series(outputs.get('b_pch_grid'))
	p_gen_pv_bat = pd.Series(outputs.get('p_gen_pv_bat'))
	p_gen_wind_bat = pd.Series(outputs.get('p_gen_wind_bat'))

	ax9.bar(time_series, prosumer_b_pch, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax9.bar(offset_series, p_gen_pv_bat, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')
	ax9.bar(offset_series, p_gen_wind_bat, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_pv_bat, color='g')
	ax9.bar(offset_series, b_pch_grid, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, bottom=p_gen_pv_bat + p_gen_wind_bat, color='b')

	ax9.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Charge'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Bat'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Bat'))
	handles.append(mpatches.Patch(color='b', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid_Bat'))

	fp_handles, _ = ax9.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax9.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax9.set_ylabel('kW')
	ax9.grid(which='major', axis='x', linestyle='--')

	box = ax9.get_position()
	ax9.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Battery Discharge Balance
	# **************************************************************************************************************
	vertical += 1
	ax10 = axes[vertical]

	b_pdis_load = pd.Series(outputs.get('b_pdis_load'))
	b_pdis_grid = pd.Series(outputs.get('b_pdis_grid'))

	ax10.bar(time_series, prosumer_b_pdis, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax10.bar(offset_series, b_pdis_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')
	ax10.bar(offset_series, b_pdis_grid, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='g', bottom=b_pdis_load)

	ax10.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Discharge'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} Bat_Load'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Bat_Grid'))

	fp_handles, _ = ax10.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax10.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax10.set_ylabel('kW')
	ax10.grid(which='major', axis='x', linestyle='--')

	box = ax10.get_position()
	ax10.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Grid Export
	# **************************************************************************************************************
	vertical += 1
	ax10 = axes[vertical]

	pcc_pout = pd.Series(outputs.get('pcc_pout'))
	pcc_pout_pv = pd.Series(outputs.get('pcc_pout_pv'))
	pcc_pout_wind = pd.Series(outputs.get('pcc_pout_wind'))
	b_pdis_grid = pd.Series(outputs.get('b_pdis_grid'))

	ax10.bar(time_series, pcc_pout, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax10.bar(offset_series, pcc_pout_pv, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')
	ax10.bar(offset_series, pcc_pout_wind, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='g', bottom=pcc_pout_pv)
	ax10.bar(offset_series, b_pdis_grid, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='b', bottom=pcc_pout_pv+pcc_pout_wind)

	ax10.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid Export'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} PV_Grid'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Wind_Grid'))
	handles.append(mpatches.Patch(color='b', edgecolor='silver', alpha=1.0,
								  label=f'{name} Bat_Grid'))

	fp_handles, _ = ax10.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax10.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax10.set_ylabel('kW')
	ax10.grid(which='major', axis='x', linestyle='--')

	box = ax10.get_position()
	ax10.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# **************************************************************************************************************
	#        PLOT - Grid Import
	# **************************************************************************************************************
	vertical += 1
	ax10 = axes[vertical]

	pcc_pin = pd.Series(outputs.get('pcc_pin'))
	pcc_pin_load = pd.Series(outputs.get('pcc_pin_load'))
	b_pch_grid = pd.Series(outputs.get('b_pch_grid'))

	ax10.bar(time_series, pcc_pin, width=1 / 2, align='edge', edgecolor='silver', color='lightgreen',
			alpha=1.0)
	ax10.bar(offset_series, pcc_pin_load, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='r')
	ax10.bar(offset_series, b_pch_grid, width=1 / 2, align='edge', edgecolor='silver',
			alpha=1.0, color='g', bottom=pcc_pin_load)

	ax10.axhline(color='k')

	handles = list()
	handles.append(mpatches.Patch(color='lightgreen', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid Import'))
	handles.append(mpatches.Patch(color='r', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid_Load'))
	handles.append(mpatches.Patch(color='g', edgecolor='silver', alpha=1.0,
								  label=f'{name} Grid_Bat'))

	fp_handles, _ = ax10.get_legend_handles_labels()
	handles.extend(fp_handles)

	# -- Tweak plot parameters
	ax10.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True)
	ax10.set_ylabel('kW')
	ax10.grid(which='major', axis='x', linestyle='--')

	box = ax10.get_position()
	ax10.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	"""
	plt.xlim(0, horizon)
	plt.xticks(np.arange(0, 24, 1.0))
	matplotlib.rcParams.update({'font.size': 4})
	now = time.strftime('%d%b%y__%H_%M_%S', time.localtime(time.time()))
	plt.savefig(os.path.abspath(os.path.join(__file__, '..', '..', 'results', f'{now}.png')))

