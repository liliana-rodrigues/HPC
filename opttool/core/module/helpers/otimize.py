import sys
import copy
import random
import numpy as np
from time import time
from loguru import logger
from module.helpers.epso import Particle

def optimize_epso(backpack):
    """
    Optimization call procedure
    :param backpack:
    :return:
    """

    # Output structure:
    outputs = {"stage_1": None, "stage_2": None}

    msg_ = "Initializing Optimization ..."
    logger.debug(msg_)
    random.seed(12345)
    t0 = time()

    #  todo: epso parameters in json file
    input_opt = {'max_gen': 100,
                 'num_particles': 25,
                 'mutation_rate': 0.4,
                 'communication_probability': 0.8}

    # decision variable bounds
    x_min, x_max = [], []  # search space lower and upper bound
    for i in range(0, int(backpack.get('horizon'))):  # todo: horizon * nr_bat
        x_min.append(-300) # minus max power discharge battery
        x_max.append(300) # max power charge battery

    # establish the swarm
    swarm = []
    g_best_pos = []
    g_best_pos_fit = sys.float_info.max
    for i in range(0, input_opt.get('num_particles')):
        swarm.append(Particle(x_min, x_max))
        swarm[i].evaluate(f, backpack, input_opt)

        # determine if current particle is the global best
        if swarm[i].fit_pos_i < g_best_pos_fit:
            g_best_pos = copy.deepcopy(swarm[i].position_i)
            g_best_pos_fit = swarm[i].fit_pos_i
    logger.debug(f"{msg_} ... Ok! ({time() - t0:.2f}s)")

    msg_ = "Running EPSO ..."
    logger.debug(msg_)
    g_best_pos, g_best_pos_fit, g_best_pos_fit_historical = run_epso(swarm, input_opt, f, backpack, g_best_pos, g_best_pos_fit)
    logger.debug(f"{msg_} ... Ok! ({time() - t0:.2f}s)")

    # Generate outputs
    outputs = generate_outputs(data_opt, milp_out_final, input_opt.get('base_category'), g_best_pos, g_best_pos_fit)
    logger.debug(f"{msg_} ... Ok! ({time() - t0:.2f}s)")

    return outputs


def f(v, data, input_opt):
    """
    Fitness function of genetic algorithm
    :param v: charging / discharging power
    :param data:
    :param input_opt:
    :return:
    """
    # arredondamento das soluções 
    v = list(np.round(v,2))
    # Assess and update v (alphas)
    data_opt = data.copy()

    # Add penalties for violated constraints
    pen, objective_1, objective_2 = check_constraints(v, data_opt)

    # Update log message
    success = 'Y' if pen < 1E-50 else 'N'
    print(f'\n[{success}][{v}] Pen: {pen}')

    # Maximize profits (Upper-Level objective function)
    w1 = 0.5
    w2 = 0.5
    obj_function = w1*objective_1 + w2*objective_2
    return np.sum(obj_function) + pen


def run_epso(swarm, input_opt, f, data, g_best_pos, g_best_pos_fit):
    """
    Main function to run iterative EPSO algorithm
    :param swarm:
    :param input_opt:
    :param f:
    :param data:
    :param g_best_pos:
    :param g_best_pos_fit:
    :return:
    """

    i = 0
    g_best_pos_fit_hist = []
    while i < input_opt.get('max_gen'):
        # create a copy swarm
        copy_swarm = copy.deepcopy(swarm)

        # cycle through swarms and mutate weights, update velocities and position, and update fitness
        for j in range(0, int(input_opt.get('num_particles'))):
            swarm[j].update_velocity(input_opt.get('communication_probability'), g_best_pos)
            swarm[j].update_position()
            swarm[j].evaluate(f, data, input_opt)
            copy_swarm[j].mutate_weights(input_opt.get('mutation_rate'))
            copy_swarm[j].update_velocity(input_opt.get('communication_probability'), g_best_pos)
            copy_swarm[j].update_position()
            copy_swarm[j].evaluate(f, data, input_opt)

        # cycle through swarms and update particles and global best position
        for j in range(0, int(input_opt.get('num_particles'))):
            if copy_swarm[j].fit_pos_i < swarm[j].fit_pos_i:
                swarm[j] = copy_swarm[j]

            # determine if current particle is the global best position
            if swarm[j].fit_pos_i < g_best_pos_fit:
                g_best_pos = copy.deepcopy(swarm[j].position_i)
                g_best_pos_fit = swarm[j].fit_pos_i
        i += 1
        g_best_pos_fit_hist.append(g_best_pos_fit)

    return g_best_pos, g_best_pos_fit, g_best_pos_fit_hist


def check_constraints(v, data):
    """
    Compute penalties for violated constraints
    :param v:
    :param alpha:
    :param data:
    :param milp_out:
    :return:
    """

    # initialize penalties
    pen = 0

    # get parameters
    buy_price = data.get('buy_price')
    sell_tariff = data.get('sell_tariff')

                      #    OBJECTIVE FUNCTION 1 FORMULATION    
   
    # constraint 1 : battery energy update  (initial energy + potencia afetada pelo rendimento)
    initial_energy = data.get('batteries')[0].get('b_init_energy')
    b_energy = [0]*data.get('horizon')
    b_update = [0]*data.get('horizon')

    if(v[0]>0):
        b_update[0] =  v[0]*data.get('batteries')[0].get('b_ch_eff')
        b_energy[0] = initial_energy + b_update[0]
    elif(v[0]<0):
        b_update[0] = v[0]/data.get('batteries')[0].get('b_dis_eff')
        b_energy[0] = initial_energy + b_update[0]
    else:
        b_energy[0] = initial_energy

    for t in range(1, data.get('horizon')):
        if v[t] > 0: 
            b_update[t] = v[t]*data.get('batteries')[0].get('b_ch_eff')
            b_energy[t] = b_energy[t-1] + b_update[t]
        else:
            b_update[t] = v[t]/data.get('batteries')[0].get('b_dis_eff')
            b_energy[t] = b_energy[t-1] + b_update[t]


    # constraint 2 : SoC calculation 
    b_state_of_charge = [b_energy[t]*100/data.get('batteries')[0].get('b_capacity') for t in range(0, data.get('horizon'))]
    print('\nSoC:', b_state_of_charge)
    if min(b_state_of_charge) < data.get('batteries')[0].get('b_soc_min'):
        print('SoC min')
        pen += 1E+4 
    if max(b_state_of_charge) > data.get('batteries')[0].get('b_soc_max'):
        pen += 1E+4
        print('SoC max')


    # constraint 3 : battery power limits
    b_pch_max = data.get('batteries')[0].get('b_pch_max')  # max charge power
    b_pdis_max = data.get('batteries')[0].get('b_pdis_max') # max discharge power

    for t in range(1, data.get('horizon')):
        if (v[t] > 0 and v[t] > b_pch_max) or (v[t] < 0 and v[t] < -b_pdis_max):
            print('bat power limit')
            pen += 1E+4
    

    # constraint 4 : power balance
    net_load = [0]*data.get('horizon')
    for t in range(1, data.get('horizon')):
        net_load[t] = data.get('p_load')[t] - data.get('p_gen_pv')[t] - data.get('p_gen_wind')[t] + v[t]


    # constraint 5 : grid power limits (pcc_pin e pcc_pout têm de ser menores que a potencia limite dos parametros)
    pcc_pin = [x if x > 0 else 0 for x in net_load] 
    pcc_pout = [x if x < 0 else 0 for x in net_load] 

    if max(pcc_pout) > data.get('pcc_limit'):
        print('export limit')
        pen += 1E+4
    if max(pcc_pin) > data.get('pcc_limit'):
        print('import limit')
        pen += 1E+4


    # constraint 6 : do not simultaneously import/export power from/to grid
    for t in range(1, data.get('horizon')):
        if pcc_pin[t] > 0 and pcc_pout[t] > 0:
            print('import/export')
            pen += 1E+4
    
                      #    OBJECTIVE FUNCTION 2 FORMULATION 
        # parameters from [operational planning using 2nd LB] 
    u = 3.2
    sigma = 0.88

    e = 15 #  battery´s power consumption per 100 kms (kWh)
    q_bc = 45 # EV battery´s rated capacity (kWh)  

    y_retire = 8 # battery´s served years at the time of its retirement

    q0 = 0.9964 # initial capacity retention rate of battery
    x = 0.0067
    tau = 0.5

        # parameters from [2nd LB article]
    a_rate = 1.07e3 # capacity of 2nd LB (Ah) 
    e_sl = 400 # SL-BESS rated energy capacity (kWh)

    # daily driving mileage
    e_d = 1.61*np.exp(u + ((sigma*sigma)/2))
    print('\nDaily driving mileage:', e_d) 
    
    #annual charging/discharging cycle number
    n_battery = (365*e_d*e)/(100*q_bc)
    print('\nAnnual charging/discharging cycles no:', n_battery)

    # cycles done in 1st life
    n_retire = y_retire*n_battery
    print('\nCyles done in 1st life:', n_retire)

    # maximum life cycle of a battery -> power function
    n_scrap = ((q0-0.6)/x) ** (1/tau)
    print('\nMaximum life cycle of a battery:', n_scrap)

    # cycles that can still do in 2nd life
    n_sec = n_scrap - n_retire
    print('\nCycles that can still do in 2nd life:', n_sec)

    # remaining capacity when CRR is degraded to treshold
    a_sl = a_rate*(q0 - x*((n_retire)**tau) - 0.6)
    print('\nRemaining capacity of 2nd LB when CRR is degraded to treshold:', a_sl)

    # average depreciated capacity of 2nd LB due to a full charging/discharging cycle
    a_fade = a_sl/n_sec
    print('\nAverage depreciated capacity of 2nd LB due to a full ch/disch cycle:', a_fade)

    # energy capacity at the end of the cycle 
    dk_half = [np.abs(x)/e_sl for x in b_update] 
    print('\nEnergy capacity at the end of the cycle:', dk_half)

    # bat equivalent to 100% DoD of discharging cycle number
    kp = 0.8 # constant value
    n_eq_day = [np.sum(0.5*x)**kp for x in dk_half]
    print('\nBat equivalent to 100% DoD of discharging cycle number:', n_eq_day)

       # parameters from [battery degradation cost - final]
    bat_cost = 120 # battery cost ($/kWh) -> from reference [x]
    dod = [100-x for x in b_state_of_charge] # depth-of-discharge
    print('\nDepth-of-discharge:', dod)
    avg_dod = np.sum(dod)/24 # average dod of battery
    print('\nAverage DoD:', avg_dod)

    e_ltp = (avg_dod/100) * n_sec * e_sl
    print('\nEnergy lifetime through BESS:', e_ltp)

    c_deg = bat_cost/e_ltp
    print('\nDegradation cost of the battery:', c_deg)

    total_deg_cost = [c_deg*np.abs(x) for x in b_update]
    print('\nTotal degradation cost:', total_deg_cost)
    daily_cost_deg = np.sum(total_deg_cost)
    print('\nTotal cost of degradation per day:', daily_cost_deg)

    #c_unit = 137 # battery cost per kWh ($/kWh)
    #c_labor = 168 # battery replacement labor cost 
    #CC = c_unit * q_bc 
    #SV = CC*0.6 # salvage value
    #avg_dod = np.sum(dod)/24 
    #c_deg = ((c_unit*q_bc)+c_labor-SV)/((n_sec+n_retire)+q_bc+avg_dod)


    # objective 1 function : minimize energy costs (maximize profit) -> in $
    objective_1= sum([buy_price[t] * pcc_pin[t] - sell_tariff[t] * pcc_pout[t] for t in range(0, data.get('horizon'))])

    # objective function 2 : minimize bat depreciation due to charging/discharging action 
    #objective_2 = [x * a_fade for x in n_eq_day]

    #objective function 2 : minimize daily cost degradation 
    objective_2 = daily_cost_deg.copy()

    return pen, objective_1, objective_2
    

def generate_outputs(data, milp_out, base_category, g_best_pos, g_best_pos_fit):
    """
    Generate final output structure to keep structure of previous milp version
    :param data:
    :param milp_out:
    :param base_category:
    :param g_best_pos:
    :param g_best_pos_fit:
    :return:
    """

    # Get initial data
    horizon = data.get('horizon').get('nr_hours_in_horizon')
    delta_t = data.get('horizon').get('delta_t') / 60
    tariff_l = data.get('tariffs').get('T1').get('value')
    tariff_r = data.get('tariffs').get('PR').get('value')
    power_l = data.get('tariffs').get('T1').get('max_power')
    power_r = data.get('tariffs').get('PR').get('max_power')
    market_prices = data.get('timeseries').get('market_price')
    occupancy_l = data.get('timeseries').get('occupancy_slow')
    occupancy_r = data.get('timeseries').get('occupancy_fast')
    occupancy = [o_l + o_r for o_l, o_r in zip(occupancy_l, occupancy_r)]
    ev_load_l = [occupancy_l[t] * power_l for t in range(int(horizon))]
    ev_load_r = [occupancy_r[t] * power_r for t in range(int(horizon))]
    ev_load = [ev_load_l[t] + ev_load_r[t] for t in range(int(horizon))]

    # Additional output variables
    zero = [0 for _ in range(int(horizon))]
    alpha_int = [np.floor(x) for x in g_best_pos]
    alpha = [1 - base_category * x for x in alpha_int]
    alpha_int_1 = [1 if x == 1 else 0 for x in alpha_int]
    alpha_int_2 = [1 if x == 2 else 0 for x in alpha_int]
    alpha_int_3 = [1 if x == 3 else 0 for x in alpha_int]
    theta = [1 if x > 0 else 0 for x in alpha_int]
    ev_load_delta = [x - y for x, y in zip(milp_out.get('new_ev_load'), ev_load)]
    tariff_l_list = [tariff_l for _ in range(int(horizon))]
    tariff_r_list = [tariff_r for _ in range(int(horizon))]
    tariffs_w_alpha_l = [x * y for x, y in zip(tariff_l_list, alpha)]
    tariffs_w_alpha_r = [x * y for x, y in zip(tariff_r_list, alpha)]
    load_delta_bin = [1 if x > 0 else 0 for x in ev_load_delta]
    forecasted_profits = calc_profit_occupancy(tariff_l_list, tariff_r_list, occupancy_l, occupancy_r, delta_t,
                                               market_prices, ev_load)
    optimal_profits = calc_profit_occupancy(tariff_l_list, tariff_r_list, milp_out.get('new_occupancy_l'),
                                            milp_out.get('new_occupancy_r'), delta_t, market_prices,
                                            milp_out.get('new_ev_load'))
    profits_w_alpha = calc_profit_occupancy(tariffs_w_alpha_l, tariffs_w_alpha_r, milp_out.get('new_occupancy_l'),
                                            milp_out.get('new_occupancy_r'), delta_t, market_prices,
                                            milp_out.get('new_ev_load'))
    extra_profits = [x - y for x, y in zip(optimal_profits, forecasted_profits)]
    extra_profits_bin = [1 if x > 0 else 0 for x in extra_profits]
    extra_profits_w_alpha = [x - y for x, y in zip(profits_w_alpha, forecasted_profits)]
    forecasted_ev_costs = sum([(tariff_l * x + tariff_r * y) * delta_t * 60 for x, y in zip(occupancy_l, occupancy_r)])
    optimal_ev_costs = sum([(tariff_l * x + tariff_r * y) * delta_t * 60 for x, y in
                            zip(milp_out.get('new_occupancy_l'), milp_out.get('new_occupancy_r'))])
    final_ev_costs = sum([(tariff_l * x + tariff_r * y) * a * delta_t * 60 for x, y, a in
                          zip(milp_out.get('new_occupancy_l'), milp_out.get('new_occupancy_r'), alpha)])

    # Prepare output dictionary
    out_stage2 = {'datetime': data.get('timeseries').get('datetime'), 'store_id': milp_out.get('store_id'),
    'obj_value': g_best_pos_fit, 'milp_status': milp_out.get('milp_status'), 'alpha': alpha,
    'alpha_int_1': alpha_int_1, 'alpha_int_2': alpha_int_2, 'alpha_int_3': alpha_int_3, 'extra_profits_bin': extra_profits_bin,
    'load_delta_bin': load_delta_bin, 'new_ev_load': milp_out.get('new_ev_load'), 'ev_load_delta': ev_load_delta,
    'incentive_bin': zero, 'forecasted_profits': forecasted_profits, 'optimal_profits': optimal_profits,
    'extra_profits': extra_profits, 'theta': theta, 'discount_on': zero, 'discount_off': zero, 'discount_bin': zero,
    'ev_load': milp_out.get('ev_load'), 'occupancy': occupancy, 'new_occupancy': milp_out.get('new_occupancy'),
    'extra_profits_w_alpha': extra_profits_w_alpha, 'forecasted_load_ev_costs': forecasted_ev_costs,
    'optimal_load_ev_costs': optimal_ev_costs, 'final_ev_costs': final_ev_costs}

    return {'stage_1': milp_out, 'stage_2': out_stage2}
