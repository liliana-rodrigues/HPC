\* stage1_HPC *\
Minimize
Objective_Function: 0.05 pcc_pin_000 + 0.03 pcc_pin_001 + 0.03 pcc_pin_002
 + 0.03 pcc_pin_003 + 0.02 pcc_pin_004 + 0.03 pcc_pin_005 + 0.04 pcc_pin_006
 + 0.05 pcc_pin_007 + 0.04 pcc_pin_008 + 0.06 pcc_pin_009 + 0.06 pcc_pin_010
 + 0.06 pcc_pin_011 + 0.06 pcc_pin_012 + 0.06 pcc_pin_013 + 0.06 pcc_pin_014
 + 0.06 pcc_pin_015 + 0.06 pcc_pin_016 + 0.06 pcc_pin_017 + 0.07 pcc_pin_018
 + 0.07 pcc_pin_019 + 0.07 pcc_pin_020 + 0.07 pcc_pin_021 + 0.06 pcc_pin_022
 + 0.06 pcc_pin_023 - 0.02 pcc_pout_000 - 0.02 pcc_pout_001
 - 0.02 pcc_pout_002 - 0.02 pcc_pout_003 - 0.01 pcc_pout_004
 - 0.02 pcc_pout_005 - 0.02 pcc_pout_006 - 0.03 pcc_pout_007
 - 0.03 pcc_pout_008 - 0.03 pcc_pout_009 - 0.03 pcc_pout_010
 - 0.04 pcc_pout_011 - 0.04 pcc_pout_012 - 0.05 pcc_pout_013
 - 0.04 pcc_pout_014 - 0.04 pcc_pout_015 - 0.04 pcc_pout_016
 - 0.04 pcc_pout_017 - 0.05 pcc_pout_018 - 0.04 pcc_pout_019
 - 0.05 pcc_pout_020 - 0.05 pcc_pout_021 - 0.04 pcc_pout_022
 - 0.03 pcc_pout_023
Subject To
Battery_charge_rate_limit_00_000: - 300 b_pbin_01_000 + b_pch_01_000 <= 0
Battery_charge_rate_limit_00_001: - 300 b_pbin_01_001 + b_pch_01_001 <= 0
Battery_charge_rate_limit_00_002: - 300 b_pbin_01_002 + b_pch_01_002 <= 0
Battery_charge_rate_limit_00_003: - 300 b_pbin_01_003 + b_pch_01_003 <= 0
Battery_charge_rate_limit_00_004: - 300 b_pbin_01_004 + b_pch_01_004 <= 0
Battery_charge_rate_limit_00_005: - 300 b_pbin_01_005 + b_pch_01_005 <= 0
Battery_charge_rate_limit_00_006: - 300 b_pbin_01_006 + b_pch_01_006 <= 0
Battery_charge_rate_limit_00_007: - 300 b_pbin_01_007 + b_pch_01_007 <= 0
Battery_charge_rate_limit_00_008: - 300 b_pbin_01_008 + b_pch_01_008 <= 0
Battery_charge_rate_limit_00_009: - 300 b_pbin_01_009 + b_pch_01_009 <= 0
Battery_charge_rate_limit_00_010: - 300 b_pbin_01_010 + b_pch_01_010 <= 0
Battery_charge_rate_limit_00_011: - 300 b_pbin_01_011 + b_pch_01_011 <= 0
Battery_charge_rate_limit_00_012: - 300 b_pbin_01_012 + b_pch_01_012 <= 0
Battery_charge_rate_limit_00_013: - 300 b_pbin_01_013 + b_pch_01_013 <= 0
Battery_charge_rate_limit_00_014: - 300 b_pbin_01_014 + b_pch_01_014 <= 0
Battery_charge_rate_limit_00_015: - 300 b_pbin_01_015 + b_pch_01_015 <= 0
Battery_charge_rate_limit_00_016: - 300 b_pbin_01_016 + b_pch_01_016 <= 0
Battery_charge_rate_limit_00_017: - 300 b_pbin_01_017 + b_pch_01_017 <= 0
Battery_charge_rate_limit_00_018: - 300 b_pbin_01_018 + b_pch_01_018 <= 0
Battery_charge_rate_limit_00_019: - 300 b_pbin_01_019 + b_pch_01_019 <= 0
Battery_charge_rate_limit_00_020: - 300 b_pbin_01_020 + b_pch_01_020 <= 0
Battery_charge_rate_limit_00_021: - 300 b_pbin_01_021 + b_pch_01_021 <= 0
Battery_charge_rate_limit_00_022: - 300 b_pbin_01_022 + b_pch_01_022 <= 0
Battery_charge_rate_limit_00_023: - 300 b_pbin_01_023 + b_pch_01_023 <= 0
Battery_charge_rate_limit_01_000: - 100 b_pbin_02_000 + b_pch_02_000 <= 0
Battery_charge_rate_limit_01_001: - 100 b_pbin_02_001 + b_pch_02_001 <= 0
Battery_charge_rate_limit_01_002: - 100 b_pbin_02_002 + b_pch_02_002 <= 0
Battery_charge_rate_limit_01_003: - 100 b_pbin_02_003 + b_pch_02_003 <= 0
Battery_charge_rate_limit_01_004: - 100 b_pbin_02_004 + b_pch_02_004 <= 0
Battery_charge_rate_limit_01_005: - 100 b_pbin_02_005 + b_pch_02_005 <= 0
Battery_charge_rate_limit_01_006: - 100 b_pbin_02_006 + b_pch_02_006 <= 0
Battery_charge_rate_limit_01_007: - 100 b_pbin_02_007 + b_pch_02_007 <= 0
Battery_charge_rate_limit_01_008: - 100 b_pbin_02_008 + b_pch_02_008 <= 0
Battery_charge_rate_limit_01_009: - 100 b_pbin_02_009 + b_pch_02_009 <= 0
Battery_charge_rate_limit_01_010: - 100 b_pbin_02_010 + b_pch_02_010 <= 0
Battery_charge_rate_limit_01_011: - 100 b_pbin_02_011 + b_pch_02_011 <= 0
Battery_charge_rate_limit_01_012: - 100 b_pbin_02_012 + b_pch_02_012 <= 0
Battery_charge_rate_limit_01_013: - 100 b_pbin_02_013 + b_pch_02_013 <= 0
Battery_charge_rate_limit_01_014: - 100 b_pbin_02_014 + b_pch_02_014 <= 0
Battery_charge_rate_limit_01_015: - 100 b_pbin_02_015 + b_pch_02_015 <= 0
Battery_charge_rate_limit_01_016: - 100 b_pbin_02_016 + b_pch_02_016 <= 0
Battery_charge_rate_limit_01_017: - 100 b_pbin_02_017 + b_pch_02_017 <= 0
Battery_charge_rate_limit_01_018: - 100 b_pbin_02_018 + b_pch_02_018 <= 0
Battery_charge_rate_limit_01_019: - 100 b_pbin_02_019 + b_pch_02_019 <= 0
Battery_charge_rate_limit_01_020: - 100 b_pbin_02_020 + b_pch_02_020 <= 0
Battery_charge_rate_limit_01_021: - 100 b_pbin_02_021 + b_pch_02_021 <= 0
Battery_charge_rate_limit_01_022: - 100 b_pbin_02_022 + b_pch_02_022 <= 0
Battery_charge_rate_limit_01_023: - 100 b_pbin_02_023 + b_pch_02_023 <= 0
Battery_discharge_rate_limit00_000: 300 b_pbin_01_000 + b_pdis_01_000 <= 300
Battery_discharge_rate_limit00_001: 300 b_pbin_01_001 + b_pdis_01_001 <= 300
Battery_discharge_rate_limit00_002: 300 b_pbin_01_002 + b_pdis_01_002 <= 300
Battery_discharge_rate_limit00_003: 300 b_pbin_01_003 + b_pdis_01_003 <= 300
Battery_discharge_rate_limit00_004: 300 b_pbin_01_004 + b_pdis_01_004 <= 300
Battery_discharge_rate_limit00_005: 300 b_pbin_01_005 + b_pdis_01_005 <= 300
Battery_discharge_rate_limit00_006: 300 b_pbin_01_006 + b_pdis_01_006 <= 300
Battery_discharge_rate_limit00_007: 300 b_pbin_01_007 + b_pdis_01_007 <= 300
Battery_discharge_rate_limit00_008: 300 b_pbin_01_008 + b_pdis_01_008 <= 300
Battery_discharge_rate_limit00_009: 300 b_pbin_01_009 + b_pdis_01_009 <= 300
Battery_discharge_rate_limit00_010: 300 b_pbin_01_010 + b_pdis_01_010 <= 300
Battery_discharge_rate_limit00_011: 300 b_pbin_01_011 + b_pdis_01_011 <= 300
Battery_discharge_rate_limit00_012: 300 b_pbin_01_012 + b_pdis_01_012 <= 300
Battery_discharge_rate_limit00_013: 300 b_pbin_01_013 + b_pdis_01_013 <= 300
Battery_discharge_rate_limit00_014: 300 b_pbin_01_014 + b_pdis_01_014 <= 300
Battery_discharge_rate_limit00_015: 300 b_pbin_01_015 + b_pdis_01_015 <= 300
Battery_discharge_rate_limit00_016: 300 b_pbin_01_016 + b_pdis_01_016 <= 300
Battery_discharge_rate_limit00_017: 300 b_pbin_01_017 + b_pdis_01_017 <= 300
Battery_discharge_rate_limit00_018: 300 b_pbin_01_018 + b_pdis_01_018 <= 300
Battery_discharge_rate_limit00_019: 300 b_pbin_01_019 + b_pdis_01_019 <= 300
Battery_discharge_rate_limit00_020: 300 b_pbin_01_020 + b_pdis_01_020 <= 300
Battery_discharge_rate_limit00_021: 300 b_pbin_01_021 + b_pdis_01_021 <= 300
Battery_discharge_rate_limit00_022: 300 b_pbin_01_022 + b_pdis_01_022 <= 300
Battery_discharge_rate_limit00_023: 300 b_pbin_01_023 + b_pdis_01_023 <= 300
Battery_discharge_rate_limit01_000: 100 b_pbin_02_000 + b_pdis_02_000 <= 100
Battery_discharge_rate_limit01_001: 100 b_pbin_02_001 + b_pdis_02_001 <= 100
Battery_discharge_rate_limit01_002: 100 b_pbin_02_002 + b_pdis_02_002 <= 100
Battery_discharge_rate_limit01_003: 100 b_pbin_02_003 + b_pdis_02_003 <= 100
Battery_discharge_rate_limit01_004: 100 b_pbin_02_004 + b_pdis_02_004 <= 100
Battery_discharge_rate_limit01_005: 100 b_pbin_02_005 + b_pdis_02_005 <= 100
Battery_discharge_rate_limit01_006: 100 b_pbin_02_006 + b_pdis_02_006 <= 100
Battery_discharge_rate_limit01_007: 100 b_pbin_02_007 + b_pdis_02_007 <= 100
Battery_discharge_rate_limit01_008: 100 b_pbin_02_008 + b_pdis_02_008 <= 100
Battery_discharge_rate_limit01_009: 100 b_pbin_02_009 + b_pdis_02_009 <= 100
Battery_discharge_rate_limit01_010: 100 b_pbin_02_010 + b_pdis_02_010 <= 100
Battery_discharge_rate_limit01_011: 100 b_pbin_02_011 + b_pdis_02_011 <= 100
Battery_discharge_rate_limit01_012: 100 b_pbin_02_012 + b_pdis_02_012 <= 100
Battery_discharge_rate_limit01_013: 100 b_pbin_02_013 + b_pdis_02_013 <= 100
Battery_discharge_rate_limit01_014: 100 b_pbin_02_014 + b_pdis_02_014 <= 100
Battery_discharge_rate_limit01_015: 100 b_pbin_02_015 + b_pdis_02_015 <= 100
Battery_discharge_rate_limit01_016: 100 b_pbin_02_016 + b_pdis_02_016 <= 100
Battery_discharge_rate_limit01_017: 100 b_pbin_02_017 + b_pdis_02_017 <= 100
Battery_discharge_rate_limit01_018: 100 b_pbin_02_018 + b_pdis_02_018 <= 100
Battery_discharge_rate_limit01_019: 100 b_pbin_02_019 + b_pdis_02_019 <= 100
Battery_discharge_rate_limit01_020: 100 b_pbin_02_020 + b_pdis_02_020 <= 100
Battery_discharge_rate_limit01_021: 100 b_pbin_02_021 + b_pdis_02_021 <= 100
Battery_discharge_rate_limit01_022: 100 b_pbin_02_022 + b_pdis_02_022 <= 100
Battery_discharge_rate_limit01_023: 100 b_pbin_02_023 + b_pdis_02_023 <= 100
CoolingLoad_000: p_cooling_000 = 0.010425
CoolingLoad_001: p_cooling_001 = 0.010425
CoolingLoad_002: p_cooling_002 = 0.010425
CoolingLoad_003: p_cooling_003 = 0.010425
CoolingLoad_004: p_cooling_004 = 0.010425
CoolingLoad_005: p_cooling_005 = 0.010425
CoolingLoad_006: p_cooling_006 = 0.010425
CoolingLoad_007: p_cooling_007 = 0.010425
CoolingLoad_008: p_cooling_008 = 0.010425
CoolingLoad_009: p_cooling_009 = 0.010425
CoolingLoad_010: p_cooling_010 = 0.010425
CoolingLoad_011: p_cooling_011 = 0.010425
CoolingLoad_012: p_cooling_012 = 0.010425
CoolingLoad_013: p_cooling_013 = 0.010425
CoolingLoad_014: p_cooling_014 = 0.010425
CoolingLoad_015: p_cooling_015 = 0.010425
CoolingLoad_016: p_cooling_016 = 0.010425
CoolingLoad_017: p_cooling_017 = 0.010425
CoolingLoad_018: p_cooling_018 = 0.010425
CoolingLoad_019: p_cooling_019 = 0.010425
CoolingLoad_020: p_cooling_020 = 0.010425
CoolingLoad_021: p_cooling_021 = 0.010425
CoolingLoad_022: p_cooling_022 = 0.010425
CoolingLoad_023: p_cooling_023 = 0.010425
Energy_to_SOC_00_000: - 0.0666666666667 b_energy_01_000 + b_soc_01_000 = 0
Energy_to_SOC_00_001: - 0.0666666666667 b_energy_01_001 + b_soc_01_001 = 0
Energy_to_SOC_00_002: - 0.0666666666667 b_energy_01_002 + b_soc_01_002 = 0
Energy_to_SOC_00_003: - 0.0666666666667 b_energy_01_003 + b_soc_01_003 = 0
Energy_to_SOC_00_004: - 0.0666666666667 b_energy_01_004 + b_soc_01_004 = 0
Energy_to_SOC_00_005: - 0.0666666666667 b_energy_01_005 + b_soc_01_005 = 0
Energy_to_SOC_00_006: - 0.0666666666667 b_energy_01_006 + b_soc_01_006 = 0
Energy_to_SOC_00_007: - 0.0666666666667 b_energy_01_007 + b_soc_01_007 = 0
Energy_to_SOC_00_008: - 0.0666666666667 b_energy_01_008 + b_soc_01_008 = 0
Energy_to_SOC_00_009: - 0.0666666666667 b_energy_01_009 + b_soc_01_009 = 0
Energy_to_SOC_00_010: - 0.0666666666667 b_energy_01_010 + b_soc_01_010 = 0
Energy_to_SOC_00_011: - 0.0666666666667 b_energy_01_011 + b_soc_01_011 = 0
Energy_to_SOC_00_012: - 0.0666666666667 b_energy_01_012 + b_soc_01_012 = 0
Energy_to_SOC_00_013: - 0.0666666666667 b_energy_01_013 + b_soc_01_013 = 0
Energy_to_SOC_00_014: - 0.0666666666667 b_energy_01_014 + b_soc_01_014 = 0
Energy_to_SOC_00_015: - 0.0666666666667 b_energy_01_015 + b_soc_01_015 = 0
Energy_to_SOC_00_016: - 0.0666666666667 b_energy_01_016 + b_soc_01_016 = 0
Energy_to_SOC_00_017: - 0.0666666666667 b_energy_01_017 + b_soc_01_017 = 0
Energy_to_SOC_00_018: - 0.0666666666667 b_energy_01_018 + b_soc_01_018 = 0
Energy_to_SOC_00_019: - 0.0666666666667 b_energy_01_019 + b_soc_01_019 = 0
Energy_to_SOC_00_020: - 0.0666666666667 b_energy_01_020 + b_soc_01_020 = 0
Energy_to_SOC_00_021: - 0.0666666666667 b_energy_01_021 + b_soc_01_021 = 0
Energy_to_SOC_00_022: - 0.0666666666667 b_energy_01_022 + b_soc_01_022 = 0
Energy_to_SOC_00_023: - 0.0666666666667 b_energy_01_023 + b_soc_01_023 = 0
Energy_to_SOC_01_000: - 0.2 b_energy_02_000 + b_soc_02_000 = 0
Energy_to_SOC_01_001: - 0.2 b_energy_02_001 + b_soc_02_001 = 0
Energy_to_SOC_01_002: - 0.2 b_energy_02_002 + b_soc_02_002 = 0
Energy_to_SOC_01_003: - 0.2 b_energy_02_003 + b_soc_02_003 = 0
Energy_to_SOC_01_004: - 0.2 b_energy_02_004 + b_soc_02_004 = 0
Energy_to_SOC_01_005: - 0.2 b_energy_02_005 + b_soc_02_005 = 0
Energy_to_SOC_01_006: - 0.2 b_energy_02_006 + b_soc_02_006 = 0
Energy_to_SOC_01_007: - 0.2 b_energy_02_007 + b_soc_02_007 = 0
Energy_to_SOC_01_008: - 0.2 b_energy_02_008 + b_soc_02_008 = 0
Energy_to_SOC_01_009: - 0.2 b_energy_02_009 + b_soc_02_009 = 0
Energy_to_SOC_01_010: - 0.2 b_energy_02_010 + b_soc_02_010 = 0
Energy_to_SOC_01_011: - 0.2 b_energy_02_011 + b_soc_02_011 = 0
Energy_to_SOC_01_012: - 0.2 b_energy_02_012 + b_soc_02_012 = 0
Energy_to_SOC_01_013: - 0.2 b_energy_02_013 + b_soc_02_013 = 0
Energy_to_SOC_01_014: - 0.2 b_energy_02_014 + b_soc_02_014 = 0
Energy_to_SOC_01_015: - 0.2 b_energy_02_015 + b_soc_02_015 = 0
Energy_to_SOC_01_016: - 0.2 b_energy_02_016 + b_soc_02_016 = 0
Energy_to_SOC_01_017: - 0.2 b_energy_02_017 + b_soc_02_017 = 0
Energy_to_SOC_01_018: - 0.2 b_energy_02_018 + b_soc_02_018 = 0
Energy_to_SOC_01_019: - 0.2 b_energy_02_019 + b_soc_02_019 = 0
Energy_to_SOC_01_020: - 0.2 b_energy_02_020 + b_soc_02_020 = 0
Energy_to_SOC_01_021: - 0.2 b_energy_02_021 + b_soc_02_021 = 0
Energy_to_SOC_01_022: - 0.2 b_energy_02_022 + b_soc_02_022 = 0
Energy_to_SOC_01_023: - 0.2 b_energy_02_023 + b_soc_02_023 = 0
EquilibriumCharge_00_000: b_pch_01_000 - b_pch_grid_01_000
 - p_gen_pv_bat_01_000 - p_gen_wind_bat_01_000 = 0
EquilibriumCharge_00_001: b_pch_01_001 - b_pch_grid_01_001
 - p_gen_pv_bat_01_001 - p_gen_wind_bat_01_001 = 0
EquilibriumCharge_00_002: b_pch_01_002 - b_pch_grid_01_002
 - p_gen_pv_bat_01_002 - p_gen_wind_bat_01_002 = 0
EquilibriumCharge_00_003: b_pch_01_003 - b_pch_grid_01_003
 - p_gen_pv_bat_01_003 - p_gen_wind_bat_01_003 = 0
EquilibriumCharge_00_004: b_pch_01_004 - b_pch_grid_01_004
 - p_gen_pv_bat_01_004 - p_gen_wind_bat_01_004 = 0
EquilibriumCharge_00_005: b_pch_01_005 - b_pch_grid_01_005
 - p_gen_pv_bat_01_005 - p_gen_wind_bat_01_005 = 0
EquilibriumCharge_00_006: b_pch_01_006 - b_pch_grid_01_006
 - p_gen_pv_bat_01_006 - p_gen_wind_bat_01_006 = 0
EquilibriumCharge_00_007: b_pch_01_007 - b_pch_grid_01_007
 - p_gen_pv_bat_01_007 - p_gen_wind_bat_01_007 = 0
EquilibriumCharge_00_008: b_pch_01_008 - b_pch_grid_01_008
 - p_gen_pv_bat_01_008 - p_gen_wind_bat_01_008 = 0
EquilibriumCharge_00_009: b_pch_01_009 - b_pch_grid_01_009
 - p_gen_pv_bat_01_009 - p_gen_wind_bat_01_009 = 0
EquilibriumCharge_00_010: b_pch_01_010 - b_pch_grid_01_010
 - p_gen_pv_bat_01_010 - p_gen_wind_bat_01_010 = 0
EquilibriumCharge_00_011: b_pch_01_011 - b_pch_grid_01_011
 - p_gen_pv_bat_01_011 - p_gen_wind_bat_01_011 = 0
EquilibriumCharge_00_012: b_pch_01_012 - b_pch_grid_01_012
 - p_gen_pv_bat_01_012 - p_gen_wind_bat_01_012 = 0
EquilibriumCharge_00_013: b_pch_01_013 - b_pch_grid_01_013
 - p_gen_pv_bat_01_013 - p_gen_wind_bat_01_013 = 0
EquilibriumCharge_00_014: b_pch_01_014 - b_pch_grid_01_014
 - p_gen_pv_bat_01_014 - p_gen_wind_bat_01_014 = 0
EquilibriumCharge_00_015: b_pch_01_015 - b_pch_grid_01_015
 - p_gen_pv_bat_01_015 - p_gen_wind_bat_01_015 = 0
EquilibriumCharge_00_016: b_pch_01_016 - b_pch_grid_01_016
 - p_gen_pv_bat_01_016 - p_gen_wind_bat_01_016 = 0
EquilibriumCharge_00_017: b_pch_01_017 - b_pch_grid_01_017
 - p_gen_pv_bat_01_017 - p_gen_wind_bat_01_017 = 0
EquilibriumCharge_00_018: b_pch_01_018 - b_pch_grid_01_018
 - p_gen_pv_bat_01_018 - p_gen_wind_bat_01_018 = 0
EquilibriumCharge_00_019: b_pch_01_019 - b_pch_grid_01_019
 - p_gen_pv_bat_01_019 - p_gen_wind_bat_01_019 = 0
EquilibriumCharge_00_020: b_pch_01_020 - b_pch_grid_01_020
 - p_gen_pv_bat_01_020 - p_gen_wind_bat_01_020 = 0
EquilibriumCharge_00_021: b_pch_01_021 - b_pch_grid_01_021
 - p_gen_pv_bat_01_021 - p_gen_wind_bat_01_021 = 0
EquilibriumCharge_00_022: b_pch_01_022 - b_pch_grid_01_022
 - p_gen_pv_bat_01_022 - p_gen_wind_bat_01_022 = 0
EquilibriumCharge_00_023: b_pch_01_023 - b_pch_grid_01_023
 - p_gen_pv_bat_01_023 - p_gen_wind_bat_01_023 = 0
EquilibriumCharge_01_000: b_pch_02_000 - b_pch_grid_02_000
 - p_gen_pv_bat_02_000 - p_gen_wind_bat_02_000 = 0
EquilibriumCharge_01_001: b_pch_02_001 - b_pch_grid_02_001
 - p_gen_pv_bat_02_001 - p_gen_wind_bat_02_001 = 0
EquilibriumCharge_01_002: b_pch_02_002 - b_pch_grid_02_002
 - p_gen_pv_bat_02_002 - p_gen_wind_bat_02_002 = 0
EquilibriumCharge_01_003: b_pch_02_003 - b_pch_grid_02_003
 - p_gen_pv_bat_02_003 - p_gen_wind_bat_02_003 = 0
EquilibriumCharge_01_004: b_pch_02_004 - b_pch_grid_02_004
 - p_gen_pv_bat_02_004 - p_gen_wind_bat_02_004 = 0
EquilibriumCharge_01_005: b_pch_02_005 - b_pch_grid_02_005
 - p_gen_pv_bat_02_005 - p_gen_wind_bat_02_005 = 0
EquilibriumCharge_01_006: b_pch_02_006 - b_pch_grid_02_006
 - p_gen_pv_bat_02_006 - p_gen_wind_bat_02_006 = 0
EquilibriumCharge_01_007: b_pch_02_007 - b_pch_grid_02_007
 - p_gen_pv_bat_02_007 - p_gen_wind_bat_02_007 = 0
EquilibriumCharge_01_008: b_pch_02_008 - b_pch_grid_02_008
 - p_gen_pv_bat_02_008 - p_gen_wind_bat_02_008 = 0
EquilibriumCharge_01_009: b_pch_02_009 - b_pch_grid_02_009
 - p_gen_pv_bat_02_009 - p_gen_wind_bat_02_009 = 0
EquilibriumCharge_01_010: b_pch_02_010 - b_pch_grid_02_010
 - p_gen_pv_bat_02_010 - p_gen_wind_bat_02_010 = 0
EquilibriumCharge_01_011: b_pch_02_011 - b_pch_grid_02_011
 - p_gen_pv_bat_02_011 - p_gen_wind_bat_02_011 = 0
EquilibriumCharge_01_012: b_pch_02_012 - b_pch_grid_02_012
 - p_gen_pv_bat_02_012 - p_gen_wind_bat_02_012 = 0
EquilibriumCharge_01_013: b_pch_02_013 - b_pch_grid_02_013
 - p_gen_pv_bat_02_013 - p_gen_wind_bat_02_013 = 0
EquilibriumCharge_01_014: b_pch_02_014 - b_pch_grid_02_014
 - p_gen_pv_bat_02_014 - p_gen_wind_bat_02_014 = 0
EquilibriumCharge_01_015: b_pch_02_015 - b_pch_grid_02_015
 - p_gen_pv_bat_02_015 - p_gen_wind_bat_02_015 = 0
EquilibriumCharge_01_016: b_pch_02_016 - b_pch_grid_02_016
 - p_gen_pv_bat_02_016 - p_gen_wind_bat_02_016 = 0
EquilibriumCharge_01_017: b_pch_02_017 - b_pch_grid_02_017
 - p_gen_pv_bat_02_017 - p_gen_wind_bat_02_017 = 0
EquilibriumCharge_01_018: b_pch_02_018 - b_pch_grid_02_018
 - p_gen_pv_bat_02_018 - p_gen_wind_bat_02_018 = 0
EquilibriumCharge_01_019: b_pch_02_019 - b_pch_grid_02_019
 - p_gen_pv_bat_02_019 - p_gen_wind_bat_02_019 = 0
EquilibriumCharge_01_020: b_pch_02_020 - b_pch_grid_02_020
 - p_gen_pv_bat_02_020 - p_gen_wind_bat_02_020 = 0
EquilibriumCharge_01_021: b_pch_02_021 - b_pch_grid_02_021
 - p_gen_pv_bat_02_021 - p_gen_wind_bat_02_021 = 0
EquilibriumCharge_01_022: b_pch_02_022 - b_pch_grid_02_022
 - p_gen_pv_bat_02_022 - p_gen_wind_bat_02_022 = 0
EquilibriumCharge_01_023: b_pch_02_023 - b_pch_grid_02_023
 - p_gen_pv_bat_02_023 - p_gen_wind_bat_02_023 = 0
EquilibriumDischarge_00_000: b_pdis_01_000 - b_pdis_grid_01_000
 - b_pdis_load_01_000 = 0
EquilibriumDischarge_00_001: b_pdis_01_001 - b_pdis_grid_01_001
 - b_pdis_load_01_001 = 0
EquilibriumDischarge_00_002: b_pdis_01_002 - b_pdis_grid_01_002
 - b_pdis_load_01_002 = 0
EquilibriumDischarge_00_003: b_pdis_01_003 - b_pdis_grid_01_003
 - b_pdis_load_01_003 = 0
EquilibriumDischarge_00_004: b_pdis_01_004 - b_pdis_grid_01_004
 - b_pdis_load_01_004 = 0
EquilibriumDischarge_00_005: b_pdis_01_005 - b_pdis_grid_01_005
 - b_pdis_load_01_005 = 0
EquilibriumDischarge_00_006: b_pdis_01_006 - b_pdis_grid_01_006
 - b_pdis_load_01_006 = 0
EquilibriumDischarge_00_007: b_pdis_01_007 - b_pdis_grid_01_007
 - b_pdis_load_01_007 = 0
EquilibriumDischarge_00_008: b_pdis_01_008 - b_pdis_grid_01_008
 - b_pdis_load_01_008 = 0
EquilibriumDischarge_00_009: b_pdis_01_009 - b_pdis_grid_01_009
 - b_pdis_load_01_009 = 0
EquilibriumDischarge_00_010: b_pdis_01_010 - b_pdis_grid_01_010
 - b_pdis_load_01_010 = 0
EquilibriumDischarge_00_011: b_pdis_01_011 - b_pdis_grid_01_011
 - b_pdis_load_01_011 = 0
EquilibriumDischarge_00_012: b_pdis_01_012 - b_pdis_grid_01_012
 - b_pdis_load_01_012 = 0
EquilibriumDischarge_00_013: b_pdis_01_013 - b_pdis_grid_01_013
 - b_pdis_load_01_013 = 0
EquilibriumDischarge_00_014: b_pdis_01_014 - b_pdis_grid_01_014
 - b_pdis_load_01_014 = 0
EquilibriumDischarge_00_015: b_pdis_01_015 - b_pdis_grid_01_015
 - b_pdis_load_01_015 = 0
EquilibriumDischarge_00_016: b_pdis_01_016 - b_pdis_grid_01_016
 - b_pdis_load_01_016 = 0
EquilibriumDischarge_00_017: b_pdis_01_017 - b_pdis_grid_01_017
 - b_pdis_load_01_017 = 0
EquilibriumDischarge_00_018: b_pdis_01_018 - b_pdis_grid_01_018
 - b_pdis_load_01_018 = 0
EquilibriumDischarge_00_019: b_pdis_01_019 - b_pdis_grid_01_019
 - b_pdis_load_01_019 = 0
EquilibriumDischarge_00_020: b_pdis_01_020 - b_pdis_grid_01_020
 - b_pdis_load_01_020 = 0
EquilibriumDischarge_00_021: b_pdis_01_021 - b_pdis_grid_01_021
 - b_pdis_load_01_021 = 0
EquilibriumDischarge_00_022: b_pdis_01_022 - b_pdis_grid_01_022
 - b_pdis_load_01_022 = 0
EquilibriumDischarge_00_023: b_pdis_01_023 - b_pdis_grid_01_023
 - b_pdis_load_01_023 = 0
EquilibriumDischarge_01_000: b_pdis_02_000 - b_pdis_grid_02_000
 - b_pdis_load_02_000 = 0
EquilibriumDischarge_01_001: b_pdis_02_001 - b_pdis_grid_02_001
 - b_pdis_load_02_001 = 0
EquilibriumDischarge_01_002: b_pdis_02_002 - b_pdis_grid_02_002
 - b_pdis_load_02_002 = 0
EquilibriumDischarge_01_003: b_pdis_02_003 - b_pdis_grid_02_003
 - b_pdis_load_02_003 = 0
EquilibriumDischarge_01_004: b_pdis_02_004 - b_pdis_grid_02_004
 - b_pdis_load_02_004 = 0
EquilibriumDischarge_01_005: b_pdis_02_005 - b_pdis_grid_02_005
 - b_pdis_load_02_005 = 0
EquilibriumDischarge_01_006: b_pdis_02_006 - b_pdis_grid_02_006
 - b_pdis_load_02_006 = 0
EquilibriumDischarge_01_007: b_pdis_02_007 - b_pdis_grid_02_007
 - b_pdis_load_02_007 = 0
EquilibriumDischarge_01_008: b_pdis_02_008 - b_pdis_grid_02_008
 - b_pdis_load_02_008 = 0
EquilibriumDischarge_01_009: b_pdis_02_009 - b_pdis_grid_02_009
 - b_pdis_load_02_009 = 0
EquilibriumDischarge_01_010: b_pdis_02_010 - b_pdis_grid_02_010
 - b_pdis_load_02_010 = 0
EquilibriumDischarge_01_011: b_pdis_02_011 - b_pdis_grid_02_011
 - b_pdis_load_02_011 = 0
EquilibriumDischarge_01_012: b_pdis_02_012 - b_pdis_grid_02_012
 - b_pdis_load_02_012 = 0
EquilibriumDischarge_01_013: b_pdis_02_013 - b_pdis_grid_02_013
 - b_pdis_load_02_013 = 0
EquilibriumDischarge_01_014: b_pdis_02_014 - b_pdis_grid_02_014
 - b_pdis_load_02_014 = 0
EquilibriumDischarge_01_015: b_pdis_02_015 - b_pdis_grid_02_015
 - b_pdis_load_02_015 = 0
EquilibriumDischarge_01_016: b_pdis_02_016 - b_pdis_grid_02_016
 - b_pdis_load_02_016 = 0
EquilibriumDischarge_01_017: b_pdis_02_017 - b_pdis_grid_02_017
 - b_pdis_load_02_017 = 0
EquilibriumDischarge_01_018: b_pdis_02_018 - b_pdis_grid_02_018
 - b_pdis_load_02_018 = 0
EquilibriumDischarge_01_019: b_pdis_02_019 - b_pdis_grid_02_019
 - b_pdis_load_02_019 = 0
EquilibriumDischarge_01_020: b_pdis_02_020 - b_pdis_grid_02_020
 - b_pdis_load_02_020 = 0
EquilibriumDischarge_01_021: b_pdis_02_021 - b_pdis_grid_02_021
 - b_pdis_load_02_021 = 0
EquilibriumDischarge_01_022: b_pdis_02_022 - b_pdis_grid_02_022
 - b_pdis_load_02_022 = 0
EquilibriumDischarge_01_023: b_pdis_02_023 - b_pdis_grid_02_023
 - b_pdis_load_02_023 = 0
EquilibriumIn_000: - b_pch_grid_01_000 - b_pch_grid_02_000 + pcc_pin_000
 - pcc_pin_load_000 = 0
EquilibriumIn_001: - b_pch_grid_01_001 - b_pch_grid_02_001 + pcc_pin_001
 - pcc_pin_load_001 = 0
EquilibriumIn_002: - b_pch_grid_01_002 - b_pch_grid_02_002 + pcc_pin_002
 - pcc_pin_load_002 = 0
EquilibriumIn_003: - b_pch_grid_01_003 - b_pch_grid_02_003 + pcc_pin_003
 - pcc_pin_load_003 = 0
EquilibriumIn_004: - b_pch_grid_01_004 - b_pch_grid_02_004 + pcc_pin_004
 - pcc_pin_load_004 = 0
EquilibriumIn_005: - b_pch_grid_01_005 - b_pch_grid_02_005 + pcc_pin_005
 - pcc_pin_load_005 = 0
EquilibriumIn_006: - b_pch_grid_01_006 - b_pch_grid_02_006 + pcc_pin_006
 - pcc_pin_load_006 = 0
EquilibriumIn_007: - b_pch_grid_01_007 - b_pch_grid_02_007 + pcc_pin_007
 - pcc_pin_load_007 = 0
EquilibriumIn_008: - b_pch_grid_01_008 - b_pch_grid_02_008 + pcc_pin_008
 - pcc_pin_load_008 = 0
EquilibriumIn_009: - b_pch_grid_01_009 - b_pch_grid_02_009 + pcc_pin_009
 - pcc_pin_load_009 = 0
EquilibriumIn_010: - b_pch_grid_01_010 - b_pch_grid_02_010 + pcc_pin_010
 - pcc_pin_load_010 = 0
EquilibriumIn_011: - b_pch_grid_01_011 - b_pch_grid_02_011 + pcc_pin_011
 - pcc_pin_load_011 = 0
EquilibriumIn_012: - b_pch_grid_01_012 - b_pch_grid_02_012 + pcc_pin_012
 - pcc_pin_load_012 = 0
EquilibriumIn_013: - b_pch_grid_01_013 - b_pch_grid_02_013 + pcc_pin_013
 - pcc_pin_load_013 = 0
EquilibriumIn_014: - b_pch_grid_01_014 - b_pch_grid_02_014 + pcc_pin_014
 - pcc_pin_load_014 = 0
EquilibriumIn_015: - b_pch_grid_01_015 - b_pch_grid_02_015 + pcc_pin_015
 - pcc_pin_load_015 = 0
EquilibriumIn_016: - b_pch_grid_01_016 - b_pch_grid_02_016 + pcc_pin_016
 - pcc_pin_load_016 = 0
EquilibriumIn_017: - b_pch_grid_01_017 - b_pch_grid_02_017 + pcc_pin_017
 - pcc_pin_load_017 = 0
EquilibriumIn_018: - b_pch_grid_01_018 - b_pch_grid_02_018 + pcc_pin_018
 - pcc_pin_load_018 = 0
EquilibriumIn_019: - b_pch_grid_01_019 - b_pch_grid_02_019 + pcc_pin_019
 - pcc_pin_load_019 = 0
EquilibriumIn_020: - b_pch_grid_01_020 - b_pch_grid_02_020 + pcc_pin_020
 - pcc_pin_load_020 = 0
EquilibriumIn_021: - b_pch_grid_01_021 - b_pch_grid_02_021 + pcc_pin_021
 - pcc_pin_load_021 = 0
EquilibriumIn_022: - b_pch_grid_01_022 - b_pch_grid_02_022 + pcc_pin_022
 - pcc_pin_load_022 = 0
EquilibriumIn_023: - b_pch_grid_01_023 - b_pch_grid_02_023 + pcc_pin_023
 - pcc_pin_load_023 = 0
EquilibriumLoad_000: - b_pdis_load_01_000 - b_pdis_load_02_000
 - p_gen_pv_load_000 - p_gen_wind_load_000 - pcc_pin_load_000 + v_load_000 = 0
EquilibriumLoad_001: - b_pdis_load_01_001 - b_pdis_load_02_001
 - p_gen_pv_load_001 - p_gen_wind_load_001 - pcc_pin_load_001 + v_load_001 = 0
EquilibriumLoad_002: - b_pdis_load_01_002 - b_pdis_load_02_002
 - p_gen_pv_load_002 - p_gen_wind_load_002 - pcc_pin_load_002 + v_load_002 = 0
EquilibriumLoad_003: - b_pdis_load_01_003 - b_pdis_load_02_003
 - p_gen_pv_load_003 - p_gen_wind_load_003 - pcc_pin_load_003 + v_load_003 = 0
EquilibriumLoad_004: - b_pdis_load_01_004 - b_pdis_load_02_004
 - p_gen_pv_load_004 - p_gen_wind_load_004 - pcc_pin_load_004 + v_load_004 = 0
EquilibriumLoad_005: - b_pdis_load_01_005 - b_pdis_load_02_005
 - p_gen_pv_load_005 - p_gen_wind_load_005 - pcc_pin_load_005 + v_load_005 = 0
EquilibriumLoad_006: - b_pdis_load_01_006 - b_pdis_load_02_006
 - p_gen_pv_load_006 - p_gen_wind_load_006 - pcc_pin_load_006 + v_load_006 = 0
EquilibriumLoad_007: - b_pdis_load_01_007 - b_pdis_load_02_007
 - p_gen_pv_load_007 - p_gen_wind_load_007 - pcc_pin_load_007 + v_load_007 = 0
EquilibriumLoad_008: - b_pdis_load_01_008 - b_pdis_load_02_008
 - p_gen_pv_load_008 - p_gen_wind_load_008 - pcc_pin_load_008 + v_load_008 = 0
EquilibriumLoad_009: - b_pdis_load_01_009 - b_pdis_load_02_009
 - p_gen_pv_load_009 - p_gen_wind_load_009 - pcc_pin_load_009 + v_load_009 = 0
EquilibriumLoad_010: - b_pdis_load_01_010 - b_pdis_load_02_010
 - p_gen_pv_load_010 - p_gen_wind_load_010 - pcc_pin_load_010 + v_load_010 = 0
EquilibriumLoad_011: - b_pdis_load_01_011 - b_pdis_load_02_011
 - p_gen_pv_load_011 - p_gen_wind_load_011 - pcc_pin_load_011 + v_load_011 = 0
EquilibriumLoad_012: - b_pdis_load_01_012 - b_pdis_load_02_012
 - p_gen_pv_load_012 - p_gen_wind_load_012 - pcc_pin_load_012 + v_load_012 = 0
EquilibriumLoad_013: - b_pdis_load_01_013 - b_pdis_load_02_013
 - p_gen_pv_load_013 - p_gen_wind_load_013 - pcc_pin_load_013 + v_load_013 = 0
EquilibriumLoad_014: - b_pdis_load_01_014 - b_pdis_load_02_014
 - p_gen_pv_load_014 - p_gen_wind_load_014 - pcc_pin_load_014 + v_load_014 = 0
EquilibriumLoad_015: - b_pdis_load_01_015 - b_pdis_load_02_015
 - p_gen_pv_load_015 - p_gen_wind_load_015 - pcc_pin_load_015 + v_load_015 = 0
EquilibriumLoad_016: - b_pdis_load_01_016 - b_pdis_load_02_016
 - p_gen_pv_load_016 - p_gen_wind_load_016 - pcc_pin_load_016 + v_load_016 = 0
EquilibriumLoad_017: - b_pdis_load_01_017 - b_pdis_load_02_017
 - p_gen_pv_load_017 - p_gen_wind_load_017 - pcc_pin_load_017 + v_load_017 = 0
EquilibriumLoad_018: - b_pdis_load_01_018 - b_pdis_load_02_018
 - p_gen_pv_load_018 - p_gen_wind_load_018 - pcc_pin_load_018 + v_load_018 = 0
EquilibriumLoad_019: - b_pdis_load_01_019 - b_pdis_load_02_019
 - p_gen_pv_load_019 - p_gen_wind_load_019 - pcc_pin_load_019 + v_load_019 = 0
EquilibriumLoad_020: - b_pdis_load_01_020 - b_pdis_load_02_020
 - p_gen_pv_load_020 - p_gen_wind_load_020 - pcc_pin_load_020 + v_load_020 = 0
EquilibriumLoad_021: - b_pdis_load_01_021 - b_pdis_load_02_021
 - p_gen_pv_load_021 - p_gen_wind_load_021 - pcc_pin_load_021 + v_load_021 = 0
EquilibriumLoad_022: - b_pdis_load_01_022 - b_pdis_load_02_022
 - p_gen_pv_load_022 - p_gen_wind_load_022 - pcc_pin_load_022 + v_load_022 = 0
EquilibriumLoad_023: - b_pdis_load_01_023 - b_pdis_load_02_023
 - p_gen_pv_load_023 - p_gen_wind_load_023 - pcc_pin_load_023 + v_load_023 = 0
EquilibriumOut_000: - b_pdis_grid_01_000 - b_pdis_grid_02_000 + pcc_pout_000
 - pcc_pout_pv_000 - pcc_pout_wind_000 = 0
EquilibriumOut_001: - b_pdis_grid_01_001 - b_pdis_grid_02_001 + pcc_pout_001
 - pcc_pout_pv_001 - pcc_pout_wind_001 = 0
EquilibriumOut_002: - b_pdis_grid_01_002 - b_pdis_grid_02_002 + pcc_pout_002
 - pcc_pout_pv_002 - pcc_pout_wind_002 = 0
EquilibriumOut_003: - b_pdis_grid_01_003 - b_pdis_grid_02_003 + pcc_pout_003
 - pcc_pout_pv_003 - pcc_pout_wind_003 = 0
EquilibriumOut_004: - b_pdis_grid_01_004 - b_pdis_grid_02_004 + pcc_pout_004
 - pcc_pout_pv_004 - pcc_pout_wind_004 = 0
EquilibriumOut_005: - b_pdis_grid_01_005 - b_pdis_grid_02_005 + pcc_pout_005
 - pcc_pout_pv_005 - pcc_pout_wind_005 = 0
EquilibriumOut_006: - b_pdis_grid_01_006 - b_pdis_grid_02_006 + pcc_pout_006
 - pcc_pout_pv_006 - pcc_pout_wind_006 = 0
EquilibriumOut_007: - b_pdis_grid_01_007 - b_pdis_grid_02_007 + pcc_pout_007
 - pcc_pout_pv_007 - pcc_pout_wind_007 = 0
EquilibriumOut_008: - b_pdis_grid_01_008 - b_pdis_grid_02_008 + pcc_pout_008
 - pcc_pout_pv_008 - pcc_pout_wind_008 = 0
EquilibriumOut_009: - b_pdis_grid_01_009 - b_pdis_grid_02_009 + pcc_pout_009
 - pcc_pout_pv_009 - pcc_pout_wind_009 = 0
EquilibriumOut_010: - b_pdis_grid_01_010 - b_pdis_grid_02_010 + pcc_pout_010
 - pcc_pout_pv_010 - pcc_pout_wind_010 = 0
EquilibriumOut_011: - b_pdis_grid_01_011 - b_pdis_grid_02_011 + pcc_pout_011
 - pcc_pout_pv_011 - pcc_pout_wind_011 = 0
EquilibriumOut_012: - b_pdis_grid_01_012 - b_pdis_grid_02_012 + pcc_pout_012
 - pcc_pout_pv_012 - pcc_pout_wind_012 = 0
EquilibriumOut_013: - b_pdis_grid_01_013 - b_pdis_grid_02_013 + pcc_pout_013
 - pcc_pout_pv_013 - pcc_pout_wind_013 = 0
EquilibriumOut_014: - b_pdis_grid_01_014 - b_pdis_grid_02_014 + pcc_pout_014
 - pcc_pout_pv_014 - pcc_pout_wind_014 = 0
EquilibriumOut_015: - b_pdis_grid_01_015 - b_pdis_grid_02_015 + pcc_pout_015
 - pcc_pout_pv_015 - pcc_pout_wind_015 = 0
EquilibriumOut_016: - b_pdis_grid_01_016 - b_pdis_grid_02_016 + pcc_pout_016
 - pcc_pout_pv_016 - pcc_pout_wind_016 = 0
EquilibriumOut_017: - b_pdis_grid_01_017 - b_pdis_grid_02_017 + pcc_pout_017
 - pcc_pout_pv_017 - pcc_pout_wind_017 = 0
EquilibriumOut_018: - b_pdis_grid_01_018 - b_pdis_grid_02_018 + pcc_pout_018
 - pcc_pout_pv_018 - pcc_pout_wind_018 = 0
EquilibriumOut_019: - b_pdis_grid_01_019 - b_pdis_grid_02_019 + pcc_pout_019
 - pcc_pout_pv_019 - pcc_pout_wind_019 = 0
EquilibriumOut_020: - b_pdis_grid_01_020 - b_pdis_grid_02_020 + pcc_pout_020
 - pcc_pout_pv_020 - pcc_pout_wind_020 = 0
EquilibriumOut_021: - b_pdis_grid_01_021 - b_pdis_grid_02_021 + pcc_pout_021
 - pcc_pout_pv_021 - pcc_pout_wind_021 = 0
EquilibriumOut_022: - b_pdis_grid_01_022 - b_pdis_grid_02_022 + pcc_pout_022
 - pcc_pout_pv_022 - pcc_pout_wind_022 = 0
EquilibriumOut_023: - b_pdis_grid_01_023 - b_pdis_grid_02_023 + pcc_pout_023
 - pcc_pout_pv_023 - pcc_pout_wind_023 = 0
EquilibriumPV_000: p_gen_pv_bat_01_000 + p_gen_pv_bat_02_000
 + p_gen_pv_load_000 + pcc_pout_pv_000 = 0
EquilibriumPV_001: p_gen_pv_bat_01_001 + p_gen_pv_bat_02_001
 + p_gen_pv_load_001 + pcc_pout_pv_001 = 0
EquilibriumPV_002: p_gen_pv_bat_01_002 + p_gen_pv_bat_02_002
 + p_gen_pv_load_002 + pcc_pout_pv_002 = 0
EquilibriumPV_003: p_gen_pv_bat_01_003 + p_gen_pv_bat_02_003
 + p_gen_pv_load_003 + pcc_pout_pv_003 = 0
EquilibriumPV_004: p_gen_pv_bat_01_004 + p_gen_pv_bat_02_004
 + p_gen_pv_load_004 + pcc_pout_pv_004 = 0
EquilibriumPV_005: p_gen_pv_bat_01_005 + p_gen_pv_bat_02_005
 + p_gen_pv_load_005 + pcc_pout_pv_005 = 0
EquilibriumPV_006: p_gen_pv_bat_01_006 + p_gen_pv_bat_02_006
 + p_gen_pv_load_006 + pcc_pout_pv_006 = 0.0094
EquilibriumPV_007: p_gen_pv_bat_01_007 + p_gen_pv_bat_02_007
 + p_gen_pv_load_007 + pcc_pout_pv_007 = 0.1587
EquilibriumPV_008: p_gen_pv_bat_01_008 + p_gen_pv_bat_02_008
 + p_gen_pv_load_008 + pcc_pout_pv_008 = 0.8303
EquilibriumPV_009: p_gen_pv_bat_01_009 + p_gen_pv_bat_02_009
 + p_gen_pv_load_009 + pcc_pout_pv_009 = 2.1178
EquilibriumPV_010: p_gen_pv_bat_01_010 + p_gen_pv_bat_02_010
 + p_gen_pv_load_010 + pcc_pout_pv_010 = 3.082
EquilibriumPV_011: p_gen_pv_bat_01_011 + p_gen_pv_bat_02_011
 + p_gen_pv_load_011 + pcc_pout_pv_011 = 3.7968
EquilibriumPV_012: p_gen_pv_bat_01_012 + p_gen_pv_bat_02_012
 + p_gen_pv_load_012 + pcc_pout_pv_012 = 4.26
EquilibriumPV_013: p_gen_pv_bat_01_013 + p_gen_pv_bat_02_013
 + p_gen_pv_load_013 + pcc_pout_pv_013 = 4.3169
EquilibriumPV_014: p_gen_pv_bat_01_014 + p_gen_pv_bat_02_014
 + p_gen_pv_load_014 + pcc_pout_pv_014 = 4.3133
EquilibriumPV_015: p_gen_pv_bat_01_015 + p_gen_pv_bat_02_015
 + p_gen_pv_load_015 + pcc_pout_pv_015 = 4.0002
EquilibriumPV_016: p_gen_pv_bat_01_016 + p_gen_pv_bat_02_016
 + p_gen_pv_load_016 + pcc_pout_pv_016 = 3.3999
EquilibriumPV_017: p_gen_pv_bat_01_017 + p_gen_pv_bat_02_017
 + p_gen_pv_load_017 + pcc_pout_pv_017 = 2.4833
EquilibriumPV_018: p_gen_pv_bat_01_018 + p_gen_pv_bat_02_018
 + p_gen_pv_load_018 + pcc_pout_pv_018 = 1.3025
EquilibriumPV_019: p_gen_pv_bat_01_019 + p_gen_pv_bat_02_019
 + p_gen_pv_load_019 + pcc_pout_pv_019 = 0.2905
EquilibriumPV_020: p_gen_pv_bat_01_020 + p_gen_pv_bat_02_020
 + p_gen_pv_load_020 + pcc_pout_pv_020 = 0.026
EquilibriumPV_021: p_gen_pv_bat_01_021 + p_gen_pv_bat_02_021
 + p_gen_pv_load_021 + pcc_pout_pv_021 = 0
EquilibriumPV_022: p_gen_pv_bat_01_022 + p_gen_pv_bat_02_022
 + p_gen_pv_load_022 + pcc_pout_pv_022 = 0
EquilibriumPV_023: p_gen_pv_bat_01_023 + p_gen_pv_bat_02_023
 + p_gen_pv_load_023 + pcc_pout_pv_023 = 0
EquilibriumWind_000: p_gen_wind_bat_01_000 + p_gen_wind_bat_02_000
 + p_gen_wind_load_000 + pcc_pout_wind_000 = 250
EquilibriumWind_001: p_gen_wind_bat_01_001 + p_gen_wind_bat_02_001
 + p_gen_wind_load_001 + pcc_pout_wind_001 = 220
EquilibriumWind_002: p_gen_wind_bat_01_002 + p_gen_wind_bat_02_002
 + p_gen_wind_load_002 + pcc_pout_wind_002 = 200
EquilibriumWind_003: p_gen_wind_bat_01_003 + p_gen_wind_bat_02_003
 + p_gen_wind_load_003 + pcc_pout_wind_003 = 130
EquilibriumWind_004: p_gen_wind_bat_01_004 + p_gen_wind_bat_02_004
 + p_gen_wind_load_004 + pcc_pout_wind_004 = 70
EquilibriumWind_005: p_gen_wind_bat_01_005 + p_gen_wind_bat_02_005
 + p_gen_wind_load_005 + pcc_pout_wind_005 = 50
EquilibriumWind_006: p_gen_wind_bat_01_006 + p_gen_wind_bat_02_006
 + p_gen_wind_load_006 + pcc_pout_wind_006 = 50
EquilibriumWind_007: p_gen_wind_bat_01_007 + p_gen_wind_bat_02_007
 + p_gen_wind_load_007 + pcc_pout_wind_007 = 40
EquilibriumWind_008: p_gen_wind_bat_01_008 + p_gen_wind_bat_02_008
 + p_gen_wind_load_008 + pcc_pout_wind_008 = 30
EquilibriumWind_009: p_gen_wind_bat_01_009 + p_gen_wind_bat_02_009
 + p_gen_wind_load_009 + pcc_pout_wind_009 = 30
EquilibriumWind_010: p_gen_wind_bat_01_010 + p_gen_wind_bat_02_010
 + p_gen_wind_load_010 + pcc_pout_wind_010 = 70
EquilibriumWind_011: p_gen_wind_bat_01_011 + p_gen_wind_bat_02_011
 + p_gen_wind_load_011 + pcc_pout_wind_011 = 120
EquilibriumWind_012: p_gen_wind_bat_01_012 + p_gen_wind_bat_02_012
 + p_gen_wind_load_012 + pcc_pout_wind_012 = 150
EquilibriumWind_013: p_gen_wind_bat_01_013 + p_gen_wind_bat_02_013
 + p_gen_wind_load_013 + pcc_pout_wind_013 = 180
EquilibriumWind_014: p_gen_wind_bat_01_014 + p_gen_wind_bat_02_014
 + p_gen_wind_load_014 + pcc_pout_wind_014 = 190
EquilibriumWind_015: p_gen_wind_bat_01_015 + p_gen_wind_bat_02_015
 + p_gen_wind_load_015 + pcc_pout_wind_015 = 210
EquilibriumWind_016: p_gen_wind_bat_01_016 + p_gen_wind_bat_02_016
 + p_gen_wind_load_016 + pcc_pout_wind_016 = 230
EquilibriumWind_017: p_gen_wind_bat_01_017 + p_gen_wind_bat_02_017
 + p_gen_wind_load_017 + pcc_pout_wind_017 = 240
EquilibriumWind_018: p_gen_wind_bat_01_018 + p_gen_wind_bat_02_018
 + p_gen_wind_load_018 + pcc_pout_wind_018 = 240
EquilibriumWind_019: p_gen_wind_bat_01_019 + p_gen_wind_bat_02_019
 + p_gen_wind_load_019 + pcc_pout_wind_019 = 250
EquilibriumWind_020: p_gen_wind_bat_01_020 + p_gen_wind_bat_02_020
 + p_gen_wind_load_020 + pcc_pout_wind_020 = 240
EquilibriumWind_021: p_gen_wind_bat_01_021 + p_gen_wind_bat_02_021
 + p_gen_wind_load_021 + pcc_pout_wind_021 = 220
EquilibriumWind_022: p_gen_wind_bat_01_022 + p_gen_wind_bat_02_022
 + p_gen_wind_load_022 + pcc_pout_wind_022 = 220
EquilibriumWind_023: p_gen_wind_bat_01_023 + p_gen_wind_bat_02_023
 + p_gen_wind_load_023 + pcc_pout_wind_023 = 230
Initial_SOC_update_00_000: b_energy_01_000 - 0.9 b_pch_01_000
 + 1.11111111111 b_pdis_01_000 = 750
Initial_SOC_update_01_000: b_energy_02_000 - 0.8 b_pch_02_000
 + 1.25 b_pdis_02_000 = 250
Maximum_SOC_00_000: b_soc_01_000 <= 80
Maximum_SOC_00_001: b_soc_01_001 <= 80
Maximum_SOC_00_002: b_soc_01_002 <= 80
Maximum_SOC_00_003: b_soc_01_003 <= 80
Maximum_SOC_00_004: b_soc_01_004 <= 80
Maximum_SOC_00_005: b_soc_01_005 <= 80
Maximum_SOC_00_006: b_soc_01_006 <= 80
Maximum_SOC_00_007: b_soc_01_007 <= 80
Maximum_SOC_00_008: b_soc_01_008 <= 80
Maximum_SOC_00_009: b_soc_01_009 <= 80
Maximum_SOC_00_010: b_soc_01_010 <= 80
Maximum_SOC_00_011: b_soc_01_011 <= 80
Maximum_SOC_00_012: b_soc_01_012 <= 80
Maximum_SOC_00_013: b_soc_01_013 <= 80
Maximum_SOC_00_014: b_soc_01_014 <= 80
Maximum_SOC_00_015: b_soc_01_015 <= 80
Maximum_SOC_00_016: b_soc_01_016 <= 80
Maximum_SOC_00_017: b_soc_01_017 <= 80
Maximum_SOC_00_018: b_soc_01_018 <= 80
Maximum_SOC_00_019: b_soc_01_019 <= 80
Maximum_SOC_00_020: b_soc_01_020 <= 80
Maximum_SOC_00_021: b_soc_01_021 <= 80
Maximum_SOC_00_022: b_soc_01_022 <= 80
Maximum_SOC_00_023: b_soc_01_023 <= 80
Maximum_SOC_01_000: b_soc_02_000 <= 95
Maximum_SOC_01_001: b_soc_02_001 <= 95
Maximum_SOC_01_002: b_soc_02_002 <= 95
Maximum_SOC_01_003: b_soc_02_003 <= 95
Maximum_SOC_01_004: b_soc_02_004 <= 95
Maximum_SOC_01_005: b_soc_02_005 <= 95
Maximum_SOC_01_006: b_soc_02_006 <= 95
Maximum_SOC_01_007: b_soc_02_007 <= 95
Maximum_SOC_01_008: b_soc_02_008 <= 95
Maximum_SOC_01_009: b_soc_02_009 <= 95
Maximum_SOC_01_010: b_soc_02_010 <= 95
Maximum_SOC_01_011: b_soc_02_011 <= 95
Maximum_SOC_01_012: b_soc_02_012 <= 95
Maximum_SOC_01_013: b_soc_02_013 <= 95
Maximum_SOC_01_014: b_soc_02_014 <= 95
Maximum_SOC_01_015: b_soc_02_015 <= 95
Maximum_SOC_01_016: b_soc_02_016 <= 95
Maximum_SOC_01_017: b_soc_02_017 <= 95
Maximum_SOC_01_018: b_soc_02_018 <= 95
Maximum_SOC_01_019: b_soc_02_019 <= 95
Maximum_SOC_01_020: b_soc_02_020 <= 95
Maximum_SOC_01_021: b_soc_02_021 <= 95
Maximum_SOC_01_022: b_soc_02_022 <= 95
Maximum_SOC_01_023: b_soc_02_023 <= 95
MinimumLoad_000: v_computing_000 >= 0.0417
MinimumLoad_001: v_computing_001 >= 0.0417
MinimumLoad_002: v_computing_002 >= 0.0417
MinimumLoad_003: v_computing_003 >= 0.0417
MinimumLoad_004: v_computing_004 >= 0.0417
MinimumLoad_005: v_computing_005 >= 0.0417
MinimumLoad_006: v_computing_006 >= 0.0417
MinimumLoad_007: v_computing_007 >= 0.0417
MinimumLoad_008: v_computing_008 >= 0.0417
MinimumLoad_009: v_computing_009 >= 0.0417
MinimumLoad_010: v_computing_010 >= 0.0417
MinimumLoad_011: v_computing_011 >= 0.0417
MinimumLoad_012: v_computing_012 >= 0.0417
MinimumLoad_013: v_computing_013 >= 0.0417
MinimumLoad_014: v_computing_014 >= 0.0417
MinimumLoad_015: v_computing_015 >= 0.0417
MinimumLoad_016: v_computing_016 >= 0.0417
MinimumLoad_017: v_computing_017 >= 0.0417
MinimumLoad_018: v_computing_018 >= 0.0417
MinimumLoad_019: v_computing_019 >= 0.0417
MinimumLoad_020: v_computing_020 >= 0.0417
MinimumLoad_021: v_computing_021 >= 0.0417
MinimumLoad_022: v_computing_022 >= 0.0417
MinimumLoad_023: v_computing_023 >= 0.0417
Minimum_SOC_00_000: b_soc_01_000 >= 20
Minimum_SOC_00_001: b_soc_01_001 >= 20
Minimum_SOC_00_002: b_soc_01_002 >= 20
Minimum_SOC_00_003: b_soc_01_003 >= 20
Minimum_SOC_00_004: b_soc_01_004 >= 20
Minimum_SOC_00_005: b_soc_01_005 >= 20
Minimum_SOC_00_006: b_soc_01_006 >= 20
Minimum_SOC_00_007: b_soc_01_007 >= 20
Minimum_SOC_00_008: b_soc_01_008 >= 20
Minimum_SOC_00_009: b_soc_01_009 >= 20
Minimum_SOC_00_010: b_soc_01_010 >= 20
Minimum_SOC_00_011: b_soc_01_011 >= 20
Minimum_SOC_00_012: b_soc_01_012 >= 20
Minimum_SOC_00_013: b_soc_01_013 >= 20
Minimum_SOC_00_014: b_soc_01_014 >= 20
Minimum_SOC_00_015: b_soc_01_015 >= 20
Minimum_SOC_00_016: b_soc_01_016 >= 20
Minimum_SOC_00_017: b_soc_01_017 >= 20
Minimum_SOC_00_018: b_soc_01_018 >= 20
Minimum_SOC_00_019: b_soc_01_019 >= 20
Minimum_SOC_00_020: b_soc_01_020 >= 20
Minimum_SOC_00_021: b_soc_01_021 >= 20
Minimum_SOC_00_022: b_soc_01_022 >= 20
Minimum_SOC_00_023: b_soc_01_023 >= 20
Minimum_SOC_01_000: b_soc_02_000 >= 5
Minimum_SOC_01_001: b_soc_02_001 >= 5
Minimum_SOC_01_002: b_soc_02_002 >= 5
Minimum_SOC_01_003: b_soc_02_003 >= 5
Minimum_SOC_01_004: b_soc_02_004 >= 5
Minimum_SOC_01_005: b_soc_02_005 >= 5
Minimum_SOC_01_006: b_soc_02_006 >= 5
Minimum_SOC_01_007: b_soc_02_007 >= 5
Minimum_SOC_01_008: b_soc_02_008 >= 5
Minimum_SOC_01_009: b_soc_02_009 >= 5
Minimum_SOC_01_010: b_soc_02_010 >= 5
Minimum_SOC_01_011: b_soc_02_011 >= 5
Minimum_SOC_01_012: b_soc_02_012 >= 5
Minimum_SOC_01_013: b_soc_02_013 >= 5
Minimum_SOC_01_014: b_soc_02_014 >= 5
Minimum_SOC_01_015: b_soc_02_015 >= 5
Minimum_SOC_01_016: b_soc_02_016 >= 5
Minimum_SOC_01_017: b_soc_02_017 >= 5
Minimum_SOC_01_018: b_soc_02_018 >= 5
Minimum_SOC_01_019: b_soc_02_019 >= 5
Minimum_SOC_01_020: b_soc_02_020 >= 5
Minimum_SOC_01_021: b_soc_02_021 >= 5
Minimum_SOC_01_022: b_soc_02_022 >= 5
Minimum_SOC_01_023: b_soc_02_023 >= 5
NominalLoad_000: v_computing_000 <= 1000
NominalLoad_001: v_computing_001 <= 1000
NominalLoad_002: v_computing_002 <= 1000
NominalLoad_003: v_computing_003 <= 1000
NominalLoad_004: v_computing_004 <= 1000
NominalLoad_005: v_computing_005 <= 1000
NominalLoad_006: v_computing_006 <= 1000
NominalLoad_007: v_computing_007 <= 1000
NominalLoad_008: v_computing_008 <= 1000
NominalLoad_009: v_computing_009 <= 1000
NominalLoad_010: v_computing_010 <= 1000
NominalLoad_011: v_computing_011 <= 1000
NominalLoad_012: v_computing_012 <= 1000
NominalLoad_013: v_computing_013 <= 1000
NominalLoad_014: v_computing_014 <= 1000
NominalLoad_015: v_computing_015 <= 1000
NominalLoad_016: v_computing_016 <= 1000
NominalLoad_017: v_computing_017 <= 1000
NominalLoad_018: v_computing_018 <= 1000
NominalLoad_019: v_computing_019 <= 1000
NominalLoad_020: v_computing_020 <= 1000
NominalLoad_021: v_computing_021 <= 1000
NominalLoad_022: v_computing_022 <= 1000
NominalLoad_023: v_computing_023 <= 1000
OptimalCooling_000: - 0.25 v_computing_000 + v_cooling_000 = 0
OptimalCooling_001: - 0.25 v_computing_001 + v_cooling_001 = 0
OptimalCooling_002: - 0.25 v_computing_002 + v_cooling_002 = 0
OptimalCooling_003: - 0.25 v_computing_003 + v_cooling_003 = 0
OptimalCooling_004: - 0.25 v_computing_004 + v_cooling_004 = 0
OptimalCooling_005: - 0.25 v_computing_005 + v_cooling_005 = 0
OptimalCooling_006: - 0.25 v_computing_006 + v_cooling_006 = 0
OptimalCooling_007: - 0.25 v_computing_007 + v_cooling_007 = 0
OptimalCooling_008: - 0.25 v_computing_008 + v_cooling_008 = 0
OptimalCooling_009: - 0.25 v_computing_009 + v_cooling_009 = 0
OptimalCooling_010: - 0.25 v_computing_010 + v_cooling_010 = 0
OptimalCooling_011: - 0.25 v_computing_011 + v_cooling_011 = 0
OptimalCooling_012: - 0.25 v_computing_012 + v_cooling_012 = 0
OptimalCooling_013: - 0.25 v_computing_013 + v_cooling_013 = 0
OptimalCooling_014: - 0.25 v_computing_014 + v_cooling_014 = 0
OptimalCooling_015: - 0.25 v_computing_015 + v_cooling_015 = 0
OptimalCooling_016: - 0.25 v_computing_016 + v_cooling_016 = 0
OptimalCooling_017: - 0.25 v_computing_017 + v_cooling_017 = 0
OptimalCooling_018: - 0.25 v_computing_018 + v_cooling_018 = 0
OptimalCooling_019: - 0.25 v_computing_019 + v_cooling_019 = 0
OptimalCooling_020: - 0.25 v_computing_020 + v_cooling_020 = 0
OptimalCooling_021: - 0.25 v_computing_021 + v_cooling_021 = 0
OptimalCooling_022: - 0.25 v_computing_022 + v_cooling_022 = 0
OptimalCooling_023: - 0.25 v_computing_023 + v_cooling_023 = 0
OptimalLoad_000: - v_computing_000 - v_cooling_000 + v_load_000 = 0
OptimalLoad_001: - v_computing_001 - v_cooling_001 + v_load_001 = 0
OptimalLoad_002: - v_computing_002 - v_cooling_002 + v_load_002 = 0
OptimalLoad_003: - v_computing_003 - v_cooling_003 + v_load_003 = 0
OptimalLoad_004: - v_computing_004 - v_cooling_004 + v_load_004 = 0
OptimalLoad_005: - v_computing_005 - v_cooling_005 + v_load_005 = 0
OptimalLoad_006: - v_computing_006 - v_cooling_006 + v_load_006 = 0
OptimalLoad_007: - v_computing_007 - v_cooling_007 + v_load_007 = 0
OptimalLoad_008: - v_computing_008 - v_cooling_008 + v_load_008 = 0
OptimalLoad_009: - v_computing_009 - v_cooling_009 + v_load_009 = 0
OptimalLoad_010: - v_computing_010 - v_cooling_010 + v_load_010 = 0
OptimalLoad_011: - v_computing_011 - v_cooling_011 + v_load_011 = 0
OptimalLoad_012: - v_computing_012 - v_cooling_012 + v_load_012 = 0
OptimalLoad_013: - v_computing_013 - v_cooling_013 + v_load_013 = 0
OptimalLoad_014: - v_computing_014 - v_cooling_014 + v_load_014 = 0
OptimalLoad_015: - v_computing_015 - v_cooling_015 + v_load_015 = 0
OptimalLoad_016: - v_computing_016 - v_cooling_016 + v_load_016 = 0
OptimalLoad_017: - v_computing_017 - v_cooling_017 + v_load_017 = 0
OptimalLoad_018: - v_computing_018 - v_cooling_018 + v_load_018 = 0
OptimalLoad_019: - v_computing_019 - v_cooling_019 + v_load_019 = 0
OptimalLoad_020: - v_computing_020 - v_cooling_020 + v_load_020 = 0
OptimalLoad_021: - v_computing_021 - v_cooling_021 + v_load_021 = 0
OptimalLoad_022: - v_computing_022 - v_cooling_022 + v_load_022 = 0
OptimalLoad_023: - v_computing_023 - v_cooling_023 + v_load_023 = 0
PCC_abs_limit_000: - 1000000000 pcc_pbin_000 + pcc_pin_000 <= 0
PCC_abs_limit_001: - 1000000000 pcc_pbin_001 + pcc_pin_001 <= 0
PCC_abs_limit_002: - 1000000000 pcc_pbin_002 + pcc_pin_002 <= 0
PCC_abs_limit_003: - 1000000000 pcc_pbin_003 + pcc_pin_003 <= 0
PCC_abs_limit_004: - 1000000000 pcc_pbin_004 + pcc_pin_004 <= 0
PCC_abs_limit_005: - 1000000000 pcc_pbin_005 + pcc_pin_005 <= 0
PCC_abs_limit_006: - 1000000000 pcc_pbin_006 + pcc_pin_006 <= 0
PCC_abs_limit_007: - 1000000000 pcc_pbin_007 + pcc_pin_007 <= 0
PCC_abs_limit_008: - 1000000000 pcc_pbin_008 + pcc_pin_008 <= 0
PCC_abs_limit_009: - 1000000000 pcc_pbin_009 + pcc_pin_009 <= 0
PCC_abs_limit_010: - 1000000000 pcc_pbin_010 + pcc_pin_010 <= 0
PCC_abs_limit_011: - 1000000000 pcc_pbin_011 + pcc_pin_011 <= 0
PCC_abs_limit_012: - 1000000000 pcc_pbin_012 + pcc_pin_012 <= 0
PCC_abs_limit_013: - 1000000000 pcc_pbin_013 + pcc_pin_013 <= 0
PCC_abs_limit_014: - 1000000000 pcc_pbin_014 + pcc_pin_014 <= 0
PCC_abs_limit_015: - 1000000000 pcc_pbin_015 + pcc_pin_015 <= 0
PCC_abs_limit_016: - 1000000000 pcc_pbin_016 + pcc_pin_016 <= 0
PCC_abs_limit_017: - 1000000000 pcc_pbin_017 + pcc_pin_017 <= 0
PCC_abs_limit_018: - 1000000000 pcc_pbin_018 + pcc_pin_018 <= 0
PCC_abs_limit_019: - 1000000000 pcc_pbin_019 + pcc_pin_019 <= 0
PCC_abs_limit_020: - 1000000000 pcc_pbin_020 + pcc_pin_020 <= 0
PCC_abs_limit_021: - 1000000000 pcc_pbin_021 + pcc_pin_021 <= 0
PCC_abs_limit_022: - 1000000000 pcc_pbin_022 + pcc_pin_022 <= 0
PCC_abs_limit_023: - 1000000000 pcc_pbin_023 + pcc_pin_023 <= 0
PCC_inj_limit_000: 1000000000 pcc_pbin_000 + pcc_pout_000 <= 1000000000
PCC_inj_limit_001: 1000000000 pcc_pbin_001 + pcc_pout_001 <= 1000000000
PCC_inj_limit_002: 1000000000 pcc_pbin_002 + pcc_pout_002 <= 1000000000
PCC_inj_limit_003: 1000000000 pcc_pbin_003 + pcc_pout_003 <= 1000000000
PCC_inj_limit_004: 1000000000 pcc_pbin_004 + pcc_pout_004 <= 1000000000
PCC_inj_limit_005: 1000000000 pcc_pbin_005 + pcc_pout_005 <= 1000000000
PCC_inj_limit_006: 1000000000 pcc_pbin_006 + pcc_pout_006 <= 1000000000
PCC_inj_limit_007: 1000000000 pcc_pbin_007 + pcc_pout_007 <= 1000000000
PCC_inj_limit_008: 1000000000 pcc_pbin_008 + pcc_pout_008 <= 1000000000
PCC_inj_limit_009: 1000000000 pcc_pbin_009 + pcc_pout_009 <= 1000000000
PCC_inj_limit_010: 1000000000 pcc_pbin_010 + pcc_pout_010 <= 1000000000
PCC_inj_limit_011: 1000000000 pcc_pbin_011 + pcc_pout_011 <= 1000000000
PCC_inj_limit_012: 1000000000 pcc_pbin_012 + pcc_pout_012 <= 1000000000
PCC_inj_limit_013: 1000000000 pcc_pbin_013 + pcc_pout_013 <= 1000000000
PCC_inj_limit_014: 1000000000 pcc_pbin_014 + pcc_pout_014 <= 1000000000
PCC_inj_limit_015: 1000000000 pcc_pbin_015 + pcc_pout_015 <= 1000000000
PCC_inj_limit_016: 1000000000 pcc_pbin_016 + pcc_pout_016 <= 1000000000
PCC_inj_limit_017: 1000000000 pcc_pbin_017 + pcc_pout_017 <= 1000000000
PCC_inj_limit_018: 1000000000 pcc_pbin_018 + pcc_pout_018 <= 1000000000
PCC_inj_limit_019: 1000000000 pcc_pbin_019 + pcc_pout_019 <= 1000000000
PCC_inj_limit_020: 1000000000 pcc_pbin_020 + pcc_pout_020 <= 1000000000
PCC_inj_limit_021: 1000000000 pcc_pbin_021 + pcc_pout_021 <= 1000000000
PCC_inj_limit_022: 1000000000 pcc_pbin_022 + pcc_pout_022 <= 1000000000
PCC_inj_limit_023: 1000000000 pcc_pbin_023 + pcc_pout_023 <= 1000000000
SOC_update_00_001: - b_energy_01_000 + b_energy_01_001 - 0.9 b_pch_01_001
 + 1.11111111111 b_pdis_01_001 = 0
SOC_update_00_002: - b_energy_01_001 + b_energy_01_002 - 0.9 b_pch_01_002
 + 1.11111111111 b_pdis_01_002 = 0
SOC_update_00_003: - b_energy_01_002 + b_energy_01_003 - 0.9 b_pch_01_003
 + 1.11111111111 b_pdis_01_003 = 0
SOC_update_00_004: - b_energy_01_003 + b_energy_01_004 - 0.9 b_pch_01_004
 + 1.11111111111 b_pdis_01_004 = 0
SOC_update_00_005: - b_energy_01_004 + b_energy_01_005 - 0.9 b_pch_01_005
 + 1.11111111111 b_pdis_01_005 = 0
SOC_update_00_006: - b_energy_01_005 + b_energy_01_006 - 0.9 b_pch_01_006
 + 1.11111111111 b_pdis_01_006 = 0
SOC_update_00_007: - b_energy_01_006 + b_energy_01_007 - 0.9 b_pch_01_007
 + 1.11111111111 b_pdis_01_007 = 0
SOC_update_00_008: - b_energy_01_007 + b_energy_01_008 - 0.9 b_pch_01_008
 + 1.11111111111 b_pdis_01_008 = 0
SOC_update_00_009: - b_energy_01_008 + b_energy_01_009 - 0.9 b_pch_01_009
 + 1.11111111111 b_pdis_01_009 = 0
SOC_update_00_010: - b_energy_01_009 + b_energy_01_010 - 0.9 b_pch_01_010
 + 1.11111111111 b_pdis_01_010 = 0
SOC_update_00_011: - b_energy_01_010 + b_energy_01_011 - 0.9 b_pch_01_011
 + 1.11111111111 b_pdis_01_011 = 0
SOC_update_00_012: - b_energy_01_011 + b_energy_01_012 - 0.9 b_pch_01_012
 + 1.11111111111 b_pdis_01_012 = 0
SOC_update_00_013: - b_energy_01_012 + b_energy_01_013 - 0.9 b_pch_01_013
 + 1.11111111111 b_pdis_01_013 = 0
SOC_update_00_014: - b_energy_01_013 + b_energy_01_014 - 0.9 b_pch_01_014
 + 1.11111111111 b_pdis_01_014 = 0
SOC_update_00_015: - b_energy_01_014 + b_energy_01_015 - 0.9 b_pch_01_015
 + 1.11111111111 b_pdis_01_015 = 0
SOC_update_00_016: - b_energy_01_015 + b_energy_01_016 - 0.9 b_pch_01_016
 + 1.11111111111 b_pdis_01_016 = 0
SOC_update_00_017: - b_energy_01_016 + b_energy_01_017 - 0.9 b_pch_01_017
 + 1.11111111111 b_pdis_01_017 = 0
SOC_update_00_018: - b_energy_01_017 + b_energy_01_018 - 0.9 b_pch_01_018
 + 1.11111111111 b_pdis_01_018 = 0
SOC_update_00_019: - b_energy_01_018 + b_energy_01_019 - 0.9 b_pch_01_019
 + 1.11111111111 b_pdis_01_019 = 0
SOC_update_00_020: - b_energy_01_019 + b_energy_01_020 - 0.9 b_pch_01_020
 + 1.11111111111 b_pdis_01_020 = 0
SOC_update_00_021: - b_energy_01_020 + b_energy_01_021 - 0.9 b_pch_01_021
 + 1.11111111111 b_pdis_01_021 = 0
SOC_update_00_022: - b_energy_01_021 + b_energy_01_022 - 0.9 b_pch_01_022
 + 1.11111111111 b_pdis_01_022 = 0
SOC_update_00_023: - b_energy_01_022 + b_energy_01_023 - 0.9 b_pch_01_023
 + 1.11111111111 b_pdis_01_023 = 0
SOC_update_01_001: - b_energy_02_000 + b_energy_02_001 - 0.8 b_pch_02_001
 + 1.25 b_pdis_02_001 = 0
SOC_update_01_002: - b_energy_02_001 + b_energy_02_002 - 0.8 b_pch_02_002
 + 1.25 b_pdis_02_002 = 0
SOC_update_01_003: - b_energy_02_002 + b_energy_02_003 - 0.8 b_pch_02_003
 + 1.25 b_pdis_02_003 = 0
SOC_update_01_004: - b_energy_02_003 + b_energy_02_004 - 0.8 b_pch_02_004
 + 1.25 b_pdis_02_004 = 0
SOC_update_01_005: - b_energy_02_004 + b_energy_02_005 - 0.8 b_pch_02_005
 + 1.25 b_pdis_02_005 = 0
SOC_update_01_006: - b_energy_02_005 + b_energy_02_006 - 0.8 b_pch_02_006
 + 1.25 b_pdis_02_006 = 0
SOC_update_01_007: - b_energy_02_006 + b_energy_02_007 - 0.8 b_pch_02_007
 + 1.25 b_pdis_02_007 = 0
SOC_update_01_008: - b_energy_02_007 + b_energy_02_008 - 0.8 b_pch_02_008
 + 1.25 b_pdis_02_008 = 0
SOC_update_01_009: - b_energy_02_008 + b_energy_02_009 - 0.8 b_pch_02_009
 + 1.25 b_pdis_02_009 = 0
SOC_update_01_010: - b_energy_02_009 + b_energy_02_010 - 0.8 b_pch_02_010
 + 1.25 b_pdis_02_010 = 0
SOC_update_01_011: - b_energy_02_010 + b_energy_02_011 - 0.8 b_pch_02_011
 + 1.25 b_pdis_02_011 = 0
SOC_update_01_012: - b_energy_02_011 + b_energy_02_012 - 0.8 b_pch_02_012
 + 1.25 b_pdis_02_012 = 0
SOC_update_01_013: - b_energy_02_012 + b_energy_02_013 - 0.8 b_pch_02_013
 + 1.25 b_pdis_02_013 = 0
SOC_update_01_014: - b_energy_02_013 + b_energy_02_014 - 0.8 b_pch_02_014
 + 1.25 b_pdis_02_014 = 0
SOC_update_01_015: - b_energy_02_014 + b_energy_02_015 - 0.8 b_pch_02_015
 + 1.25 b_pdis_02_015 = 0
SOC_update_01_016: - b_energy_02_015 + b_energy_02_016 - 0.8 b_pch_02_016
 + 1.25 b_pdis_02_016 = 0
SOC_update_01_017: - b_energy_02_016 + b_energy_02_017 - 0.8 b_pch_02_017
 + 1.25 b_pdis_02_017 = 0
SOC_update_01_018: - b_energy_02_017 + b_energy_02_018 - 0.8 b_pch_02_018
 + 1.25 b_pdis_02_018 = 0
SOC_update_01_019: - b_energy_02_018 + b_energy_02_019 - 0.8 b_pch_02_019
 + 1.25 b_pdis_02_019 = 0
SOC_update_01_020: - b_energy_02_019 + b_energy_02_020 - 0.8 b_pch_02_020
 + 1.25 b_pdis_02_020 = 0
SOC_update_01_021: - b_energy_02_020 + b_energy_02_021 - 0.8 b_pch_02_021
 + 1.25 b_pdis_02_021 = 0
SOC_update_01_022: - b_energy_02_021 + b_energy_02_022 - 0.8 b_pch_02_022
 + 1.25 b_pdis_02_022 = 0
SOC_update_01_023: - b_energy_02_022 + b_energy_02_023 - 0.8 b_pch_02_023
 + 1.25 b_pdis_02_023 = 0
TotalLoad_000: - p_cooling_000 + p_total_000 = 0.0417
TotalLoad_001: - p_cooling_001 + p_total_001 = 0.0417
TotalLoad_002: - p_cooling_002 + p_total_002 = 0.0417
TotalLoad_003: - p_cooling_003 + p_total_003 = 0.0417
TotalLoad_004: - p_cooling_004 + p_total_004 = 0.0417
TotalLoad_005: - p_cooling_005 + p_total_005 = 0.0417
TotalLoad_006: - p_cooling_006 + p_total_006 = 0.0417
TotalLoad_007: - p_cooling_007 + p_total_007 = 0.0417
TotalLoad_008: - p_cooling_008 + p_total_008 = 0.0417
TotalLoad_009: - p_cooling_009 + p_total_009 = 0.0417
TotalLoad_010: - p_cooling_010 + p_total_010 = 0.0417
TotalLoad_011: - p_cooling_011 + p_total_011 = 0.0417
TotalLoad_012: - p_cooling_012 + p_total_012 = 0.0417
TotalLoad_013: - p_cooling_013 + p_total_013 = 0.0417
TotalLoad_014: - p_cooling_014 + p_total_014 = 0.0417
TotalLoad_015: - p_cooling_015 + p_total_015 = 0.0417
TotalLoad_016: - p_cooling_016 + p_total_016 = 0.0417
TotalLoad_017: - p_cooling_017 + p_total_017 = 0.0417
TotalLoad_018: - p_cooling_018 + p_total_018 = 0.0417
TotalLoad_019: - p_cooling_019 + p_total_019 = 0.0417
TotalLoad_020: - p_cooling_020 + p_total_020 = 0.0417
TotalLoad_021: - p_cooling_021 + p_total_021 = 0.0417
TotalLoad_022: - p_cooling_022 + p_total_022 = 0.0417
TotalLoad_023: - p_cooling_023 + p_total_023 = 0.0417
_C1: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C10: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C11: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C12: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C13: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C14: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C15: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C16: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C17: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C18: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C19: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C2: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C20: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C21: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C22: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C23: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C24: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C3: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C4: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C5: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C6: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C7: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C8: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
_C9: - p_total_000 - p_total_001 - p_total_002 - p_total_003 - p_total_004
 - p_total_005 - p_total_006 - p_total_007 - p_total_008 - p_total_009
 - p_total_010 - p_total_011 - p_total_012 - p_total_013 - p_total_014
 - p_total_015 - p_total_016 - p_total_017 - p_total_018 - p_total_019
 - p_total_020 - p_total_021 - p_total_022 - p_total_023 + v_load_000
 + v_load_001 + v_load_002 + v_load_003 + v_load_004 + v_load_005 + v_load_006
 + v_load_007 + v_load_008 + v_load_009 + v_load_010 + v_load_011 + v_load_012
 + v_load_013 + v_load_014 + v_load_015 + v_load_016 + v_load_017 + v_load_018
 + v_load_019 + v_load_020 + v_load_021 + v_load_022 + v_load_023 = 0
Binaries
b_pbin_01_000
b_pbin_01_001
b_pbin_01_002
b_pbin_01_003
b_pbin_01_004
b_pbin_01_005
b_pbin_01_006
b_pbin_01_007
b_pbin_01_008
b_pbin_01_009
b_pbin_01_010
b_pbin_01_011
b_pbin_01_012
b_pbin_01_013
b_pbin_01_014
b_pbin_01_015
b_pbin_01_016
b_pbin_01_017
b_pbin_01_018
b_pbin_01_019
b_pbin_01_020
b_pbin_01_021
b_pbin_01_022
b_pbin_01_023
b_pbin_02_000
b_pbin_02_001
b_pbin_02_002
b_pbin_02_003
b_pbin_02_004
b_pbin_02_005
b_pbin_02_006
b_pbin_02_007
b_pbin_02_008
b_pbin_02_009
b_pbin_02_010
b_pbin_02_011
b_pbin_02_012
b_pbin_02_013
b_pbin_02_014
b_pbin_02_015
b_pbin_02_016
b_pbin_02_017
b_pbin_02_018
b_pbin_02_019
b_pbin_02_020
b_pbin_02_021
b_pbin_02_022
b_pbin_02_023
pcc_pbin_000
pcc_pbin_001
pcc_pbin_002
pcc_pbin_003
pcc_pbin_004
pcc_pbin_005
pcc_pbin_006
pcc_pbin_007
pcc_pbin_008
pcc_pbin_009
pcc_pbin_010
pcc_pbin_011
pcc_pbin_012
pcc_pbin_013
pcc_pbin_014
pcc_pbin_015
pcc_pbin_016
pcc_pbin_017
pcc_pbin_018
pcc_pbin_019
pcc_pbin_020
pcc_pbin_021
pcc_pbin_022
pcc_pbin_023
End
