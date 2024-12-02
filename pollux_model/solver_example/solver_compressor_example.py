from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from pollux_model.power_supply_demand.power_supply import PowerSupply
from pollux_model.power_supply_demand.power_demand import PowerDemand
from pollux_model.hydrogen_demand.hydrogen_demand import HydrogenDemand
from pollux_model.splitter.splitter import Splitter
from pollux_model.adder.adder import Adder
# from pollux_model.electrolyser.electrolyser_physics_based import ElectrolyserDeGroot
from pollux_model.electrolyser.electrolyser_physics_based_with_compressor \
    import ElectrolyserWithCompressor
from pollux_model.gas_storage.hydrogen_tank_model import HydrogenTankModel
from pollux_model.compressor.compressor import Compressor
from pollux_model.solver.solver import Solver
from pollux_model.solver.step_function import StepFunction
from pollux_model.solver.key_performance_indicators import Objective, rmse

test_folder = "C:\\Clones\\pollux\\pollux-model\\pollux_model\\solver_example\\"
test_file = "optimization_problem_test4.json"
with open(test_folder + test_file, 'r') as file:
    problem = json.load(file)

##########################################################################
# Setting up time arrays for user input profiles and for the control
##########################################################################
time_horizon = problem["time_horizon"]  # hours (integer)
step_size_control = problem["step_size_control"]  # time_horizon/step_size_control (integer)
if time_horizon % step_size_control != 0:
    raise ValueError(f"time_horizon ({time_horizon}) is not divisible \
                      by step_size_control ({step_size_control})")

# time_vector = np.linspace(0, time_horizon, 97) # stepsize=0.25 hrs
time_vector = np.linspace(0, time_horizon, time_horizon+1)[:-1]  # stepsize=1 hrs
# time_vector_control = np.linspace(0, time_horizon, time_horizon//step_size_control + 1)[:-1]
# if using stairs instead of step for plotting:
time_vector_control = np.linspace(0, time_horizon, time_horizon//step_size_control + 1)
zeros_array = np.zeros(len(time_vector_control))

##########################################################################
# Defining time profiles for supply and demand
##########################################################################


# power supply
def power_supply_profile(t): return 10E6 * (2 + np.sin(t))  # Watt


# power_supply_profile = lambda t: 10E6 * 2 * (t+1)/(t+1) # Watt constant profile for testing
power_supply = PowerSupply(power_supply_profile)


# power demand
def power_demand_profile(t): return 10E6  # Watt


power_demand = PowerDemand(power_demand_profile)


# hydrogen demand
def hydrogen_demand_profile(t): return 200/3600  # kg/s


hydrogen_demand = HydrogenDemand(hydrogen_demand_profile)


##########################################################################
# Setting up the components
##########################################################################

# splitter1
step_function = StepFunction(zeros_array, step_size_control)
splitter1 = Splitter(step_function)

# splitter2
step_function = StepFunction(zeros_array, step_size_control)
splitter2 = Splitter(step_function)

# electrolyser
# electrolyser = ElectrolyserDeGroot()
electrolyser = ElectrolyserWithCompressor()

param = dict()
param['T_cell'] = 273.15 + 40  # cell temperature in K
param['p_cathode'] = 10e5  # cathode pressure in Pa
param['p_anode'] = 10e5  # anode pressure in Pa
param['p_0_H2O'] = 10e5  # Pa
param['eta_Faraday_array'] = 1  # just a constant, in reality is a variable
param['Faraday_const'] = 96485.3329  # Faraday constant [(s A)/mol]
param['delta_t'] = np.diff(time_vector)[0]*3600  # 3600  # timestep in seconds
param['A_cell'] = 0.436  # area in m2
param['cell_type'] = 'low_power_cell'
param['capacity'] = 100 * 1e6  # capacity in Watt

# specific_heat_ratio (Cp/Cv) also called gas isentropic coefficient
param['specific_heat_ratio'] = 1.41  # = 1.41 for hydrogen
param['inlet_temperature'] = 298  # K
param['inlet_pressure'] = 1E5  # Pa
param['outlet_pressure'] = 20E6  # Pa
param['R'] = 4124.2  # J/(kg K), gas constant = 4124 for hydrogen
param['Z'] = 1  # Z = 1 assuming ideal gas
param['mechanical_efficiency'] = 0.97
param['compressor_efficiency'] = 0.88
param['number_of_stages'] = 2

electrolyser.update_parameters(param)

# compressor
compressor = Compressor()
param = dict()
# specific_heat_ratio (Cp/Cv) also called gas isentropic coefficient
param['specific_heat_ratio'] = 1.41  # = 1.41 for hydrogen
param['inlet_temperature'] = 298  # K
param['inlet_pressure'] = 1E5  # Pa
param['outlet_pressure'] = 20E6  # Pa
param['R'] = 4124.2  # J/(kg K), gas constant = 4124 for hydrogen
param['Z'] = 1  # Z = 1 assuming ideal gas
param['mechanical_efficiency'] = 0.97
param['compressor_efficiency'] = 0.88
param['number_of_stages'] = 2
compressor.update_parameters(param)

# storage
step_function = StepFunction(zeros_array, step_size_control)
hydrogen_storage = HydrogenTankModel(step_function)

param = dict()
param['timestep'] = np.diff(time_vector)[0]*3600  # should be taken equal to delta_t
param['maximum_capacity'] = 5000  # kg
param['initial_mass'] = 1000.0  # kg
hydrogen_storage.update_parameters(param)

# adder
adder = Adder()
u = dict()
u['input_0'] = 0
u['input_1'] = 0
adder.input = u

# A list to retrieve object by their names. Specific order of components is not relevant.
components = {
    "power_supply": power_supply,
    "power_demand": power_demand,
    "splitter1": splitter1,
    "electrolyser": electrolyser,
    "splitter2": splitter2,
    "adder": adder,
    "hydrogen_storage": hydrogen_storage,
    "hydrogen_demand": hydrogen_demand,
    "compressor": compressor
}

##########################################################################
# Setting up the system and solve
##########################################################################

# List of components with control. For now: number of controls per component are equal
components_with_control = problem["controls"]["components"]

# Initial control
# initial guess. This is the super control vector containing all control variables
control_init = np.array(problem['controls']['init'])

# The controls of the components are updated with values provided by the json input file
number_of_components_with_control = len(components_with_control)
control_reshaped = control_init.reshape(number_of_components_with_control, -1)
for ii in range(number_of_components_with_control):
    components[components_with_control[ii]].update_time_function(control_reshaped[ii])

# Solver object
solver = Solver(time_vector, components, components_with_control)

# Connect the components.
# Ordering is important here. For now we assume a fixed configuration.
# The ordering can be calculated to generalise the implementation but is not done yet.
# solver.connect(predecessor,     successor,        'predecessor_output', 'successor_input')
solver.connect(power_supply,     splitter1,        'power_supply',  'input')
solver.connect(splitter1,        power_demand,     'output_0',      'power_input')

#  ## A: no compressor
# solver.connect(splitter1,        electrolyser,     'output_1',      'power_input')
# solver.connect(electrolyser,     splitter2,        'massflow_H2',   'input')
#  ## B: compressor independent component
# solver.connect(splitter1,        electrolyser,      'output_1',      'power_input')
# solver.connect(electrolyser,     compressor,        'massflow_H2',   'mass_flow')
# solver.connect(compressor,       splitter2,         'mass_flow',     'input')
#  ## C: compressor integrated with electrolyser
solver.connect(splitter1,        electrolyser,     'output_1',      'power_input')
solver.connect(electrolyser,     splitter2,        'massflow_H2',   'input')

solver.connect(splitter2,        adder,            'output_0',      'input_0')
solver.connect(splitter2,        hydrogen_storage, 'output_1',      'mass_flow_in')
solver.connect(hydrogen_storage, adder,            'mass_flow_out', 'input_1')
solver.connect(adder,            hydrogen_demand,  'output',        'hydrogen_input')

objective_name = problem["objective"]["name"]
if objective_name == "":
    ##########################################################################
    # Run the solver, loop over time
    ##########################################################################

    solver.run(control_init)  # run the solver with the initial values
else:
    ##########################################################################
    # Run an optimisation
    ##########################################################################

    # bounds
    scaling_factor = np.array(problem['controls']['ub'])
    bounds = [(problem['controls']['lb'][ii]/scaling_factor[ii],
               problem['controls']['ub'][ii]/scaling_factor[ii])
              for ii in range(len(problem['controls']['init']))]
    control_init_scaled = [(control_init[ii]/scaling_factor[ii])
                           for ii in range(len(control_init))]
    objective_label = problem["objective"]["label"]  # Just for plotting

    # objective function
    objective = Objective(solver, scaling_factor)
    objective_function = getattr(objective, objective_name)
    # Note: objective function is function of scaled control

    method = problem['optimisation']['optimiser']
    # method='trust-constr', 'SLSQP', 'L-BFGS-B', 'Nelder-Mead'

    function_value = []
    control_scaled_value = []
    match method:
        case "trust-constr":
            def call_back(x, convergence):
                function_value.append(objective_function(x))
                control_scaled_value.append(x)

            result = minimize(objective_function, control_init_scaled, method=method, tol=1E-6,
                              options={'maxiter': problem["optimisation"]["maxiter"],
                                       'verbose': True, 'disp': True,
                                       'finite_diff_rel_step':
                                       problem["optimisation"]["finite_diff_step"],
                                       },
                              bounds=bounds, callback=call_back)

        case "L-BFGS-B":
            def call_back(x):
                function_value.append(objective_function(x))
                control_scaled_value.append(x)

            result = minimize(objective_function, control_init_scaled, method=method,
                              options={'maxiter': problem["optimisation"]["maxiter"],
                                       'verbose': True, 'disp': True,
                                       # absolute stepsize:
                                       'eps': problem["optimisation"]["finite_diff_step"],
                                       'ftol': 1E-16
                                       },
                              bounds=bounds, callback=call_back)
        case "Powell":
            # TNC does only do one iteration, not clear why. dont use it for now!
            # Powell has a reporting issue result.x does not reproduce reported
            # minimal objective function value. below an attempt to repair it in
            # the callback function. Powell seems however very efficient.
            def call_back(x):
                function_value.append(objective_function(x))
                iteration = len(function_value)
                if iteration >= 3:
                    minimum = min(function_value[:-2])
                    if function_value[-1] > minimum:
                        control_scaled_value.append(control_scaled_value[-1])
                    else:
                        control_scaled_value.append(x)
                else:
                    control_scaled_value.append(x)

            result = minimize(objective_function, control_init_scaled, method=method,
                              options={'maxiter': problem["optimisation"]["maxiter"],
                                       'verbose': True, 'disp': True,
                                       # absolute stepsize
                                       'eps': problem["optimisation"]["finite_diff_step"],
                                       'return_all': True
                                       # 'maxfun': 10000,
                                       # 'stpmax': 0.001
                                       },
                              bounds=bounds, callback=call_back)

        case "differential-evolution":
            def call_back(x, convergence=None):
                function_value.append(objective_function(x))
                control_scaled_value.append(x)

            result = differential_evolution(objective_function,
                                            maxiter=problem["optimisation"]["maxiter"],
                                            bounds=bounds, callback=call_back)

    function_value = np.array(function_value)
    control_scaled_value = np.array(control_scaled_value)
    print(f"Objective function values: {function_value}")
    control_scaled = result.x  # optimized control
    control = np.array([control_scaled[ii] * scaling_factor[ii]
                        for ii in range(len(control_scaled))])

    number_of_components_with_control = len(components_with_control)
    control_reshaped = control.reshape(number_of_components_with_control, -1)
    for ii in range(number_of_components_with_control):
        print(f"control variables {components_with_control[ii]}")
        print(f"Optimized solution: {control_reshaped[ii]}")
    print(f"Objective function value: {result.fun}")

    # Do a run with optimized control
    solver.run(control)

##########################################################################
# Retrieve simulation input and output profiles
##########################################################################

power_supply_outputs = solver.outputs[power_supply]
splitter1_outputs = solver.outputs[splitter1]
power_demand_outputs = solver.outputs[power_demand]
hydrogen_demand_outputs = solver.outputs[hydrogen_demand]
electrolyser_outputs = solver.outputs[electrolyser]
# compressor_outputs = solver.outputs[compressor]
splitter2_outputs = solver.outputs[splitter2]
hydrogen_storage_outputs = solver.outputs[hydrogen_storage]
adder_outputs = solver.outputs[adder]

# Power profiles [MW]
power_supply = [row[0] * 1E-6 for row in power_supply_outputs]
power_delivered = [row[0] * 1E-6 for row in splitter1_outputs]  # output_0
electrolyser_power_input = [row[1] * 1E-6 for row in splitter1_outputs]  # output_1

power_electrolyser = [row[15] * 1E-6 for row in electrolyser_outputs]  # power_electrolyser
power_compressor = [row[16] * 1E-6 for row in electrolyser_outputs]  # power_compressor

power_demand = [row[0] * 1E-6 for row in power_demand_outputs]
# power_difference = [row[1] * 1E-6 for row in power_demand_outputs]
power_difference = [(power_demand[ii] - power_delivered[ii])/power_demand[ii]
                    for ii in range(len(power_demand))]
# power_compressor = [row[0] * 1E-6 for row in compressor_outputs]

# Hydrogen profiles
hydrogen_electrolyser_mass_flow_out = [row[3]*3600 for row in electrolyser_outputs]  # massflow_H2
hydrogen_electrolyser_to_demand = [row[0]*3600 for row in splitter2_outputs]  # output_0
hydrogen_electrolyser_to_storage = [row[1]*3600 for row in splitter2_outputs]  # output_1
hydrogen_demand = [row[0]*3600 for row in hydrogen_demand_outputs]
hydrogen_delivered = [row[0]*3600 for row in adder_outputs]
hydrogen_difference = [(hydrogen_demand[ii] - hydrogen_delivered[ii])/hydrogen_demand[ii]
                       for ii in range(len(hydrogen_demand))]

# Hydrogen storage profiles
hydrogen_mass_stored = [row[0] for row in hydrogen_storage_outputs]
fill_level = [row[1]*100 for row in hydrogen_storage_outputs]
hydrogen_storage_mass_flow_out = [row[2]*3600 for row in hydrogen_storage_outputs]
hydrogen_storage_mass_flow_in = [row[1]*3600 for row in splitter2_outputs]  # output_1

# KPI profiles
# conversion factor kg (H2)/hr to MW=MJ/s
conversion_factor_hydrogen = 0.0333
efficiency_electrolyser = [100 * conversion_factor_hydrogen *
                           hydrogen_electrolyser_mass_flow_out[ii]/electrolyser_power_input[ii]
                           for ii in range(len(electrolyser_power_input))]

##########################################################################
# Calculate KPIs
##########################################################################

# KPIs
kpi_rmse_power_demand = rmse(power_demand, power_delivered)  # MW
kpi_rmse_hydrogen_demand = rmse(hydrogen_demand, hydrogen_delivered)  # [kg/hr]
kpi_rmse_hydrogen_demand = 0.03333 * kpi_rmse_hydrogen_demand
kpi_rmse_demand = kpi_rmse_power_demand + kpi_rmse_hydrogen_demand

print(f"KPI, rmse of power demand and power delivered {kpi_rmse_power_demand} [MW]")
print(f"KPI, rmse of hydrogen demand and hydrogen delivered {kpi_rmse_hydrogen_demand} [MW]")
print(f"KPI, sum of rmse of demand {kpi_rmse_demand} [MW]")

electrolyser_power_input_sum = sum(electrolyser_power_input)
hydrogen_electrolyser_mass_flow_out_sum = 0.03333*sum(hydrogen_electrolyser_mass_flow_out)
efficiency_electrolyser_total = 100 * hydrogen_electrolyser_mass_flow_out_sum / \
                                electrolyser_power_input_sum
print(f"KPI, Efficiency electrolyser {efficiency_electrolyser_total} [-]")

##########################################################################
# Plotting
##########################################################################
plt.ioff()
if objective_name != "":
    # Optimization process
    plt.subplots(1, 1, sharex=True, figsize=(10, 6))
    ax = plt.subplot(111)
    ax.plot(function_value)
    plt.ylabel(objective_label)
    plt.title('Objective function')
    plt.xlabel('Iteration')
    ax.grid(axis='y', alpha=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    fig.suptitle('Scaled Control', fontsize=14)
    n = len(time_vector_control)-1
    ax = plt.subplot(311)
    lines = plt.plot(control_scaled_value[:, :n])
    plt.ylabel('Control values')
    # plt.xlabel('Iteration')
    plt.title(components_with_control[0])
    ax.grid(axis='y', alpha=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = plt.subplot(312)
    lines = plt.plot(control_scaled_value[:, n:2*n])
    plt.ylabel('Control values')
    # plt.xlabel('Iteration')
    plt.title(components_with_control[1])
    ax.grid(axis='y', alpha=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = plt.subplot(313)
    lines = plt.plot(control_scaled_value[:, 2*n:])
    plt.ylabel('Control values')
    plt.xlabel('Iteration')
    plt.title(components_with_control[2])
    ax.grid(axis='y', alpha=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Control figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.stairs(control_reshaped[0], time_vector_control, color='r', label=components_with_control[0])
ax1.stairs(control_reshaped[1], time_vector_control, color='r', label=components_with_control[1],
           linestyle='--')
# ax1.step(time_vector_control, control_reshaped[0], color='r', where='post',
#          label=components_with_control[0])
# ax1.step(time_vector_control, control_reshaped[1], color='r',  where='post',
#          label=components_with_control[1], linestyle='--')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Splitter Control [-]', color='r')
ax1.legend(loc='upper left')
ax1.set_xticks(time_vector_control)
plt.grid(True)
ax2 = ax1.twinx()
ax2.stairs(3600*control_reshaped[2], time_vector_control, color='b',
           label='Hydrogen Storage mass flow out')
# ax2.step(time_vector_control, 3600*control_reshaped[2], color='b',  where='post',
#          label='Hydrogen Storage mass flow out')
ax2.legend(loc='upper right')
ax2.set_ylabel('Storage production rate [kg/hr]', color='b')
plt.title('Control Profiles')
plt.grid(True)

# Power figure
fig = plt.figure(figsize=(10, 6))
plt.step(time_vector, power_supply, color='r', where='post', linewidth=2, label='Power Supply')
plt.step(time_vector, power_delivered, color='g', where='post', label='Power Delivered')
plt.step(time_vector, electrolyser_power_input, color='c', where='post',
         label='Power Electrolyser+Compressor')
plt.step(time_vector, power_electrolyser, color='y', where='post', linewidth=2,
         label='Power Electrolyser')
plt.step(time_vector, power_compressor, color='m', where='post', linewidth=2,
         label='Power Compressor')
# just for checking, should be equal to power supply:
# sum = [x + y for x, y in zip(power_delivered, electrolyser_power_input)]
# plt.step(time_vector, sum, label='sum', linestyle='--')
plt.step(time_vector, power_demand, color='b', where='post', linewidth=2, label='Power Demand')

# plt.step(time_vector, power_difference, color='m', label='Demand - Delivered')
plt.xlabel('Time (hr)')
plt.ylabel('Power [MW]')
# plt.xticks(time_vector_control)
plt.title('Power Profiles')
plt.legend()
plt.grid(True)

# Hydrogen figure
fig = plt.figure(figsize=(10, 6))
# plt.step(time_vector, hydrogen_electrolyser_mass_flow_out,  color='r',
#          label='Hydrogen Electrolyser output')
plt.step(time_vector, hydrogen_electrolyser_to_demand, color='c', where='post',
         label='Hydrogen from Electrolyser to Demand')
plt.step(time_vector, hydrogen_electrolyser_to_storage, color='m',  where='post',
         label='Hydrogen from Electrolyser to Storage')
plt.step(time_vector, hydrogen_storage_mass_flow_out, color='y', where='post',
         label='Hydrogen from Storage to Demand')
plt.step(time_vector, hydrogen_demand,  color='b', where='post', linewidth=2,
         label='Hydrogen Demand')
plt.step(time_vector, hydrogen_delivered, color='g', where='post', label='Hydrogen Delivered')
# plt.step(time_vector, hydrogen_difference, color='m', label='Demand - Delivered')
plt.xlabel('Time (hr)')
plt.ylabel('Hydrogen flow [kg/hr]')
plt.title('Hydrogen Profiles')
plt.legend()
plt.grid(True)

# Mismatch figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.step(time_vector, power_difference, color='r', where='post', label='Power')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Power [MW]', color='r')
ax1.legend(loc='upper left')
ax1.set_ylim(-1, 1)
ax2 = ax1.twinx()
ax2.step(time_vector, hydrogen_difference, color='b', where='post', label='Hydrogen')
ax2.set_ylabel('Hydrogen [kg/hr]', color='b')
ax2.legend(loc='upper right')
ax2.set_ylim(-1, 1)
plt.title('(Demand - Delivered)/Demand')
plt.grid(True)

# KPI figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.step(time_vector, efficiency_electrolyser, color='r', where='post', label='Efficiency %')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Efficiency [%]', color='r')
ax1.legend(loc='upper left')
# ax1.set_ylim(-1, 1)
# ax2=ax1.twinx()
# ax2.step(time_vector, hydrogen_difference, color='b', where='post', label='electrolyser')
# ax2.set_ylabel('COP [-]', color='b')
# ax2.legend(loc='upper right')
# ax2.set_ylim(-1, 1)
plt.title('Electrolyser')
plt.grid(True)

# Storage figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.step(time_vector, hydrogen_mass_stored, color='r', where='post', label='Hydrogen Mass Stored')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Hydrogen Mass Stored [kg]', color='r')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.step(time_vector, fill_level, color='k', where='post', label='Fill Level %')
ax2.step(time_vector, hydrogen_storage_mass_flow_in, color='b', where='post',
         label='Hydrogen Storage mass flow in')
ax2.step(time_vector, hydrogen_storage_mass_flow_out, color='g', where='post',
         label='Hydrogen Storage mass flow out')
ax2.set_ylabel('Fill Level [%] / Hydrogen mass flow [kg/hr]', color='b')
plt.title('Hydrogen Storage profiles')
ax2.legend(loc='upper right')
plt.grid(True)

plt.show()

# Write output to files
df = pd.DataFrame(control_reshaped, index=['splitter1', 'splitter2', 'hydrogen_storage'])
# Save the DataFrame to a CSV file
df.to_csv('control.csv', index=True)
