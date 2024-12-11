from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from pollux_model.power_supply.power_supply_profiles import PowerSupply
from pollux_model.power_demand.power_demand_profiles import PowerDemand
from pollux_model.heat_demand.heat_demand_profiles import HeatDemand
from pollux_model.splitter.splitter import Splitter
from pollux_model.heat_pump.heat_pump_physics_based import HeatpumpNREL
from pollux_model.solver.solver import Solver
from pollux_model.solver.step_function import StepFunction
from pollux_model.solver.key_performance_indicators import Objective, rmse
import os

test_folder = os.path.dirname(__file__)
test_file = "run_P2Heat_test1.json"
with open(os.path.join(test_folder, test_file), 'r') as file:
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
power_supply = PowerSupply()
power_supply.set_time_function(power_supply_profile)


# power demand
def power_demand_profile(t): return 10E6  # Watt


power_demand = PowerDemand()
power_demand.set_time_function(power_demand_profile)


# hydrogen demand
def heat_demand_profile(t): return 10E6  # W


heat_demand = HeatDemand(heat_demand_profile)
heat_demand.set_time_function(heat_demand_profile)

##########################################################################
# Setting up the components
##########################################################################

# splitter1
step_function = StepFunction(zeros_array, step_size_control)
splitter1 = Splitter()
splitter1.set_time_function(step_function)

# heat pump
heatpump = HeatpumpNREL()
param = dict()
param['second_law_efficiency_flag'] = False
param['print_results'] = False
param['refrigerant_flag'] = True
param['refrigerant'] = 'R365MFC'
heatpump.update_parameters(param)

u = dict()
u['hot_temperature_desired'] = 150
u['hot_temperature_return'] = 100
u['cold_temperature_available'] = 80
u['cold_deltaT'] = 40
u['process_heat_requirement'] = 'NaN'
u['hot_mass_flowrate'] = 'NaN'
u['electricity_power_in'] = 1E5
heatpump.input = u

# A list to retrieve object by their names. Specific order of components is not relevant.
components = {
    "power_supply": power_supply,
    "power_demand": power_demand,
    "splitter1": splitter1,
    "heatpump": heatpump,
    "heat_demand": heat_demand
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
solver.connect(splitter1,        heatpump,         'output_1',      'electricity_power_in')
solver.connect(heatpump,         heat_demand,      'process_heat_requirement', 'heat_input')

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
heatpump_outputs = solver.outputs[heatpump]
heat_demand_outputs = solver.outputs[heat_demand]

# Power profiles [MW]
power_supply = [row[0] * 1E-6 for row in power_supply_outputs]
power_delivered = [row[0] * 1E-6 for row in splitter1_outputs]  # output_0
heatpump_power_input = [row[1] * 1E-6 for row in splitter1_outputs]  # output_1
power_demand = [row[0] * 1E-6 for row in power_demand_outputs]
# power_difference = [row[1] * 1E-6 for row in power_demand_outputs]
power_difference = [(power_demand[ii] - power_delivered[ii])/power_demand[ii]
                    for ii in range(len(power_demand))]

# Heat profiles
heat_delivered = [row[5]*1E-6 for row in heatpump_outputs]  # heat [MW]
heat_demand = [row[0]*1E-6 for row in heat_demand_outputs]
heat_difference = [(heat_demand[ii] - heat_delivered[ii])/heat_demand[ii]
                   for ii in range(len(heat_demand))]


##########################################################################
# Calculate KPIs
##########################################################################

# KPIs
kpi_rmse_power_demand = rmse(power_demand, power_delivered)  # MW
kpi_rmse_heat_demand = rmse(heat_demand, heat_delivered)  # [kg/hr]
kpi_rmse_demand = kpi_rmse_power_demand + kpi_rmse_heat_demand

print(f"KPI, rmse of power demand and power delivered {kpi_rmse_power_demand} [MW]")
print(f"KPI, rmse of heat demand and heat delivered {kpi_rmse_heat_demand} [MW]")
print(f"KPI, sum of rmse of demand {kpi_rmse_demand} [MW]")

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
    fig.suptitle('Scaled Control', fontsize=16)
    n = len(time_vector_control)-1
    ax = plt.subplot(311)
    lines = plt.plot(control_scaled_value[:, :n])
    plt.ylabel('Control values')
    # plt.xlabel('Iteration')
    plt.title(components_with_control[0])
    ax.grid(axis='y', alpha=0.75)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax = plt.subplot(312)
    # lines = plt.plot(control_scaled_value[:, n:2*n])
    # plt.ylabel('Control values')
    # # plt.xlabel('Iteration')
    # plt.title(components_with_control[1])
    # ax.grid(axis='y', alpha=0.75)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax = plt.subplot(313)
    # lines = plt.plot(control_scaled_value[:, 2*n:])
    # plt.ylabel('Control values')
    # plt.xlabel('Iteration')
    # plt.title(components_with_control[2])
    # ax.grid(axis='y', alpha=0.75)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Control figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.stairs(control_reshaped[0], time_vector_control, color='r', label=components_with_control[0])
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Splitter Control [-]', color='r')
ax1.legend(loc='upper left')
ax1.set_xticks(time_vector_control)
plt.grid(True)
# ax2 = ax1.twinx()
# ax2.stairs(3600*control_reshaped[2], time_vector_control, color='b',
#            label='Hydrogen Storage mass flow out')
# # ax2.step(time_vector_control, 3600*control_reshaped[2], color='b',  where='post',
# #          label='Hydrogen Storage mass flow out')
# ax2.legend(loc='upper right')
# ax2.set_ylabel('Storage production rate [kg/hr]', color='b')
plt.title('Control Profiles')
plt.grid(True)

# Power figure
fig = plt.figure(figsize=(10, 6))
plt.step(time_vector, power_supply, color='r', where='post', linewidth=2, label='Power Supply')
plt.step(time_vector, power_delivered, color='g', where='post', label='Power delivered')
# just for checking, should be equal to power supply:
# sum = [x + y for x, y in zip(power_delivered, electrolyser_power_input)]
# plt.step(time_vector, sum, label='sum', linestyle='--')
plt.step(time_vector, power_demand, color='b', where='post', linewidth=2, label='Power Demand')
plt.step(time_vector, heatpump_power_input, color='m', where='post', linewidth=2,
         label='Power Heatpump')
# plt.step(time_vector, power_difference, color='m', label='Demand - Delivered')
plt.xlabel('Time (hr)')
plt.ylabel('Power [MW]')
# plt.xticks(time_vector_control)
plt.title('Power Profiles')
plt.legend()
plt.grid(True)

# Heat figure
fig = plt.figure(figsize=(10, 6))
plt.step(time_vector, heat_demand,  color='b', where='post', linewidth=2,
         label='Heat Demand')
plt.step(time_vector, heat_delivered, color='g', where='post',
         label='Heat Delivered')
# plt.step(time_vector, hydrogen_difference, color='m', label='Demand - Delivered')
plt.xlabel('Time (hr)')
plt.ylabel('Heat[MW]')
plt.title('Heat Profiles')
plt.legend()
plt.grid(True)

# Mismatch figure
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.step(time_vector, power_difference, color='r', where='post', label='Power')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Power [MW]', color='r')
ax1.legend(loc='upper left')
# ax1.set_ylim(-1, 1)
ax2 = ax1.twinx()
ax2.step(time_vector, heat_difference, color='b', where='post', label='Heat')
ax2.set_ylabel('Heat [MW]', color='b')
ax2.legend(loc='upper right')
# ax2.set_ylim(-1, 1)
plt.title('(Demand - Delivered)/Demand')
plt.grid(True)

# KPI figure
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.step(time_vector, efficiency_electrolyser, color='r', where='post', label='Efficiency %')
# ax1.set_xlabel('Time (hr)')
# ax1.set_ylabel('Efficiency [%]', color='r')
# ax1.legend(loc='upper left')
# # ax1.set_ylim(-1, 1)
# # ax2=ax1.twinx()
# # ax2.step(time_vector, hydrogen_difference, color='b', where='post', label='electrolyser')
# # ax2.set_ylabel('COP [-]', color='b')
# # ax2.legend(loc='upper right')
# # ax2.set_ylim(-1, 1)
# plt.title('Electrolyser')
# plt.grid(True)

plt.show()

# # Write output to files
# df = pd.DataFrame(control_reshaped, index=['splitter1', 'splitter2', 'hydrogen_storage'])
# # Save the DataFrame to a CSV file
# df.to_csv('control.csv', index=True)
