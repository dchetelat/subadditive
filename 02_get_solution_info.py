import pickle
import numpy as np
from pathlib import Path

# ----------------------------------------------------
# setcover

instance_folder = Path('data/instances/setcover')
print(f"Solving instances in {instance_folder}")

instance_values = {}
instance_paths = list(instance_folder.glob("*.lp"))
for count, instance_path in enumerate(instance_paths):
    if count % 20 == 0:
        print(f"[{count}/{len(instance_paths)}] Obtaining solving info for {instance_path.name}")
    A, b, c, vtypes, objective_offset = load_instance(str(instance_path), 
                                                      add_variable_bounds=False, 
                                                      presolve=True)

    optimal_value, *_ = solve_ilp(A, b, c, vtypes)
    lp_optimal_value, *_ = solve_lp(A, b, c)
    instance_values[instance_path.name] = {"optimal_value": optimal_value.item(),
                                           "lp_optimal_value": lp_optimal_value.item(),
                                           "add_variable_bounds": False,
                                           "presolve": True}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")


# ----------------------------------------------------
# cauctions

instance_folder = Path('data/instances/cauctions')
print(f"Solving instances in {instance_folder}")

instance_values = {}
instance_paths = list(instance_folder.glob("*.lp"))
for count, instance_path in enumerate(instance_paths):
    if count % 20 == 0:
        print(f"[{count}/{len(instance_paths)}] Obtaining solving info for {instance_path.name}")
    A, b, c, vtypes, objective_offset = load_instance(str(instance_path), 
                                                      add_variable_bounds=False, 
                                                      presolve=True)

    optimal_value, *_ = solve_ilp(A, b, c, vtypes)
    lp_optimal_value, *_ = solve_lp(A, b, c)
    instance_values[instance_path.name] = {"optimal_value": optimal_value.item(),
                                           "lp_optimal_value": lp_optimal_value.item(),
                                           "add_variable_bounds": False,
                                           "presolve": True}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")


# ----------------------------------------------------
# indset

instance_folder = Path('data/instances/indset')
print(f"Solving instances in {instance_folder}")

instance_values = {}
instance_paths = list(instance_folder.glob("*.lp"))
for count, instance_path in enumerate(instance_paths):
    if count % 20 == 0:
        print(f"[{count}/{len(instance_paths)}] Obtaining solving info for {instance_path.name}")
    A, b, c, vtypes, objective_offset = load_instance(str(instance_path), 
                                                      add_variable_bounds=False, 
                                                      presolve=True)

    optimal_value, *_ = solve_ilp(A, b, c, vtypes)
    lp_optimal_value, *_ = solve_lp(A, b, c)
    instance_values[instance_path.name] = {"optimal_value": optimal_value.item(),
                                           "lp_optimal_value": lp_optimal_value.item(),
                                           "add_variable_bounds": False,
                                           "presolve": True}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")


# ----------------------------------------------------
# facilities

instance_folder = Path('data/instances/facilities')
print(f"Solving instances in {instance_folder}")

instance_values = {}
instance_paths = list(instance_folder.glob("*.lp"))
for count, instance_path in enumerate(instance_paths):
    if count % 20 == 0:
        print(f"[{count}/{len(instance_paths)}] Obtaining solving info for {instance_path.name}")
    A, b, c, vtypes, objective_offset = load_instance(str(instance_path), 
                                                      add_variable_bounds=False, 
                                                      presolve=True)

    optimal_value, *_ = solve_ilp(A, b, c, vtypes)
    lp_optimal_value, *_ = solve_lp(A, b, c)
    instance_values[instance_path.name] = {"optimal_value": optimal_value.item(),
                                           "lp_optimal_value": lp_optimal_value.item(),
                                           "add_variable_bounds": False,
                                           "presolve": True}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")


# ----------------------------------------------------
# 2-matching

instance_folder = Path('data/instances/2-matching')
print(f"Solving instances in {instance_folder}")

instance_values = {}
instance_paths = list(instance_folder.glob("*.lp"))
for count, instance_path in enumerate(instance_paths):
    if count % 20 == 0:
        print(f"[{count}/{len(instance_paths)}] Obtaining solving info for {instance_path.name}")
    A, b, c, vtypes, objective_offset = load_instance(str(instance_path), 
                                                      add_variable_bounds=True, 
                                                      presolve=True)
    optimal_value, *_ = solve_ilp(A, b, c, vtypes)
    lp_optimal_value, *_ = solve_lp(A, b, c)
    instance_values[instance_path.name] = {"optimal_value": optimal_value.item(),
                                           "lp_optimal_value": lp_optimal_value.item(),
                                           "add_variable_bounds": True,
                                           "presolve": True}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")


# ----------------------------------------------------
# small-miplib3

instance_folder = Path('data/instances/small-miplib3')
print(f"Solving instances in {instance_folder}")

instance_values = {}
with open(instance_folder/'solution_info.txt', 'r') as file:
    for line in file:
        instance_name = line[:10].strip()
        optimal_value = float(line[50:67].strip())
        lp_optimal_value = float(line[67:-1].strip())
        
        instance_name = instance_name+".mps.gz"
        if (instance_folder/instance_name).exists():
            print(f"Obtaining solving info for {instance_name}")
            instance_values[instance_name] = {"optimal_value": optimal_value,
                                              "lp_optimal_value": lp_optimal_value,
                                              "add_variable_bounds": True,
                                              "presolve": False}

with (instance_folder/"solution_info.pkl").open('wb') as file:
    pickle.dump(instance_values, file)
print("Done\n")
