from BaseCuckooSearch import nests, best_cost
from BaseCuckooSearch import top_nests
from ObjectiveFunctions import cost_calculator
from MainVariables import vessel_length, berth_length
import numpy as np
from collections import defaultdict
from MainVariables import time_window_per_berth
import heapq
import python_weather
import asyncio
import random
import os


def print_top_nests(top_nests, best_cost):
    nest_costs = []
    print("Top Nests (in ascending order of cost):")
    for i, nest in enumerate(top_nests):
        cost = cost_calculator.calculate_cost_component2(nest)
        nest_costs.append(cost)
        print(f"Nest {i+1}: {nest}")
        print(f"Cost: {cost}")

    print("Best Cost:")
    print(best_cost)

    return nest_costs


def assign_vessel_ids(nests, best_cost_list, vessel_lengths):
    best_cost_list = print_top_nests(top_nests, best_cost)
    closest_cost_idx = min(
        range(len(best_cost_list)),
        key=lambda i: abs(best_cost_list[i] - sum(top_nests[i])),
    )
    vessel_id = closest_cost_idx + 1
    sorted_nests = sorted(enumerate(best_cost_list), key=lambda x: x[1])
    vessel_ids = [
        sorted_nests.index((i, cost)) + 1 for i, cost in enumerate(best_cost_list)
    ]

    # ad_data = [[sorted_nests[4], sorted_nests[5], sorted_nests[6]] for i, sorted_nests in enumerate(nests)]
    ad_data = [[nest for nest in sorted_nests] for i, sorted_nests in enumerate(nests)]

    result = []
    vessel_costs = {}
    result_dict = {}
    for i, nest in enumerate(nests):
        result.append(
            (
                f"Nest {i+1}: {nest} - Vessel ID: {vessel_ids[i]} - arrival and departure time: {ad_data[i]} - vessel length: {vessel_lengths[i]} - cost: {best_cost_list[i]}"
            )
        )
        result_dict[vessel_ids[i]] = ad_data[i]
        # result_dict = {vessel_ids[i] : ad_data[i]}
        if vessel_ids[i] not in vessel_costs:
            vessel_costs[vessel_ids[i]] = [best_cost_list[i]]
        else:
            vessel_costs[vessel_ids[i]].append(best_cost_list[i])

    return vessel_id, result, vessel_costs, ad_data, result_dict


vessel_id, result, vessel_costs, ad_data, result_dict = assign_vessel_ids(
    top_nests, best_cost, vessel_length
)

print("Vessel ID:", vessel_id)
print("Result:")
for res in result:
    print(res)

print("Vessel Costs:")
for vessel_id, costs in vessel_costs.items():
    avg_cost = np.mean(costs)
    print(f"Vessel ID: {vessel_id}, Average Cost: {avg_cost}")

print("ad_data")
print(ad_data)
print(result_dict)

print("result")
print(result)
print("end of result")

# 6) time slot required for each vessel
# 7) create a time slot queue for each berth
def calculate_time_slots(result_dict, time_window_per_berth):
    total_time_slots = {}

    for vessel_id, data in result_dict.items():
        processing_time = data[5] - data[4]
        time_slots = processing_time // time_window_per_berth

        total_time_slots[vessel_id] = time_slots
    return total_time_slots


total_time_slots = calculate_time_slots(result_dict, time_window_per_berth)

print("total_time_slots")
print(total_time_slots)


def extract_arrival_time_and_vessel_id_from_data(result_dict, vessel_costs):
    arrival_times = {key: value[0] for key, value in result_dict.items()}
    vessel_ids = list(result_dict.keys())

    arrival_times_and_vessel_costs = {}
    for vessel_id, arrival_time in zip(vessel_ids, arrival_times.values()):
        cost = vessel_costs.get(
            vessel_id, "Unknown"
        )  # Get the cost for the vessel id or set to "Unknown" if not found
        arrival_times_and_vessel_costs[vessel_id] = {
            "arrival_time": arrival_time,
            "cost": cost,
        }

    return arrival_times_and_vessel_costs


arrival_times_and_vessel_costs = extract_arrival_time_and_vessel_id_from_data(
    result_dict, vessel_costs
)
print("arrival_times_and_vessel_costs")
print(arrival_times_and_vessel_costs)
############# sorting based on 2 factors using bubble sort ##################
#############################################################################
def process_data(data):
    restructured_data = []

    key_actions = {
        "Nest": lambda d, k, v: d.update({"Nest": int(k.split()[1])}),
        "Vessel ID": lambda d, k, v: d.update({"Vessel ID": int(v)}),
        "arrival and departure time": lambda d, k, v: d.update(
            {"Arrival and Departure Time": [int(x) for x in v[1:-1].split(", ")]}
        ),
        "vessel length": lambda d, k, v: d.update({"Vessel Length": int(v[1:-1])}),
        "cost": lambda d, k, v: d.update({"Cost": float(v)}),
    }

    for item in data:
        nest_info = item.split(" - ")
        nest_dict = {}

        for info in nest_info:
            key, value = info.split(": ")
            for keyword in key_actions:
                if key.startswith(keyword):
                    key_actions[keyword](nest_dict, key, value)
                    break
            else:
                values = key.split(": ")[1][1:-1].split(", ")
                nest_dict["Values"] = [int(x) for x in values]

        restructured_data.append(nest_dict)

    return restructured_data


processed_data = process_data(result)

# Accessing data example:
print(processed_data)

def add_time_slots(data, vessel_time_slots):
    """
    Adds time slots to the data based on vessel IDs.

    Args:
        data (list[dict]): List of dictionaries containing vessel information.
        vessel_time_slots (dict): Dictionary mapping vessel IDs to time slots.

    Returns:
        list[dict]: Updated data with time slots added.
    """
    updated_data = []
    for entry in data:
        vessel_id = entry.get("Vessel ID")
        time_slot = vessel_time_slots.get(vessel_id)
        if time_slot is not None:
            entry["Vessel Time Slot"] = time_slot
        updated_data.append(entry)
    return updated_data


updated_data = add_time_slots(processed_data, total_time_slots)
for entry in updated_data:
    print(entry)

for entry in updated_data:
    vessel_id = entry["Vessel ID"]
    vessel_length = entry["Vessel Length"]
    print(f"Vessel ID: {vessel_id}, Vessel Length: {vessel_length}")


def bubble_sort(data):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j]["Arrival and Departure Time"][4] > data[j + 1][
                "Arrival and Departure Time"
            ][4] or (
                data[j]["Arrival and Departure Time"][4]
                == data[j + 1]["Arrival and Departure Time"][4]
                and data[j]["Cost"] > data[j + 1]["Cost"]
            ):
                data[j], data[j + 1] = data[j + 1], data[j]


# Sort the list
bubble_sort(updated_data)


print("updated_data")
print(updated_data)
# Output the sorted list
for entry in updated_data:
    print("sorted")
    print(entry)


class PreferredVessel:
    def __init__(self, berth_lengths, vessel_lengths):
        self.berth_lengths = berth_lengths
        self.vessel_lengths = vessel_lengths

    def assign_berths(self, updated_data):
        preferred_berths = {}

        for entry in updated_data:
            vessel_id = entry["Vessel ID"]
            vessel_length = entry["Vessel Length"]

            # Filter out berth lengths that are greater than or equal to the vessel length
            valid_berth_lengths = [
                berth_length
                for berth_length in self.berth_lengths
                if berth_length[0] >= vessel_length
            ]

            if valid_berth_lengths:  # Check if there are valid berth lengths
                preferred_berth_length = min(valid_berth_lengths)
                preferred_berths[vessel_id] = {
                    "Preferred Berth Length": preferred_berth_length,
                    "Vessel Length": vessel_length,
                }
            else:
                print(f"No valid berth length found for Vessel ID {vessel_id}")

        return preferred_berths

preferred_vessel = PreferredVessel(berth_length, vessel_length)
result = preferred_vessel.assign_berths(updated_data)

print("Preferred Vessel Lengths:")
for vessel_id, data in result.items():
    print(
        f"Vessel ID {vessel_id}: Preferred Berth Length {data['Preferred Berth Length']}, Vessel Length {data['Vessel Length']}"
    )

########################## Berth allocation #####################
########################## Find Z parameter here ###################################
import pandas as pd

def create_berth_info(berth_lengths):
    # Initialize an empty dictionary to store berth information
    berth_info = {}

    # Assume departure time for all berths (you can adjust this as needed)
    departure_time = 0

    # Iterate through each berth and add relevant information
    for idx, length_list in enumerate(berth_lengths):
        # Extract the actual berth length from the sublist
        length = length_list[0]

        # Example: Berth status (you can customize this based on your logic)
        status = "Unknown"

        # Create a dictionary for each berth
        berth_dict = {
            "berth_index": idx + 1,
            "berth_length": length,
            "status": status,
            "departure_time": departure_time,
        }

        # Add the berth information to the main dictionary
        berth_info[f"Berth {idx + 1}"] = berth_dict

    return berth_info


# Example usage:

berth_information = create_berth_info(berth_length)

# Print the resulting berth information
for berth, info in berth_information.items():
    print(f"{berth}: {info}")


# List of vessels with priority and arrival/departure times
vessels = updated_data

# def allocate_vessels_to_berths(vessels):
# berth_information  # Your berth information dictionary
# preferred_vessel  # Your preferred vessel dictionary
result = preferred_vessel.assign_berths(vessels)


import pandas as pd

allocated_vessels = set()  # Set to keep track of allocated vessels

for vessel in vessels:
    if vessel["Vessel ID"] in allocated_vessels:
        continue  # Skip this vessel if it has already been allocated a berth
    allocated_berth = None
    for berth_name, berth_info in berth_information.items():
        if (
            berth_info["departure_time"] is None
            or vessel["Arrival and Departure Time"][5] >= berth_info["departure_time"]
        ):
            if vessel["Vessel Length"] <= berth_info["berth_length"]:
                allocated_berth = berth_name
                print("Allocated berth:")
                print(allocated_berth)
                print("Preferred Berth Index:")
                preferred_berth_index = result[vessel["Vessel ID"]]
                print(preferred_berth_index)
                berth_info["departure_time"] = vessel["Arrival and Departure Time"][5]
                print(f"Vessel {vessel['Vessel ID']} allocated to {allocated_berth}")
                allocated_vessels.add(vessel["Vessel ID"])
                break  # Exit the loop once a berth is allocated for the vessel

# Define the time slots available for allocation
time_slots = 24  # Assuming there are 24 time slots in a day

# Initialize a list to store the data for the CSV
csv_data = [["Time Slot"] + list(berth_information.keys())]

# Fill in the time slots and allocated vessels in the CSV data
for i in range(1, time_slots + 1):
    row_data = [str(i)] + [""] * len(berth_information)
    for vessel in vessels:
        if vessel["Vessel ID"] in allocated_vessels:
            for berth_name, berth_info in berth_information.items():
                if berth_info["departure_time"] is None or vessel["Arrival and Departure Time"][5] >= berth_info["departure_time"]:
                    if vessel["Vessel Length"] <= berth_info["berth_length"]:
                        allocated_berth = berth_name
                        if i <= 12:  # Assuming each vessel requires 12 time slots
                            row_data[list(berth_information.keys()).index(allocated_berth) + 1] = f"{vessel['Vessel ID']} ({berth_info['berth_length']} meters)"
                            break

    csv_data.append(row_data)

# Create a DataFrame from the CSV data
df = pd.DataFrame(csv_data)

# Write the DataFrame to a CSV file
df.to_csv("berth_allocation.csv", index=False)

print("CSV file 'berth_allocation.csv' created successfully!")










""" 
                
                if allocated_berth != f"Berth {preferred_berth_index + 1}":
                    print(f"Warning: Allocated berth ({allocated_berth}) does not match preferred berth (Berth {preferred_berth_index + 1})")
                    
                    
                    preferred_berth_length = result[vessel['Vessel ID']] ####### this is wrong 
                    print("preferred_berth_length")
                    print(preferred_berth_length)
                    allocated_berth_length = vessel['Vessel Length']
                    print("allocated_berth_length")
                    print(allocated_berth_length)


                break  # Exit the inner loop after finding an allocation
    else:
        print(f"No suitable berth found for Vessel {vessel['Vessel ID']}")

"""


""" 
        if allocated_berth:
            print(f"Vessel {vessel['Vessel ID']} allocated to {allocated_berth}")
            preferred_berth_index = result[vessel['Vessel ID']]
            if allocated_berth != f"Berth{preferred_berth_index + 1}":
            
                preferred_berth_length = result[vessel['Vessel ID']]
                allocated_berth_length = vessel['Vessel Length']
                Z = abs(preferred_length - allocated_length)
                C1 = vessel['Arrival and Departure Time'][0]
                params = C1, Z

                cost_2 = cost_calculator.calculate_cost_component1(params)
                total_cost = cost_calculator.default_objective_function(params)
                print(f"Total cost for Vessel {vessel['Vessel ID']}: {total_cost}")

        else:
            print(f"Vessel {vessel['Vessel ID']} could not be allocated to any berth")

"""

# allocate_vessels_to_berths(vessels)
