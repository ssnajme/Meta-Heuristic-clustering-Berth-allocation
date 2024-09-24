import queue
import random
from MainNestGeneration import c_min, c_max
import numpy as np
import csv
from queue import PriorityQueue
from ObjectiveFunctions import cost_calculator
from utils import (
    vessel_length,
    berth_length,
    weather_conditions,
    tide_levels,
    identify_critical_levels,
    water_depths_instance,
)
from utils import total_time_slots, time_window_length, num_vessels, num_berths
# --- Rename these -----
from BenhancedClusteringCuckooSearch import top_nests_3, convergence_costs_3
#from BaseCuckooSearch import nests, best_cost, top_nests
#from CEDA import best_cost_2, top_nests_2
#from CPSO import top_nests_4, top_costs_4
#from BaseClusteringCuckooSearch import  top_nests_5, best_cost_5




class BerthManagement:
    def __init__(self, cost_calculator):
        """
        Initializes the NestManager with a cost calculator.

        Parameters:
        - cost_calculator: An object with a method `calculate_cost_component2` for cost calculations.
        """
        self.cost_calculator = cost_calculator

    def top_nests_costs(self, top_nests, best_cost):
        """
        Calculates and prints costs for the top nests.

        Parameters:
        - top_nests (list): A list of the top nests.
        - best_cost (float): The best cost to compare against.

        Returns:
        - list: A list of costs associated with the top nests.
        """
        nest_costs = []
        print("Top Nests (in ascending order of cost):")

        for i, nest in enumerate(top_nests):
            cost = self.cost_calculator.calculate_cost_component2(nest)
            nest_costs.append(cost)
            print(f"Nest {i+1}: {nest}")
            print(f"Cost: {cost}")

        print("Best Cost:")
        print(best_cost)
        return nest_costs

    def assign_vessel_ids(self, nests, best_cost_list, vessel_lengths):
        #best_cost_list = self.top_nests_costs(top_nests_3, convergence_costs_3)
        best_cost_list = self.top_nests_costs(top_nests_3, convergence_costs_3)
        closest_cost_idx = min(
            range(len(best_cost_list)),
            key=lambda i: abs(best_cost_list[i] - sum(top_nests_3[i])),
        )
        vessel_id = closest_cost_idx + 1
        sorted_nests = sorted(enumerate(best_cost_list), key=lambda x: x[1])
        vessel_ids = [
            sorted_nests.index((i, cost)) + 1 for i, cost in enumerate(best_cost_list)
        ]

        # ad_data = [[sorted_nests[4], sorted_nests[5], sorted_nests[6]] for i, sorted_nests in enumerate(nests)]
        ad_data = [
            [nest for nest in sorted_nests] for i, sorted_nests in enumerate(nests)
        ]

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

    def calculate_time_slots(self, result_dict, time_window_per_berth):
        """
        Calculates the total time slots for each vessel based on processing times.

        Parameters:
        - result_dict (dict): A dictionary containing vessel IDs and their associated data.
        - time_window_per_berth (int): The time window allocated per berth.

        Returns:
        - dict: A dictionary mapping vessel IDs to total time slots.
        """
        total_time_slots = {}

        for vessel_id, data in result_dict.items():
            processing_time = (
                data[5] - data[4]
            )  # Assuming data[4] is arrival time and data[5] is departure time
            time_slots = processing_time // time_window_per_berth

            total_time_slots[vessel_id] = time_slots

        return total_time_slots

    def extract_arrival_time_and_vessel_id_from_data(self, result_dict, vessel_costs):
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

    # --- RENAME these data ---------------
    def process_data(self, data):
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

    def add_time_slots(self, data, vessel_time_slots):
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

    def sort_nests(self, data):
        n = len(data)

        # Sort the data
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

        # Add the new key-value pair after sorting
        # --- RENAME THESE VARIABLES ------
        new_key = "Allocated"
        new_value = False
        for entry in data:
            entry[new_key] = new_value
        return data

    def Final_order(self):# I changed this line of code 
        vessel_id, result, vessel_costs, ad_data, result_dict = self.assign_vessel_ids(
            top_nests_3, convergence_costs_3, vessel_length
        )
        time_slots = self.calculate_time_slots(result_dict, time_window_length)
        arrival_time = self.extract_arrival_time_and_vessel_id_from_data(
            result_dict, vessel_costs
        )
        data_reshape = self.process_data(result)
        data_with_time_slots = self.add_time_slots(data_reshape, time_slots)
        order_nest = self.sort_nests(data_with_time_slots)
        return order_nest, result, time_slots


BerthManagement_instance = BerthManagement(cost_calculator)
ordered_nest, result, time_slots = BerthManagement_instance.Final_order()
# --- do not need this now -- print("ordered_nest")
# --- do not need this now -- print(ordered_nest)


# ------------------------------------------------------------
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
            #else:
                
        return preferred_berths


PreferredVessel_instance = PreferredVessel(berth_length, vessel_length)
assign_berths_instance = PreferredVessel_instance.assign_berths(ordered_nest)
# --- do not need this now -- print("PreferredVessel_instance")
# --- do not need this now -- print(assign_berths_instance)


# -------------------------------------------------------------------------------------
class vesselInformation:
    def preferred_berth_length_and_vessel_draft(
        self,
        updated_data,
        result,
        new_key="Preferred Berth Length",
        draft_key="Draft at Full Load",
        min_draft=5.0,
        max_draft=15.0,
    ):
        for entry in updated_data:
            vessel_id = entry["Vessel ID"]
            if vessel_id in result:
                # Get the new values from the result dictionary
                new_value = result[vessel_id].get(new_key, "Default Value")
                draft_value = round(random.uniform(min_draft, max_draft), 1)
                # Update the entry with new values
                entry[new_key] = new_value
                entry[draft_key] = draft_value
        return updated_data


# ----- we have a problem here --------
vesselInformation_instance = vesselInformation()
updated_data_1 = vesselInformation_instance.preferred_berth_length_and_vessel_draft(
    ordered_nest, assign_berths_instance
)
print("updated_data_1")
print(updated_data_1)


# --------------------------------------------------------------------------------------
class BerthInformation:
    def __init__(self, berth_lengths, total_time_slots):
        """
        Initializes the BerthManager with berth lengths and total time slots.

        :param berth_lengths: A list of lists containing berth lengths.
        :param total_time_slots: The total number of time slots for each berth.
        """
        self.berth_lengths = berth_lengths
        self.water_depths_instance = water_depths_instance
        self.total_time_slots = total_time_slots
        self.berth_info = self.create_berth_info()

    def create_berth_info(self):
        """
        Creates a dictionary containing information about each berth.

        :return: A dictionary with berth information.
        """
        berth_info = {}
        # Iterate through each berth and add relevant information
        for idx, length_list in enumerate(self.berth_lengths):
            length = length_list[0]  # Extract the actual berth length
            water_depth = self.water_depths_instance[idx]
            status = "unoccupied"
            berth_dict = {
                "berth_index": idx + 1,
                "berth_length": length,
                "water_depth": water_depth,
                "status": status,
                "total time slots": self.total_time_slots,
                "statuses_per_time_slot": {
                    time_slot: "unoccupied"
                    for time_slot in range(self.total_time_slots)
                },
            }
            # Add the berth information to the main dictionary
            berth_info[f"Berth {idx + 1}"] = berth_dict

        return berth_info

    def get_berth_info(self):
        """
        Returns the berth information.

        :return: The berth information dictionary.
        """
        return self.berth_info


BerthInformation_instance = BerthInformation(berth_length, total_time_slots)
berth_info_instance = BerthInformation_instance.create_berth_info()

print("berth_info_instance")
print(berth_info_instance)
# --------------------------------------------------------
# --- including the uncertain variables ------------------

# uncertainFactors_instance = uncertainFactors()
# can_berth_accommodate_instance  = uncertainFactors_instance.can_berth_accommodate(updated_data_1, berth_info_instance  ,tide_levels, weather_conditions)


# ------- this is the start of the berth allocation ---------
# -----------------------------------------------------------
priority_queues = {
    berth_index: PriorityQueue() for berth_index in range(1, num_berths + 1)
}

class BerthAllocation:
    def allocate_vessels(
        self, vessels, berth_information, c_min, c_max, cost_calculator, priority_queues
    ):
        water_depths = [berth["water_depth"] for berth in berth_information.values()]
        actual_depths = [
            w_depth + tide for w_depth, tide in zip(water_depths, tide_levels)
        ]
        weather_risks = {
            "high_winds": 0.3,
            "poor_visibility": 0.2,
            "heavy_rain": 0.1,
            "storm": 0.5,
            "fog": 0.2,
            "snow": 0.3,
            "hail": 0.4,
            "extreme_temperatures": 0.1,
            "lightning": 0.6,
            "thunderstorms": 0.5,
        }
        risk_level = sum(
            weather_risks.get(condition, 0)
            for condition in weather_conditions
            if weather_conditions[condition]
        )
        decision_threshold = 0.5

        for vessel in vessels:
            if not vessel["Allocated"]:
                print(f"Vessel ID: {vessel['Vessel ID']}")
                print(
                    f"Draft at Full Load: {vessel.get('Draft at Full Load', 'Not available')}"
                )

                # ------ this part includes uncertain factors
                # -------------------------------------------
                #required_draft = vessel[
                #    "Draft at Full Load"
                #]  # this access is wrong -----
                depth_with_risk = actual_depths[0] * (1 - risk_level)
                print("depth_with_risk")
                print(depth_with_risk)

                for berth_name, berth_info in berth_information.items():
                    if (
                        berth_info["status"] == "unoccupied"
                        and berth_info["berth_length"] >= vessel["Vessel Length"]
                        #and depth_with_risk < max(required_draft, decision_threshold)
                    ):
                        print(
                            f"Vessel {vessel['Vessel ID']} with {vessel['Arrival and Departure Time'][4]} and initial cost {vessel['Cost']} can be allocated to {berth_name}."
                        )
                        vessel["Allocated"] = True
                        berth_info["status"] = "occupied"
                        filtered_records_1 = {}
                        # Check if the preferred berth length is different from the allocated berth length
                        if (
                            vessel["Preferred Berth Length"][0]
                            != berth_info["berth_length"]
                        ):
                            z = (
                                berth_info["berth_length"]
                                - vessel["Preferred Berth Length"][0]
                            )

                            filtered_records_1[vessel["Vessel ID"]] = {
                                "berth_length": berth_info["berth_length"],
                                "Preferred Berth Length": vessel[
                                    "Preferred Berth Length"
                                ],
                                "Z": z,
                                "Arrival and Departure Time": vessel[
                                    "Arrival and Departure Time"
                                ],
                            }
                            z_value = filtered_records_1[vessel["Vessel ID"]]["Z"]
                            C_1 = random.randint(c_min, c_max)
                            arrival_departure_time = filtered_records_1[
                                vessel["Vessel ID"]
                            ]["Arrival and Departure Time"]
                            params = arrival_departure_time.insert(3, z_value)
                            params = arrival_departure_time.insert(0, C_1)
                            final_cost = cost_calculator.default_objective_function(
                                arrival_departure_time
                            )
                            # Update filtered_records_1 with the final cost
                            filtered_records_1[vessel["Vessel ID"]][
                                "Final Cost"
                            ] = final_cost
                            # here update the information of the vessels
                            [
                                vessel.__setitem__(
                                    "Final Cost",
                                    filtered_records_1[vessel["Vessel ID"]][
                                        "Final Cost"
                                    ],
                                )
                                for vessel in vessels
                                if vessel["Vessel ID"] in filtered_records_1
                                and "Final Cost"
                                in filtered_records_1[vessel["Vessel ID"]]
                            ]
                        new_time_slots = {
                            slot: (
                                f"Occupied with {vessel['Vessel ID']} "
                                f"with Arrival time of {vessel['Arrival and Departure Time'][4]} "
                                f"and initial cost {vessel['Cost']},"
                                f"and final cost {vessel.get('Final Cost', 'N/A')}"
                            )
                            for slot in range(vessel["Vessel Time Slot"])
                        }
                        berth_info["statuses_per_time_slot"].update(new_time_slots)

                        print("statuses_per_time_slot for Berth:")
                        print(berth_info["statuses_per_time_slot"])
                        allocated_time_slots = [
                            slot
                            for slot, status in berth_info[
                                "statuses_per_time_slot"
                            ].items()
                            if status.startswith("Occupied")
                        ]

                        print("Allocated time slots for Berth:")
                        print(allocated_time_slots)

                        print("Empty time slots for Berth:")
                        berth_index = berth_info["berth_index"]
                        unoccupied_slots = sum(
                            1
                            for slot_status in berth_info[
                                "statuses_per_time_slot"
                            ].values()
                            if slot_status == "unoccupied"
                        )
                        priority_queues[berth_index].put(
                            (-unoccupied_slots, berth_index)
                        )

                        vessel["Allocated Berth"] = berth_name

                        break
        # ---- Re-allocation starts here----------
        # ----------------------------------------
        for vessel in vessels:
            if not vessel["Allocated"]:
                print(f"Vessel ID: {vessel['Vessel ID']}")
                allocated = False  # Flag to track allocation
                print(
                    f"Draft at Full Load: {vessel.get('Draft at Full Load', 'Not available')}"
                )

                # ------ this part includes uncertain factors
                # -------------------------------------------
                required_draft = vessel[
                    "Draft at Full Load"
                ]  # this access is wrong -----
                depth_with_risk = actual_depths[0] * (1 - risk_level)

                for berth_name, berth_info in berth_information.items():
                    for time_slot, slot_status in berth_info[
                        "statuses_per_time_slot"
                    ].items():
                        if (
                            slot_status == "unoccupied"
                            and berth_info["berth_length"] >= vessel["Vessel Length"]
                            and depth_with_risk
                            < max(required_draft, decision_threshold)
                        ):

                            if (
                                vessel["Vessel Time Slot"]
                                <= len(berth_info["statuses_per_time_slot"])
                                - time_slot 
                            ):
                                print(
                                    f"Vessel {vessel['Vessel ID']} with Arrival time {vessel['Arrival and Departure Time'][4]} and initial cost {vessel['Cost']} can be allocated to {berth_name}"
                                )
                                vessel["Allocated"] = True
                                allocated = True
                                filtered_records_2 = {}
                                if (
                                    vessel["Preferred Berth Length"][0]
                                    != berth_info["berth_length"]
                                ):
                                    z = (
                                        berth_info["berth_length"]
                                        - vessel["Preferred Berth Length"][0]
                                    )

                                    filtered_records_2[vessel["Vessel ID"]] = {
                                        "berth_length": berth_info["berth_length"],
                                        "Preferred Berth Length": vessel[
                                            "Preferred Berth Length"
                                        ],
                                        "Z": z,
                                        "Arrival and Departure Time": vessel[
                                            "Arrival and Departure Time"
                                        ],
                                    }
                                    z_value = filtered_records_2[vessel["Vessel ID"]][
                                        "Z"
                                    ]
                                    C_2 = random.randint(c_min, c_max)
                                    arrival_departure_time = filtered_records_2[
                                        vessel["Vessel ID"]
                                    ]["Arrival and Departure Time"]
                                    params = arrival_departure_time.insert(3, z_value)
                                    params = arrival_departure_time.insert(0, C_2)
                                    final_cost = (
                                        cost_calculator.default_objective_function(
                                            arrival_departure_time
                                        )
                                    )
                                    # Update filtered_records_1 with the final cost
                                    filtered_records_2[vessel["Vessel ID"]][
                                        "Final Cost"
                                    ] = final_cost
                                    print(filtered_records_2)

                                    [
                                        vessel.__setitem__(
                                            "Final Cost",
                                            filtered_records_2[vessel["Vessel ID"]][
                                                "Final Cost"
                                            ],
                                        )
                                        for vessel in vessels
                                        if vessel["Vessel ID"] in filtered_records_2
                                        and "Final Cost"
                                        in filtered_records_2[vessel["Vessel ID"]]
                                    ]
                                for slot in range(vessel["Vessel Time Slot"]):
                                    if (
                                        berth_info["statuses_per_time_slot"].get(
                                            time_slot + slot
                                        )
                                        == "unoccupied"
                                    ):
                                        berth_info["statuses_per_time_slot"][
                                            time_slot + slot
                                        ] = f"Occupied with {vessel['Vessel ID']} with Vessel Arrival Time {vessel['Arrival and Departure Time'][4]} and initial cost {vessel['Cost']} and final cost of {vessel.get('Final Cost', 'N/A')}"
                                print("statuses_per_time_slot for Berth:")
                                print(berth_info["statuses_per_time_slot"])
                                break  # Allocate only once per vessel
                    if allocated:
                        break  # No need to check other berths if already allocated
        return berth_information

BerthAllocation_instance = BerthAllocation()
allocate_vessels_instance = BerthAllocation_instance.allocate_vessels(
    updated_data_1, berth_info_instance, c_min, c_max, cost_calculator, priority_queues
)
print("allocate_vessels_instance")
print(allocate_vessels_instance)

# ---- CSV file ---- reverse everything here -------
class finalCSV:
    def generate_berth_status_csv(
        self, num_berths, total_time_slots, berth_information, filename="berth_data.csv"
    ):
        """
        Generates a CSV file containing the status of each berth over specified time slots.

        Parameters:
        - num_berths (int): The total number of berths.
        - total_time_slots (int): The total number of time slots.
        - berth_information (dict): A dictionary containing berth names and their status information.
        - filename (str): The name of the CSV file to write to.
        """
        # Create a list of time slots
        time_slots = ["Time Slot " + str(i) for i in range(1, total_time_slots + 1)]

        # Initialize an empty dictionary to store berth names and slot statuses
        berth_status_dict = {}

        # Iterate through each berth
        for berth_name, berth_info in berth_information.items():
            # Retrieve the slot status for this berth
            slot_status = berth_info["statuses_per_time_slot"]

            # Store the pair in the dictionary
            berth_status_dict[berth_name] = slot_status

        # Open the CSV file for writing
        with open(filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the status data in reverse order
            for i in range(total_time_slots - 1, -1, -1):
                row_data = [time_slots[i]] + [
                    berth_info["statuses_per_time_slot"][i]
                    for berth_info in berth_information.values()
                ]
                csv_writer.writerow(row_data)

            # Write the header row
            csv_writer.writerow([""] + [f"Berth {i}" for i in range(1, num_berths + 1)])

    # generate_berth_status_csv(num_berths, total_time_slots, berth_information)


csv_instance = finalCSV()
csv_final = csv_instance.generate_berth_status_csv(num_berths,total_time_slots ,allocate_vessels_instance)

#---------------------Thesis Table ---------
# 1) berth occupancy rate 

""" 

class finalCSV:  
    def generate_berth_status_csv(  
        self, num_berths, total_time_slots, berth_information, filename="berth_data.csv"  
    ):  
        # Create a list of time slots  
        time_slots = ["Time Slot " + str(i) for i in range(1, total_time_slots + 1)]  

        # Initialize an empty dictionary to store berth names and slot statuses  
        berth_status_dict = {}  

        # Iterate through each berth  
        for berth_name, berth_info in berth_information.items():  
            # Retrieve the slot status for this berth  
            slot_status = berth_info["statuses_per_time_slot"]  

            # Store the pair in the dictionary  
            berth_status_dict[berth_name] = slot_status  

        # Open the CSV file for writing  
        with open(filename, "w", newline="") as csvfile:  
            csv_writer = csv.writer(csvfile)  

            # Write the header row  
            csv_writer.writerow([""] + [f"Berth {i}" for i in range(1, num_berths + 1)])  

            # Write the status data in reverse order  
            for i in range(total_time_slots - 1, -1, -1):  
                row_data = [time_slots[i]] + [  
                    berth_info["statuses_per_time_slot"][i]  
                    for berth_info in berth_information.values()  
                ]  
                csv_writer.writerow(row_data)  

    def calculate_berth_usage_productivity(self, berth_information, total_time_slots):  

        productivity = {}  

        # Iterate through each berth  
        for berth_name, berth_info in berth_information.items():  
            # Retrieve the slot status for this berth  
            slot_status = berth_info["statuses_per_time_slot"]  

            # Calculate the number of occupied slots  
            occupied_slots = sum(1 for status in slot_status if status == "Occupied")  

            # Calculate productivity as a percentage  
            productivity_percentage = (occupied_slots / total_time_slots) * 100 if total_time_slots > 0 else 0  

            # Store productivity in the dictionary  
            productivity[berth_name] = productivity_percentage  

        return productivity  

# Create an instance of finalCSV  
csv_instance = finalCSV()  

# Generate the CSV (optional, if you want to create the file)  
csv_instance.generate_berth_status_csv(num_berths, total_time_slots, allocate_vessels_instance)  

# Calculate berth usage productivity  
productivity = csv_instance.calculate_berth_usage_productivity(allocate_vessels_instance, total_time_slots)  

# Print the productivity results  
for berth, prod in productivity.items():  
    print(f"{berth}: {prod:.2f}% productivity")
""" 
 