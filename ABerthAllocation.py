from BaseCuckooSearch import nests, best_cost
from BaseCuckooSearch import top_nests
from ObjectiveFunctions import cost_calculator
from MainVariables import time_window_per_berth
from MainVariables import vessel_length, berth_length
from collections import defaultdict
import numpy as np
import pandas as pd
import heapq 
import python_weather
import asyncio
import random
import os


class BerthAllocationStrategy:
    def __init__(self, ship_id, arrival_time, service_time, priority):
        self.ship_id = ship_id
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.priority = priority

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

    def assign_vessel_ids(self, nests, best_cost_list):
        best_cost_list = self.print_top_nests(top_nests, best_cost)
        closest_cost_idx = min(range(len(best_cost_list)), key=lambda i: abs(best_cost_list[i] - sum(top_nests[i])))
        vessel_id = closest_cost_idx + 1
        sorted_nests = sorted(enumerate(best_cost_list), key=lambda x: x[1])
        vessel_ids = [sorted_nests.index((i, cost)) + 1 for i, cost in enumerate(best_cost_list)]
        
        ad_data = [[sorted_nests[4], sorted_nests[5], sorted_nests[6]] for i, sorted_nests in enumerate(nests)]
        
        result = []
        vessel_costs = {}
        result_dict = {}
        for i, nest in enumerate(nests):
            result.append((f"Nest {i+1}: {nest} - Vessel ID: {vessel_ids[i]} - arrival and departure time: {ad_data[i]}"))
            result_dict[vessel_ids[i]] = ad_data[i]
            #result_dict = {vessel_ids[i] : ad_data[i]}
            if vessel_ids[i] not in vessel_costs:
                vessel_costs[vessel_ids[i]] = [best_cost_list[i]]
            else:
                vessel_costs[vessel_ids[i]].append(best_cost_list[i])

        return vessel_id, result, vessel_costs, ad_data, result_dict

    ## 6) time slot required for each vessel 
    ## 7) create a time slot queue for each berth 
    def calculate_time_slots(result_dict, time_window_per_berth):
            total_time_slots = {}
            time_slot_queues = defaultdict(list)
            
            for vessel_id, data in result_dict.items():
                processing_time = data[1] - data[0]
                time_slots = processing_time // time_window_per_berth
                
                total_time_slots[vessel_id] = time_slots
                
            
            return total_time_slots, time_slot_queues
    
    ##  get cost, get arrival time and vessels lengths in this function based on 
    ## vessel id ###############################################################
    ## This goes to the bubble sort #####
    def extract_arrival_time_and_vessel_id_from_data(result_dict, vessel_costs):
        arrival_times = {key: value[0] for key, value in result_dict.items()}
        vessel_ids = list(result_dict.keys())
        
        arrival_times_and_vessel_costs = {}
        for vessel_id, arrival_time in zip(vessel_ids, arrival_times.values()):
            cost = vessel_costs.get(vessel_id, "Unknown")  # Get the cost for the vessel id or set to "Unknown" if not found
            arrival_times_and_vessel_costs[vessel_id] = {"arrival_time": arrival_time, "cost": cost}

        return arrival_times_and_vessel_costs
    
    ### process the structure of the data #######
    ### Get the vessel length from here #########
    def process_data(data):
        restructured_data = []

        key_actions = {
            "Nest": lambda d, k, v: d.update({"Nest": int(k.split()[1])}),
            "Vessel ID": lambda d, k, v: d.update({"Vessel ID": int(v)}),
            "arrival and departure time": lambda d, k, v: d.update({"Arrival and Departure Time": [int(x) for x in v[1:-1].split(', ')]}),
            "vessel length": lambda d, k, v: d.update({"Vessel Length": int(v[1:-1])})
        }

        for item in data:
            nest_info = item.split(' - ')
            nest_dict = {}

            for info in nest_info:
                key, value = info.split(': ')
                for keyword in key_actions:
                    if key.startswith(keyword):
                        key_actions[keyword](nest_dict, key, value)
                        break
                else:
                    values = key.split(': ')[1][1:-1].split(', ')
                    nest_dict["Values"] = [int(x) for x in values]

            restructured_data.append(nest_dict)

        return restructured_data
    
    ### add the vessel time slots to the vessel information 
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
            vessel_id = entry.get('Vessel ID')
            time_slot = vessel_time_slots.get(vessel_id)
            if time_slot is not None:
                entry['Vessel Time Slot'] = time_slot
            updated_data.append(entry)
        return updated_data

    ### get the preferred berth 
    def assign_berths(self, updated_data):
        preferred_berths = {}

        for entry in updated_data:
            vessel_id = entry['Vessel ID']
            vessel_length = entry['Vessel Length']

            # Calculate differences for the current vessel length
            differences = [
                abs(vessel_length - berth_length[0]) for berth_length in self.berth_lengths
            ]

            # Assign preferred berth index based on minimum difference
            preferred_berth_index = differences.index(min(differences))

            # Store the preferred berth index for the vessel ID
            preferred_berths[vessel_id] = preferred_berth_index

        return preferred_berths

    ### include uncertain variables   
    async def get_weather():
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            weather = await client.get('Shanghai')  # Change location to Shanghai
            print("Current temperature:", weather.temperature)
            for daily in weather.daily_forecasts:
                print("Daily forecast:", daily)
                for hourly in daily.hourly_forecasts:
                    print(f"Hourly forecast: {hourly!r}")
            
            return weather

    def generate_sea_level_and_tide(weather):
        # Add your logic to generate sea level and tide statuses based on weather conditions
        sea_level_conditions = ["Low", "Normal", "High"]
        tide_status = "Low"
        
        # Generate random sea level and sea depth conditions
        sea_level = random.choice(sea_level_conditions)
        sea_depth_conditions = ["Shallow", "Moderate", "Deep"]
        sea_depth = random.choice(sea_depth_conditions)
        
        return sea_level, tide_status, sea_depth
    

    # Create a allocation queue for each berth based on
    # 1) the arrival time of the vessels 
    # 2) cost priority 

    def bubble_sort_dict(data):
        keys = list(data.keys())
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[keys[j]]['arrival_time'] > data[keys[j + 1]]['arrival_time'] or \
                (data[keys[j]]['arrival_time'] == data[keys[j + 1]]['arrival_time'] and data[keys[j]]['cost'][0] > data[keys[j + 1]]['cost'][0]):
                    keys[j], keys[j + 1] = keys[j + 1], keys[j]
        sorted_data = {k: data[k] for k in keys}
        return sorted_data




        # 3) based on the berth lengths And the allocation is based on the berth availability, 
        # 5) weather conditions and sea depth level
   
        # for now just a sample 

        # check if the allocated berth is different from the preferred berth
        # call the second objective function
        # and print the total updated cost
        # reassign all of the berths 
        # print the output in a CSV file 
        # plot bar charts for the updated costs 


    def create_time_slots_and_berths_csv(num_berths=8):
        column_headers = [f"Berth {i+1}" for i in range(num_berths)]
        time_slots = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 5)]

        df_time_slots = pd.DataFrame({"Time Slot": time_slots})
        df_berths = pd.DataFrame(columns=column_headers)
        df_berths.loc[0] = range(1, num_berths + 1)

        df_combined = pd.concat([df_time_slots, df_berths], axis=1)

        df_combined.to_csv("time_slots_and_berths.csv", index=False)

        print("CSV file 'time_slots_and_berths.csv' created successfully!")

    # Call the function to generate the CSV file
    create_time_slots_and_berths_csv()


