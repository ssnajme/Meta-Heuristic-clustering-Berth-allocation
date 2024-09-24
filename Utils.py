import random
import numpy as np
from BaseCuckooSearch import top_nests

class Input: 
    def __init__(self):
        # Constructor 
        self.port_opening_hour = None
        self.port_closing_hour = None
      
    def get_user_input(self):
        """
        Get user input for the number of time windows and port hours.

        Returns:
            tuple: (num_time_windows, port_opening_hour, port_closing_hour)
        """
        try:
            num_time_windows = int(
                input("Enter the number of hours for each time windows for berths: ")
            )
            if num_time_windows <= 0:
                print("Please enter a positive integer for the number of time windows in hours")
                return self.get_user_input()  # Call the method recursively

            port_operating_days = int(
                input("Enter the number of port operating days: ")
            )
            port_operating_nights = int(
                input("Enter the number of port operating nights : ")
            )
            num_berths = int(
                input("Enter the number of berths in container terminal: ")
            )
            return num_time_windows, port_operating_days, port_operating_nights, num_berths
        except ValueError:
            print(
                "Invalid input. Please enter valid positive integers for time windows and hours."
            )
            return self.get_user_input()  # Call the method recursively

class length: 
    def initialize_vessel_lengths(self, num_vessels):
        """
        Initializes random vessel lengths for each vessel within nests.

        Args:
            num_vessels (int): Number of vessels.

        Returns:
            list: List of lists representing vessel lengths.
        """
        # Generate random lengths for each vessel (assuming a range of 50 to 350)
        vessel_lengths = []
        for _ in range(num_vessels):
            lengths = [random.randint(50, 350)]
            vessel_lengths.append(lengths)
        return vessel_lengths
    
    def initialize_berth_lengths(self, num_berths, min_length=90, max_length=450):
                """
        Initializes random berth lengths for each wharf within nests.

        Args:
            num_berths (int): Number of berths.

        Returns:
            list: List of lists representing berths lengths.
        """
                
                return [[random.randint(min_length, max_length)] for _ in range(num_berths)]

class PortTimeWindow:
    def __init__(
        self,
        port_operating_days,
        port_operating_nights,
        num_vessels,
        berth_lengths,
        vessel_lengths,
    ):
        self.num_incoming_vessels = num_vessels
        self.berth_lengths = berth_lengths
        self.port_operating_nights = port_operating_nights
        self.port_operating_days = port_operating_days
        self.vessel_lengths = vessel_lengths
        self.time_slots = {}
    
    def create_time_slots(self, time_window_length):
        """
        Create time slots for each berth based on the specified time window length.

        Args:
            time_window_length (int): The length of each time window in hours.
        """
        for berth_id, berth_length in enumerate(self.berth_lengths):
            total_time_slots = 0
            for day in range(1, self.port_operating_days + 1):
                total_time_slots += 12 // time_window_length
            for night in range(self.port_operating_days + 1, self.port_operating_days + self.port_operating_nights + 1):
                total_time_slots += 12 // time_window_length
            self.time_slots[berth_id] = total_time_slots

class uncertainFactors: 
    def weather_conditions(self):  
        # Define positive and negative weather conditions  
        positive_conditions = [  
            "clear_sky",  
            "light_breeze",  
            "mild_temperature",  
            "sunny",  
            "partly_cloudy",  
            "warm_temps",  
            "gentle_rain",          
            "clear_night",          
            "comfortable_humidity",  
            "mild_wind",            
            "sunny_intervals",      
            "mid_sunrise",     
            "bright_sunset"      
        ]  
        
        negative_conditions = [  
            "high_winds",  
            "poor_visibility",  
            "heavy_rain",  
            "storm",  
            "fog",  
            "snow",  
            "hail",  
            "extreme_temperatures",  # High heat or cold  
            "lightning",  
            "thunderstorms"  
        ]  

        # Initialize weather conditions  
        weather_conditions = {}  

        # Set positive conditions  
        for condition in positive_conditions:  
            weather_conditions[condition] = random.choice([True, True, True, True, False])  # More True values  

        # Set negative conditions  
        for condition in negative_conditions:  
            weather_conditions[condition] = random.choice([False, False, True])  # More False values  
        return weather_conditions  

    def tide_levels(self, num_levels):
        tide_levels = [round(random.uniform(0.0, 1.0), 2) for _ in range(num_levels)]  
        return tide_levels 
    
    def identify_critical_levels(self, tide_levels, low_threshold=0.2, high_threshold=0.8):  
        critical_levels = []  
        for level in tide_levels:  
            if level <= low_threshold:  
                critical_levels.append([level, "Low"])  
            elif level >= high_threshold:  
                critical_levels.append([level, "High"])  
        return critical_levels  

    def generate_water_depths(self, num_berths, min_depth=5.0, max_depth=20.0):  
        # Generate random water depths between specified min and max values  
        water_depths = [round(random.uniform(min_depth, max_depth), 1) for _ in range(num_berths)]  
        return water_depths  


#------- The inputs for  num_time_windows, port_opening_hour, port_closing_hour, num_berths------
InputInstance = Input()
num_vessels = len(top_nests)
num_time_windows, port_operating_days, port_operating_nights, num_berths = InputInstance.get_user_input()

# -------------The inputs for the lengths-----------------------------
length_instance = length()
berth_length = length_instance.initialize_berth_lengths(num_berths)
vessel_length = length_instance.initialize_vessel_lengths(num_vessels)
print(berth_length)
print(vessel_length)

# ---------------The inputs for the operating days--------------------
port_operating_days = port_operating_days  
port_operating_nights = port_operating_nights  
num_vessels = num_vessels
berth_lengths = berth_length
vessel_lengths = vessel_length 

time_window_length = num_time_windows
port_time_window = PortTimeWindow(port_operating_days, port_operating_nights, num_vessels, berth_lengths, vessel_lengths)
port_time_window.create_time_slots(time_window_length)

# Access the total number of time slots for each berth
for berth_id, total_time_slots in port_time_window.time_slots.items():
    print(f"Berth {berth_id} - Total Number of Time Slots in hours: {total_time_slots}") 

# ----- The input for uncertain factors -------------------------------
uncertainFactors_instance = uncertainFactors()
weather_conditions  =  uncertainFactors_instance.weather_conditions()
tide_levels = uncertainFactors_instance.tide_levels(num_berths)
identify_critical_levels = uncertainFactors_instance.identify_critical_levels(tide_levels)
# --- Exactly based on the number of berths ---------------------------
water_depths_instance = uncertainFactors_instance.generate_water_depths(num_berths)

print("Weather Conditions:", weather_conditions)  
print("Tide Levels:", tide_levels)  
print("identify_critical_levels:", identify_critical_levels)  
print("generate_water_depths:", water_depths_instance) 