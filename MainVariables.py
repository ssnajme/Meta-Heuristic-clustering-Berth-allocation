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
                input("Enter the number of time windows for berths: ")
            )
            if num_time_windows <= 0:
                print("Please enter a positive integer for the number of time windows.")
                return self.get_user_input()  # Call the method recursively

            port_opening_hour = int(
                input("Enter the port opening hour (24-hour format): ")
            )
            port_closing_hour = int(
                input("Enter the port closing hour (24-hour format): ")
            )
            num_berths = int(
                input("Enter the number of berths in container terminal: ")
            )
            return num_time_windows, port_opening_hour, port_closing_hour, num_berths
        except ValueError:
            print(
                "Invalid input. Please enter valid positive integers for time windows and hours."
            )
            return self.get_user_input()  # Call the method recursively


# the inputs for  num_time_windows, port_opening_hour, port_closing_hour, num_berths
InputInstance = Input()
num_vessels = len(top_nests)
num_time_windows, port_opening_hour, port_closing_hour, num_berths = InputInstance.get_user_input()
#print(num_time_windows, port_opening_hour, port_closing_hour, num_berths)

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
    
    def initialize_berth_lengths(self, num_berths, min_length=50, max_length=350):
                """
        Initializes random berth lengths for each wharf within nests.

        Args:
            num_berths (int): Number of berths.

        Returns:
            list: List of lists representing berths lengths.
        """
                
                return [[random.randint(min_length, max_length)] for _ in range(num_berths)]
    
    def calculate_berth_distance(self, berth1, berth2):
            """
            Calculates the absolute distance between two berth lengths.

            Args:
                berth1 (int): First berth length.
                berth2 (int): Second berth length.

            Returns:
                int: Absolute distance between the berths.
            """
            return abs(berth1 - berth2)

## 
length_instance = length()
berth_length = length_instance.initialize_berth_lengths(num_berths)
vessel_length = length_instance.initialize_vessel_lengths(num_vessels)

# The value of Z_k
#berth1 = berth_length[berth1_index][0] # this comes from the preferred berth
#berth2 = berth_length[berth2_index][0] # and the actual berth allocated 

#berth_distance = length_instance.calculate_berth_distance()
print(berth_length)
print(vessel_length)


class PortTimeWindow:
    def __init__(
        self,
        port_opening_hour,
        port_closing_hour,
        num_vessels,
        berth_length,
        vessel_lengths,
    ):
        """
        Initializes the PortTimeWindow class.

        Args:
            port_opening_hour (int): The hour when the port opens (24-hour format).
            port_closing_hour (int): The hour when the port closes (24-hour format).
        """

        self.num_incoming_vessels = num_vessels
        self.berth_lengths = berth_length
        self.vessel_lengths = vessel_lengths
        self.quay_availability = np.ones(len(berth_length))
        self.processing_time = {}

        # self.port_opening_hour = port_opening_hour
        # self.port_closing_hour = port_closing_hour
        # self.vessel_lengths = vessel_lengths
        # self.processing_times = processing_times

    def initialize_time_window(self, num_time_windows, port_opening_hour, port_closing_hour):
        """
        Calculates the total time window (T) based on port operational hours.

        Args:
            num_time_windows (int): Number of time windows available for berths.
            port_opening_hour (int): The opening hour (24-hour format).
            port_closing_hour (int): The closing hour (24-hour format).

        Returns:
            tuple: (T_start, T_end, T) representing the time window.
        """
        # Convert opening and closing hours to minutes
        def convert_to_minutes(hours):
            return hours * 60


        # Calculate the total time window
        T = convert_to_minutes(port_closing_hour - port_opening_hour)

        # Distribute the time window equally among the time windows for berths
        time_window_per_berth = T // num_time_windows

        # Calculate start and end times for the time window
        T_start = port_opening_hour
        T_end = port_closing_hour

        return T_start, T_end, T, num_time_windows, time_window_per_berth
    

    def create_equal_time_windows(opening_minutes, closing_minutes, num_windows):
        """
        Creates equal-sized time windows based on the specified opening and closing minutes,
        and the desired number of windows.

        Args:
            opening_minutes (int): Opening time in minutes (e.g., 8:00 AM = 480 minutes).
            closing_minutes (int): Closing time in minutes (e.g., 5:00 PM = 1020 minutes).
            num_windows (int): Number of equally sized time windows.

        Returns:
            list: List of tuples representing time windows (start_time, end_time).
        """
        total_duration = closing_minutes - opening_minutes
        window_duration = total_duration // num_windows

        time_windows = []
        for i in range(num_windows):
            start_time = opening_minutes + i * window_duration
            end_time = start_time + window_duration
            time_windows.append((start_time, end_time))
        return time_windows
     
### 
port_instance = PortTimeWindow(
    port_opening_hour, port_closing_hour, num_vessels, berth_length, vessel_length
)
T_start, T_end, T, num_time_windows, time_window_per_berth = port_instance.initialize_time_window(
    num_time_windows, port_opening_hour, port_closing_hour)
print(f"Total time window (T) in minutes: {T_start, T_end, T, num_time_windows, time_window_per_berth}")


