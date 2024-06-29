from MainVariables import berth_length, vessel_length
from ztest import vessel_id

class PreferredVessel:
    def __init__(self, berth_length, vessel_length):
        self.berth_lengths = berth_length
        self.vessel_lengths = vessel_length

    def assign_berths(self):
        preferred_berths = {}

        for vessel_id, vessel_length_list in enumerate(self.vessel_lengths):
                    differences = [
                        [abs(vessel_length - berth_length[0]) for berth_length in berth_length]
                        for vessel_length in vessel_length_list
                    ]
                    preferred_berth_indices = [diff.index(min(diff)) for diff in differences]

                    preferred_berths[vessel_id] = preferred_berth_indices
                    
                    #print(preferred_berths)
        return preferred_berths
           

preferred_vessel = PreferredVessel(berth_length, vessel_length)
preferred_berths = preferred_vessel.assign_berths()
print(f"Preferred Berths: {preferred_berths}")