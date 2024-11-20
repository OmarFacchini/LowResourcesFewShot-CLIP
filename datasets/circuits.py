import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

import torchvision.transforms as transforms


circuit_classes = ["others", "audio amplifier/amplifier", "converter/power supply/charger/inverter", "mosquito repellent", "infrared sensor",
                   "(infrared) controller", "lock", "alarm", "meter/sensor/monitor/tester", "LED", "timer", "dice/toss/random number generator",
                   "(AM/FM/infrared/RF) transmitter", "(AM/FM/infrared/RF) receiver", "sound generator/tone control/tone generator", "switch",
                   "protector", "counter", "indicator", "bell/buzzer/ringer/horn/siren/beeper", "audio mixer", "warmer", "wave generator/signal generator/multivibrator",
                   "(motor) driver", "signal converter", "signal fader", "relay", "UPS", "jammer", "limiter/regulator", "fan", "intercom"]

label_map = {0: "others", 1: "audio amplifier/amplifier", 2: "converter/power supply/charger/inverter", 3: "mosquito repellent",
             4: "infrared sensor", 5: "(infrared) controller", 6: "lock", 7: "alarm", 8: "meter/sensor/monitor/tester",
             9: "LED", 10: "timer", 11: "dice/toss/random number generator", 12: "(AM/FM/infrared/RF) transmitter",
             13: "(AM/FM/infrared/RF) receiver", 14: "sound generator/tone control/tone generator", 15: "switch",
             16: "protector", 17: "counter", 18: "indicator", 19: "bell/buzzer/ringer/horn/siren/beeper",
             20: "audio mixer", 21: "warmer", 22: "wave generator/signal generator/multivibrator", 23: "(motor) driver",
             24: "signal converter", 25: "signal fader", 26: "relay", 27: "UPS", 28: "jammer", 29: "limiter/regulator",
             30: "fan", 31: "intercom"}

circuit_templates = ["a photo of a {}."]

class Circuits(DatasetBase):

    dataset_dir = 'circuit-diagrams'

    def __init__(self, root, num_shots):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'data')
        self.split_path = os.path.join(self.dataset_dir, 'split_config.json')

        self.template = circuit_templates

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        n_shots_val = min(num_shots, 4)
        #val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

        

