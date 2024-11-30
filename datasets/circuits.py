import os
import re
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

import torchvision.transforms as transforms
import random


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

    def __init__(self, root, num_shots, breaking_loss):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'data')
        self.split_path = os.path.join(self.dataset_dir, 'split_config.json')
        self.use_breaking_loss = breaking_loss

        self.template = circuit_templates

        train, val, test = self.read_split(self.split_path, self.image_dir, self.use_breaking_loss)

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    
    @staticmethod
    def read_split(filepath, path_prefix, use_breaking_loss):
        def _convert(items, train=False, use_breaking_loss=False):
            out = []
            for impath, label, classname in items:
                # remove .png and keep file name eg: 1
                original_filename = os.path.splitext(impath)[0] 

                # path of original image
                # path_prefix = dataset/circuit-diagrams/data
                # impath = dataset/circuit-diagrams/data/1.png
                impath = os.path.join(path_prefix, impath) 

                # concept of label preserving and breaking is applied only to training
                if train and use_breaking_loss:
                    # dataset/circuit-diagrams/data/label_preserving
                    preserving_dir_path = os.path.join(path_prefix, "label_preserving")
                    
                    # dataset/circuit-diagrams/data/label_breaking
                    breaking_dir_path = os.path.join(path_prefix, "label_breaking")

                    for preserving_path in [preserving_dir_path]:
                        preserving_img_path = os.path.join(preserving_path, original_filename)
                        if os.path.exists(preserving_img_path):
                            for path in [breaking_dir_path]:
                                breaking_paths = []
                                img_path = os.path.join(path, original_filename)
                                '''check if folder exists, should always be true as we generate samples from train set
                                and enter here only if we are training'''
                                if os.path.exists(img_path):
                                    # loop on all samples generated for the original image
                                    # 1.png, 2.png.....
                                    # get 2 breaking labels paths
                                    for filename in os.listdir(img_path):
                                        # dataset/circuit-diagrams/data/label_breaking/1.png
                                        sample_path = os.path.join(img_path, filename)
                                        breaking_paths.append(sample_path)

                                    # insert the original image with its breaking paths
                                    item = Datum(
                                        impath=impath,
                                        label=int(label),
                                        classname=label_map[int(label)],
                                        imgtype="original",
                                        breaking_paths = random.sample(breaking_paths,2) #select 2 samples
                                    )
                                    out.append(item)

                                    for filename in os.listdir(preserving_img_path):
                                        preserving_sample_path = os.path.join(preserving_img_path, filename)
                                    
                                        # insert preserving img with its breaking paths
                                        item = Datum(
                                            impath=preserving_sample_path,
                                            label=int(label),
                                            classname=label_map[int(label)],
                                            imgtype="preserving",
                                            breaking_paths = random.sample(breaking_paths,2) #select 2 samples
                                        )
                                        out.append(item)
                else:
                    # insert the original image with no breaking things
                    
                    item = Datum(
                        impath=impath,
                        label=int(label),
                        classname=label_map[int(label)],
                        imgtype="original",
                        breaking_paths = []
                    )
                    out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'], True, use_breaking_loss)
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test

        

