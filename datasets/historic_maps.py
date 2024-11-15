import os
import re

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

import torchvision.transforms as transforms

nations_dict = {'Australia': 0, 'Austria': 1, 'Belgium': 2, 'Canada': 3, 'China': 4, 'Denmark': 5, 'Egypt': 6, 'England': 7,
                'France': 8, 'Germany': 9, 'Greece': 10, 'Guernsey': 11, 'Iceland': 12, 'Ireland': 13, 'Isle-of-Man': 14, 'Italy': 15,
                'Japan': 16, 'Jersey': 17, 'Luxemburg': 18, 'Netherlands': 19, 'New-Zealand': 20, 'Norway': 21, 'Portugal': 22,
                'Russia': 23, 'Spain': 24, 'Sweden': 25, 'Switzerland': 26, 'Turkey': 27, 'United-States': 28} 

class HistoricMaps(DatasetBase):

    dataset_dir = 'historic-maps'

    def __init__(self, root, num_shots):
        

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'Satellite')
        self.split_path = os.path.join(self.dataset_dir, 'split_config.json')

        #train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train, val, test = self.read_split(self.split_path, self.image_dir)

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for today_path, historic_path in items:
                today_path = os.path.join(path_prefix, today_path)
                historic_path = os.path.join(path_prefix, historic_path)
                
                # paths are like ..\Japan/Tokyo/1.jpeg
                classname = re.split(r'[\\/]', historic_path)[-3]
                item = Datum(
                    impath=historic_path,
                    todaypath=today_path,
                    label=nations_dict[classname],
                    classname=classname,
                    task_type='retrieval'
                )
                #print(f"historic path: {item.label}")
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test