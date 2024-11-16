import os
import re

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

import torchvision.transforms as transforms

nations_dict = {'Australia': 0, 'Austria': 1, 'Belgium': 2, 'Canada': 3, 'China': 4, 'Denmark': 5, 'Egypt': 6, 'England': 7,
                'France': 8, 'Germany': 9, 'Greece': 10, 'Guernsey': 11, 'Iceland': 12, 'Ireland': 13, 'Isle-of-Man': 14, 'Italy': 15,
                'Japan': 16, 'Jersey': 17, 'Luxemburg': 18, 'Netherlands': 19, 'New-Zealand': 20, 'Norway': 21, 'Portugal': 22,
                'Russia': 23, 'Spain': 24, 'Sweden': 25, 'Switzerland': 26, 'Turkey': 27, 'United-States': 28} 

city_dict = {'Amsterdam': 0, 'Utrecht': 1, 'Middelburg': 2, 'Breda': 3, 'Arnhem': 4, 'Maastricht': 5, 'Hertogenbosch': 6, 'Zwolle': 7, 'Alkmaar': 8, 
             'Leeuwarden': 9, 'Haarlem': 10, 'Nijmegen-1': 11, 'Groningen': 12, 'Nijmegen-2': 13, 'Wonderland': 14, 'New-Zealand': 15, 'Oslo': 16, 
             'Porto': 17, 'Lisbon': 18, 'Moscow': 19, 'Stockholm': 20, 'Basel-2': 21, 'Luzern-1': 22, 'Lausanne': 23, 'Bern': 24, 'Basel-1': 25, 
             'Luzern-2': 26, 'Geneva': 27, 'Zurich': 28, 'Lugano': 29, 'Istanbul': 30, 'London-3': 31, 'Manchester': 32, 'Argyll-and-Bute': 33, 'Glasgow-1': 34, 
             'Perth-1': 35, 'Carlisle': 36, 'Durham': 37, 'Truro': 38, 'Bristol-1': 39, 'Newport-1': 40, 'Southampton-1': 41, 'Bath-1': 42, 'Chichester-2': 43, 
             'Plymouth': 44, 'British-Isles': 45, 'Orkney': 46, 'Oxford': 47, 'Perth-2': 48, 'Glasgow-2': 49, 'Isle-of-Wight': 50, 'Newport-2': 51, 'Bristol-2': 52, 
             'Hull': 53, 'Chichester-1': 54, 'Bath-2': 55, 'Bangor': 56, 'Sheffield': 57, 'Southampton-2': 58, 'Edinburgh': 59, 'Cambridge': 60, 'York': 61, 
             'Brighton': 62, 'Portsmouth-2': 63, 'Inverness-1': 64, 'Newcastle-upon-Tyne-2': 65, 'Aberdeen-2': 66, 'Chelmsford': 67, 'Leeds': 68, 'London-2': 69, 
             'Liverpool': 70, 'Winchester': 71, 'Inverness-2': 72, 'Portsmouth-1': 73, 'Newcastle-upon-Tyne-1': 74, 'Aberdeen-1': 75, 'Shetland': 76, 'Sunderland': 77, 
             'Birmingham': 78, 'Derby': 79, 'London-1': 80, 'Bremen-3': 81, 'Hannover': 82, 'Nurnberg-2': 83, 'Koln-2': 84, 'Leipzig-2': 85, 'Koln-5': 86, 
             'Stuttgart': 87, 'Mannheim': 88, 'Frankfurt': 89, 'Koln-1': 90, 'Nurnberg-1': 91, 'Leipzig-1': 92, 'Lubeck-1': 93, 'Koln-4': 94, 'Koln-3': 95, 
             'Bremen-2': 96, 'Dusseldorf': 97, 'Magdeburg': 98, 'Lubeck-2': 99, 'Wurzburg': 100, 'Hamburg': 101, 'Munich': 102, 'Berlin': 103, 'Bremen-1': 104, 
             'Sydney': 105, 'Vienna': 106, 'Linz': 107, 'Graz': 108, 'Innsbruck': 109, 'Klagenfurt': 110, 'Kunming': 111, 'Chongqing': 112, 'Hongkong': 113, 
             'Nanchang': 114, 'Panzhihua': 115, 'Changsha': 116, 'Xiamen': 117, 'Shanghai': 118, 'Guangzhou': 119, 'Beijing': 120, 'Qingdao': 121, 'Dali': 122, 
             'Taiwan': 123, 'Dalian': 124, 'Tianjin': 125, 'Wuhan': 126, 'Chengdu': 127, 'Quebec': 128, 'Edmonton': 129, 'Winnipeg': 130, 'Montreal': 131, 
             'Vancouver': 132, 'Ottawa': 133, 'Copenhagen-3': 134, 'Odense': 135, 'Copenhagen-2': 136, 'Copenhagen-1': 137, 'Cairo-1': 138, 'Aswan-1': 139, 
             'Ismailia': 140, 'Cairo-2': 141, 'Aswan-2': 142, 'Cairo-3': 143, 'Napa': 144, 'Gloucester': 145, 'Portland': 146, 'Saginaw-1': 147, 'Seattle': 148, 
             'Honolulu': 149, 'Cincinnati': 150, 'Columbia': 151, 'Miami': 152, 'Charleston-1': 153, 'New-Orleans': 154, 'Wilmington-2': 155, 'Hopewell': 156, 
             'Kansas-City-2': 157, 'New-York': 158, 'Santa-Barbara': 159, 'San-Francisco': 160, 'Frederickburg': 161, 'Memphis': 162, 'Saginaw-2': 163, 
             'Cleveland': 164, 'Nashville': 165, 'Maui': 166, 'Rochester': 167, 'Petersburg': 168, 'Baltimore': 169, 'Wenatchee': 170, 'San-Diego': 171, 
             'Los-Angeles': 172, 'Charleston-2': 173, 'Molokai': 174, 'Wilmington-1': 175, 'Kansas-City-1': 176, 'Albany': 177, 'Leesburg': 178, 'Antioch': 179, 
             'Boston': 180, 'Columbus': 181, 'Eureka': 182, 'Vallejo': 183, 'New-Bern': 184, 'Pocatello': 185, 'Pittsburgh-2': 186, 'Tampa': 187, 'Detroit': 188, 
             'Mobile': 189, 'Chattanooga-1': 190, 'Atlanta': 191, 'Hawaii': 192, 'Boise': 193, 'Suffolk': 194, 'Billings': 195, 'Portsmouth': 196, 'Havre': 197, 
             'Sacramento': 198, 'Toledo': 199, 'Carson': 200, 'Lanai': 201, 'Jacksonville': 202, 'Walla-Walla': 203, 'Philadelphia': 204, 'Nampa': 205, 
             'Richland': 206, 'Pittsburgh-1': 207, 'Niagara-Falls': 208, 'Indianapolis': 209, 'Salt-Lake': 210, 'Houston': 211, 'Idaho-Falls': 212, 
             'Chattanooga-2': 213, 'Louisville': 214, 'St-Louis': 215, 'Washington': 216, 'Newburyport': 217, 'EI-Cajon': 218, 'Montrose': 219, 'Bluffton': 220, 
             'Youngstown': 221, 'Pasco': 222, 'Austin': 223, 'Dunkirk-1': 224, 'Rouen-1': 225, 'Strasbourg-3': 226, 'Paris-2': 227, 'Toulouse': 228, 'Brest': 229, 
             'Avignon': 230, 'Lille-2': 231, 'Montpellier': 232, 'Rouen-2': 233, 'Dunkirk-2': 234, 'Paris-1': 235, 'Mulhouse': 236, 'Orleans-3': 237, 'Lyon': 238, 
             'Reims': 239, 'Lille-1': 240, 'Grenoble': 241, 'Orleans-1': 242, 'Paris-4': 243, 'Paris-3': 244, 'Tours': 245, 'Marseille-2': 246, 'Strasbourg-2': 247, 
             'Nancy': 248, 'Toulon': 249, 'Orleans-2': 250, 'Caen': 251, 'Troyes': 252, 'Bordeaux': 253, 'Strasbourg-1': 254, 'Marseille-1': 255, 'Athens': 256, 
             'Volos': 257, 'Corfu': 258, 'Guernsey': 259, 'Iceland': 260, 'Cork': 261, 'Limerick': 262, 'Dublin': 263, 'Isle-of-Man': 264, 'La-Spezia-2': 265, 
             'Padova': 266, 'Verona': 267, 'Turin': 268, 'Livorno': 269, 'Treviso': 270, 'La-Spezia-1': 271, 'Catania': 272, 'Vicenza': 273, 'Bologna': 274, 
             'Rome': 275, 'Venice': 276, 'Cuneo': 277, 'Milan-1': 278, 'Naples-2': 279, 'Florence': 280, 'Genoa': 281, 'Milan-2': 282, 'Parma': 283, 'Naples-1': 284, 
             'Siracusa': 285, 'Luxemburg': 286, 'Valladolid': 287, 'Barcelona': 288, 'Madrid-3': 289, 'Malaga': 290, 'A-Coruna': 291, 'Alacant': 292, 'Cadiz-2': 293, 
             'Madrid-2': 294, 'Zaragoza-2': 295, 'Bilbao': 296, 'Tarragona': 297, 'Cadiz-1': 298, 'Madrid-1': 299, 'Vigo': 300, 'Murcia': 301, 'Zaragoza-1': 302, 
             'Liege': 303, 'Brugge': 304, 'Tournai-1': 305, 'Mons-1': 306, 'Oostende': 307, 'Tournai-2': 308, 'Mons-2': 309, 'Gent-1': 310, 'Kortrijk': 311, 
             'Leuven': 312, 'Gent-2': 313, 'Antwerpen': 314, 'Brussel': 315, 'Tokyo': 316, 'Jersey': 317}

class HistoricMaps(DatasetBase):

    dataset_dir = 'historic-maps'

    def __init__(self, root, num_shots):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'Satellite')
        self.split_path = os.path.join(self.dataset_dir, 'split_config.json')

        #train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train, val, test = self.read_split(self.split_path, self.image_dir)

        #n_shots_val = min(num_shots, 4)
        #val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for today_path, historic_path in items:
                today_path = os.path.join(path_prefix, today_path)
                historic_path = os.path.join(path_prefix, historic_path)
                
                # paths are like ..\Japan/Tokyo/1.jpeg
                classname = re.split(r'[\\/]', historic_path)[-2]
                item = Datum(
                    impath=historic_path,
                    todaypath=today_path,
                    label=city_dict[classname],
                    classname=classname,
                    task_type='retrieval'
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test