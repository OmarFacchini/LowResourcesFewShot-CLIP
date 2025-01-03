from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .circuits import Circuits
from .historic_maps import HistoricMaps


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "circuits": Circuits,
                "historic_maps": HistoricMaps,
                }


def build_dataset(dataset, root_path, shots, preprocess, breaking_loss):
    if dataset == 'imagenet':
        return dataset_list[dataset](root_path, shots, preprocess)
    elif dataset == 'circuits':
        return dataset_list[dataset](root_path, shots, breaking_loss)
    else:
        return dataset_list[dataset](root_path, shots)