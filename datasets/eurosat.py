import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

"""
template = ['a centered satellite photo of {}.']
"""
template = ['a photo of a {}.']

NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        
        self.template = template

        #train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train, val, test = self.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items, train=False):
            out = []
            for impath, label, classname in items:
                # get folder(class) and filename
                folder, filename = os.path.split(impath)

                # number of filename, eg annualcrop_181 -> 181
                number = filename.split("_")[1].split(".")[0] 

                # path of original image
                # path_prefix = dataset/eurosat/2750
                # impath = dataset/eurosat/2750/AnnualCrop/annualCrop_1.png
                impath = os.path.join(path_prefix, impath) 

                # insert the original image
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname,
                    imgtype="original"
                )
                out.append(item)

                # concept of label preserving and breaking is applied only to training
                if train:
                    # dataset/dataset/eurosat/2750/label_preserving/AnnualCrop
                    preserving_dir_path = os.path.join(path_prefix, "label_preserving", folder)
                    
                    # dataset/dataset/eurosat/2750/label_breaking/AnnualCrop
                    breaking_dir_path = os.path.join(path_prefix, "label_breaking", folder)

                    # for each original image, take its preserving and its breaking images
                    for path in [preserving_dir_path, breaking_dir_path]:
                        # select imgtype based on path i am currently in
                        imgtype = "preserving" if "preserving" in path else "breaking"

                        # dataset/dataset/eurosat/2750/label_*/AnnualCrop/1
                        img_path = os.path.join(path, number)

                        '''check if folder exists, should always be true as we generate samples from train set
                        and enter here only if we are training'''
                        if os.path.exists(img_path):
                            # loop on all samples generated for the original image
                            # 1.png, 2.png.....
                            for filename in os.listdir(img_path):
                                # dataset/dataset/eurosat/2750/label_*/1.png
                                sample_path = os.path.join(img_path, filename)
                                
                                item = Datum(
                                    impath=sample_path,
                                    label=int(label),
                                    classname=classname,
                                    imgtype=imgtype
                                )
                                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'], True)
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test