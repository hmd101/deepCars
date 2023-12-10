from typing import Callable
from skimage import io
import pandas as pd
import torch
import numpy as np
from torchvision import transforms



def year2label_fn(year:int, min_year:int, max_year:int, year_bucket_size:int= 2) -> int:
    """
    converts year to label (int)
    in ranges determined through year_bucket_size
    """
    if year < min_year:
        raise ValueError(f"Year {year} smaller than the minimum year {min_year}.")
    if year > max_year:
        raise ValueError(f"Year {year} bigger than the maximum year {max_year}.")
    if (year <0)| (min_year <0)| (max_year <0) | (year_bucket_size <0):
        raise ValueError("One of the arguments is negative and should be >= 0.")

    year_range = max_year - min_year + 1
    num_buckets = year_range // year_bucket_size
    year_ratio = (year - min_year) / year_range
    label = int(np.floor( year_ratio * num_buckets))

    assert label >= 0
    return label



def bodytype2label_fn(bodytype:str, possible_bodytypes:list=['Convertible', 'Coupe', 'Hatchback', 'MPV', 'Saloon', 'Estate', 'Van',
       'SUV', 'Minibus', 'Pickup',
       'Manual', 'Tipper', 'Camper', 'Chassis Cab',
       'Limousine']) -> int:
    """
    converts body-type to label
    """
    # if bodytype contains "van": then bodytype = van
    # all van bodytypes: 'Combi Van', 'Panel Van', 'Window Van', 'Car Derived Van'
    if "Van" in bodytype:
        bodytype = "Van"

    
    # create dictionary key:bodytype, value: label
    # create labels in range of bodytype list length
    labels_list = np.arange(len(possible_bodytypes)).tolist()
    bodytype2label_dict = dict(zip(possible_bodytypes, labels_list))
    return  bodytype2label_dict[bodytype]


class CarDataset(torch.utils.data.Dataset):
    """
    DVM-CAR dataset (A Large-Scale Automotive Dataset for Visual Marketing Research and Applications)
    """

    def __init__(
        self, 
        features:pd.DataFrame,
        year2label_fn:Callable, 
        bodytype2label_fn:Callable = bodytype2label_fn,
        transform:Callable = None,
        all_cars:bool = True,
        img_root_dir:str = "../raw_data/", 
    ):
        self.features = features
        self.bodytype2label_fn = bodytype2label_fn
        self.year2label_fn = year2label_fn
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.all_cars = all_cars
        



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # access row indicated by idx and extrac values to be returned except for image
        row_of_interest = self.features.iloc[idx] 
        bodytype = row_of_interest["Bodytype"]
        launch_year = row_of_interest["Launch_Year"]
        model_id = row_of_interest["Model_ID"]
        viewpoint = row_of_interest["Viewpoint"]

         # to get image, concatenate root-dir and file-path in features df
         # different image organizatin depending on all cars datset or small datset
        if self.all_cars:
            image_file_path = self.img_root_dir  +  "all_cars/"+ row_of_interest["Brand_Name"] + "/"+ str(row_of_interest["Model_Name"])+"/"+ str(row_of_interest["Launch_Year"])+"/"+str(row_of_interest["Color"])+"/" + row_of_interest["file_path"]
        else:
            image_file_path = self.img_root_dir  + "confirmed_fronts/"+ row_of_interest["Brand_Name"] + "/"+ str(row_of_interest["Launch_Year"])+"/" + row_of_interest["file_path"]

        # load image as tensor
        image = io.imread(image_file_path)

        # transform image
        if self.transform is not None:
            image = self.transform(image)


        return image, bodytype, model_id, launch_year, self.bodytype2label_fn(bodytype), self.year2label_fn(year=launch_year), viewpoint
   

# Data agumentation and normalization for training
img_rgb_mean=[0.485, 0.456, 0.406]
img_rgb_std = [0.229, 0.224, 0.225]

data_transforms = {
    "train":transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.ColorJitter(),
            transforms.RandomInvert(),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
            transforms.Normalize(mean=img_rgb_mean,std = img_rgb_std),
        ]
    ),
    "val":transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
                        transforms.Normalize(mean=img_rgb_mean,std = img_rgb_std),

        ]
    )
}

def inverse_transform(y:torch.Tensor, mean:list=img_rgb_mean, std:list=img_rgb_std):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    mean = torch.reshape(mean,[3,1,1])
    std = torch.reshape(std,[3,1,1])
    return y*std + mean