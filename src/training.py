import fire
import torch
from typing import Optional
from data.car_dataset import CarDataset, data_transforms, inverse_transform, year2label_fn

from data.data_splitting import split_dataset_dfs

from data import data_cleaning
from models import get_fine_tuneable_model

def training_loop(model, loss, optimizer, num_epochs=3,phase="train"):
    


def train(year_bucket_size:int = 2, data_subset_size:Optional[int]=10,features_path:str="../raw_data/tables/features.csv",batch_size:int=32, learning_rate:float=5e-4,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),seed:int=100):
    # set seed for random number generator
    # fixes order in dataloader
    torch.manual_seed(seed)

    # load all features
    #features_df = data_cleaning.create_feature_df() # recreates feature dataframe
    features_df = pd.read_csv(features_path)

    # create feature dataframes for train, val, test
    train_df, val_df, test_df = split_dataset_dfs(features_path)

    # set bucket_size for year -> how many years should correspond to one label?
    min_year = features_df["Launch_Year"].min() # oldest car launch_year in data
    max_year = features_df["Launch_Year"].max() # # most recent car launch_year in data

    # create training, test, and validation torch.dataset 
    if data_subset_size is not None:
        train_df = train_df[:data_subset_size]
        val_df = val_df[:data_subset_size]
        test_df = test_df[:data_subset_size]
        
    train_set = CarDataset(features=train_df, transform=data_transforms["train"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))

    test_set = CarDataset(features=test_df, transform=data_transforms["val"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))

    val_set = CarDataset(features=val_df, transform=data_transforms["val"],year2label_fn=lambda year:year2label_fn(year, min_year = min_year, max_year = max_year, year_bucket_size=year_bucket_size))



    # data loaders for all train, test, val datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)

    # instantiate model
    # compute number of classes of final layer which depends on year_bucket size and year_range
    year_range = max_year - min_year
    num_year_classes = 1 +  (year_range // year_bucket_size) # floor division

    my_model = get_fine_tuneable_model(num_classes=num_year_classes)
    my_model.to(device=device)
    # loss function
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device=device)

    # optimizer
    optimizer = torch.optim.SGD(my_mode.fc.paramaters(), lr=learning_rate)
    optimizer.to(device=device)

    # training loop


if __name__ == "__main__":
    fire.Fire(train)