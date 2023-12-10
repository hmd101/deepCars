from . import data_cleaning
import pandas as pd
import numpy as np


def split_dataset_dfs(data_path:str, data_set_ratios:tuple=(0.7,0.1,0.2), seed:int=100):

    # test if data_set ratios sum up to 1.0
    if np.asarray(data_set_ratios).sum() != 1.0:
        raise ValueError("Data set splits don't sum up to 1.0.")

    # load features dataframe
    features_df = pd.read_csv(data_path) 

    #  random number generator
    rng = np.random.default_rng(seed)
    # TODO: stratify test, train, val by classes (e.g., brand, year, etc.)
    # we need to make sure that images of a car model (specified by Model_ID and launch_year) are not distributed over training, validation and test dataset when splitting.    


    # subset dataframe by car model (specified by launch_year and model_id) and find unique combinations and then sample accordingly
    unique_cars = features_df[["Model_ID", "Launch_Year"]].drop_duplicates()

    # draw random indices according to splits from unique cars
    rand_nums = rng.uniform(size = len(unique_cars))

    # training dataset:
    # draws random numbers from uniform distribution,
    msk_train = rand_nums < data_set_ratios[0]

    msk_val_test = ~msk_train # complement


    # subset unique cars by indices
    #   validation dataset
    msk_val = rand_nums >= 1 - data_set_ratios[1] 

    #msk_test = ~msk_val & ~msk_train #  20%
    msk_test = (rand_nums > data_set_ratios[0]) & (rand_nums < (1- data_set_ratios[1]))

    # test if all datasets only contain unique cars
    assert not (np.any(msk_val_test & msk_train))
    assert not (np.any(msk_val & msk_test))


    # slice unique cars with datamasks 
    train_cars_df = unique_cars[msk_train]
    test_cars_df = unique_cars[msk_test]
    val_cars_df = unique_cars[msk_val]


    # check unique cars len is equal to the sum of its sub dfs
    assert len(val_cars_df) + len(test_cars_df) + len(train_cars_df) == len(unique_cars)


    # complement unique cars per dataset with corresponding rows 
    # train set
    train_df = features_df[features_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1).isin(train_cars_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1))]

    
    # val set
    val_df = features_df[features_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1).isin(val_cars_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1))]

    # test set
    test_df = features_df[features_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1).isin(test_cars_df[['Launch_Year', 'Model_ID']].apply(tuple, axis=1))]

    # check if len of entire data set is equal to sum of test, train and val dataset
    assert len(test_df) + len(train_df) + len(val_df) == len(features_df)

    return train_df, val_df, test_df