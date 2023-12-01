"""
This file creates one big feature.csv table out of several .csv tables provided by the authors of the DVM-Car dataset.
"""
import pandas as pd


def create_feature_df(filepath="../data/tables/", export_to_csv=True):
    # load dataframes
    basic_df = pd.read_csv(filepath + "Basic_table.csv")
    ad_df = pd.read_csv(filepath + "Ad_table.csv")
    image_names_df = pd.read_csv(filepath + "Image_table.csv")

    # rename columns and update dtypes
    column_name_dict = {
        "Maker": "Brand_Name",
        "Automaker": "Brand_Name",
        "Genmodel": "Model_Name",
        "Reg_year": "Launch_Year",
        "Adv_ID": "Advertiser_ID",
        "Genmodel_ID": "Model_ID",
        "Automaker_ID": "Brand_ID",
        "Predicted_viewpoint":"Viewpoint",

    }

    # loop over dataframes and rename column names
    for df in [basic_df, ad_df, image_names_df]:
        df.rename(column_name_dict, axis=1, inplace=True)
        if "Launch_Year" in df.columns:
            # check for Nans
            df = df.dropna(axis=0)
            df["Launch_Year"] = df["Launch_Year"].astype("int")
            assert df["Launch_Year"].dtype == "int"

    # create dataframe that contains first selling year per auto id
    year_df = ad_df.groupby(["Model_ID", "Bodytype"])["Launch_Year"].min().reset_index()

    year_df["Launch_Year"] = year_df["Launch_Year"].astype("int")
    assert year_df["Launch_Year"].dtype == "int"

    # create dataframe that contains information of image-file-paths
    file_attributes = [
        "Brand_Name",
        "Model_Name",
        "Launch_Year",
        "Color",
        "Model_ID",
        "Advertiser_ID",
        "Image_ID",
        "Viewpoint",
        "file_path",
    ]

    row_list = []

    len_feature_table = image_names_df.shape[0]
    img_names_lst = image_names_df["Image_name"].tolist()
    img_viewpoint_lst = image_names_df["Viewpoint"].tolist()

    for i in range(len_feature_table): 
        # store file path
        file_path = img_names_lst[i]
        viewpoint = img_viewpoint_lst[i]
        values_xs = file_path.split("$$")
        values_xs.append(viewpoint)
        values_xs.append(file_path)

        # add row to df
        row_list.append(dict(zip(file_attributes, values_xs)))

    image_values_df = pd.DataFrame(row_list)
    image_values_df["Launch_Year"] = image_values_df["Launch_Year"].astype("int")
    assert image_values_df["Launch_Year"].dtype == "int"

    # merge dataframes
    features_df = basic_df.merge(year_df)
    features_df = features_df.merge(image_values_df)
    
  

    # drop rows where Launch_Year < 1980 since launch_year = 1900 in Vauxhaul brand exists (probably due to data entry error)
    features_df = features_df[features_df['Launch_Year'] > 1980]

    # option: export features_df df to csv
    if export_to_csv:
        features_df.to_csv(filepath + "features.csv", encoding="utf-8", index=False)

    return features_df
