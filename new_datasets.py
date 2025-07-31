import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tabpfn.datasets.dist_shift_datasets import (
    dataframe_to_distribution_shift_ds,
    TASK_TYPE_MULTICLASS
)
from dynamic_domains import master_shift


MODULE_DIR = "Drift-Resilient_TabPFN/tabpfn/datasets"


def get_housing_ames_data(config):
    """
    @article{cock2011ames,
        author = {De Cock, Dean},
        year = {2011},
        month = {11},
        pages = {},
        title = {Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project},
        volume = {19},
        journal = {Journal of Statistics Education},
        doi = {10.1080/10691898.2011.11889627}
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/housing_ames.csv"), keep_default_na=False
    )

    orderings = {
        "Street": ["Pave", "Grvl"],
        "Alley": ["Pave", "Grvl", "NA"],
        "Utilities": ["AllPub", "NoSewr", "NoSeWa", "ELO"],
        "LandSlope": ["Gtl", "Mod", "Sev"],
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"],
        "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"],
        "CentralAir": ["Y", "N"],
        "Electrical": ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix", "NA"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "Functional": ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
        "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "GarageFinish": ["Fin", "RFn", "Unf", "NA"],
        "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "PavedDrive": ["Y", "P", "N"],
        "PoolQC": ["Ex", "Gd", "TA", "Fa", "NA"],
        "Fence": ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"],
    }

    for col, ordering in orderings.items():
        df[col] = pd.Categorical(df[col], categories=ordering, ordered=True)

    # Set MSSubClass as categorical since it denotes a type not a number
    df["MSSubClass"] = pd.Categorical(df["MSSubClass"])

    # Delete the ID column
    df.drop("Id", axis=1, inplace=True)

    # Create the domain column in which every 10 year period is considered a new domain
    min_year = df["YearBuilt"].min()
    max_year = df["YearBuilt"].max()

    df.sort_values(by=["YearBuilt"], inplace=True)

    # Create a new column 'domain', each unique group will have a unique identifier
    # Define bins as every 15 years
    bins_domain = np.arange(min_year, max_year + 15, 15)

    if config['mode'] == "og":
        df["domain"] = pd.cut(df["YearBuilt"], bins=bins_domain, right=False).cat.codes
    else:
        config['data'] = df
        df = master_shift(config=config)


    bins_target = [0, 125000, 300000, np.inf]
    labels = ["<= 125k", "125k-300k", "> 300k"]

    # Discretize the sale price of a housing, use fixed size bins to create 5 bins
    df["SalePrice"] = pd.cut(
        df["SalePrice"], bins=bins_target, labels=labels, include_lowest=True
    ).cat.codes

    return dataframe_to_distribution_shift_ds(
        name="Ames Housing Prices",
        df=df,
        target="SalePrice",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_chess_data(config):
    """
    @article{vzliobaite2011combining,
      title={Combining similarity in time and space for training set formation under concept drift},
      author={{\v{Z}}liobait{\.e}, Indr{\.e}},
      journal={Intelligent Data Analysis},
      volume={15},
      number={4},
      pages={589--611},
      year={2011},
      publisher={IOS Press}
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/chess.csv"), sep=",", skipinitialspace=True
    )

    # Date ranged from 7 December 2007 to 26 March 2010
    # Build the date column out of the year, month and day column
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Sort by the date
    df = df.sort_values(by="date").reset_index(drop=True)

    # Group every 20 consecutive games as a single domain, this should better track the progress a player makes
    # as the time contains gaps, that are probably not relevant to a player's progress
    if config['mode'] == "og":
        df["domain"] = df.index // 20
    else:
        config['data'] = df
        if config['shift_col'] == 'default':
            df['temp_col'] = df.index.values
            config['shift_col'] = 'temp_col'
        df = master_shift(config=config)
        if 'temp_col' in df.columns:
            df.drop(columns=['temp_col'], inplace=True)
    # Change the data type of some columns
    cat_columns = ["white/black", "type", "outcome"]
    df[cat_columns] = df[cat_columns].astype("category")

    # Drop the columns not intended for the model
    df.drop(["date"], axis=1, inplace=True)

    return dataframe_to_distribution_shift_ds(
        name="Chess",
        df=df,
        target="outcome",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


## figure these out 

def get_parking_birmingham_data(config):
    """
    @misc{misc_parking_birmingham_482,
      author       = {Stolfi,Daniel},
      title        = {{Parking Birmingham}},
      year         = {2019},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C51K5Z}
    }

    https://archive.ics.uci.edu/dataset/482/parking+birmingham

    This dataset is licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(os.path.join(MODULE_DIR, "data/parking_birmingham.csv"), sep=",")

    # Convert 'LastUpdated' to datetime format
    data["LastUpdated"] = pd.to_datetime(
        data["LastUpdated"], format="%Y-%m-%d %H:%M:%S"
    )

    # Create 'Percentage_Occupied' column and discretize into intervals of 25 percent
    data["Percentage_Occupied"] = (data["Occupancy"] / data["Capacity"]) * 100
    # values smaller than 25 get 0, values between 25 and 50 get 1, values between 50 and 75 get 2, values larger than 75 get 3
    data["Percentage_Occupied"] = np.digitize(
        data["Percentage_Occupied"], bins=[25, 50, 75]
    ).astype(int)
    # Create 'Day', 'Week', and 'Domain' columns
    data["Hour"] = data["LastUpdated"].dt.hour
    data["Day"] = data["LastUpdated"].dt.day
    data["Month"] = data["LastUpdated"].dt.month
    data = data.sample(frac=0.2, random_state=42).reset_index(drop=True)
    
    if config['mode'] == "og":
        data["domain"] = (
            data["LastUpdated"].dt.isocalendar().week
            - min(data["LastUpdated"].dt.isocalendar().week)
        ).astype(int)

    else:
        config['data'] = data
        data = master_shift(config=config)

    # Filter the data to only include the car park with the largest capacity
    data = data[data["SystemCodeNumber"] == "Others-CCCPS133"]

    # Remove 'LastUpdated', 'SystemCodeNumber', 'Week' and 'Year' columns
    data.drop(["LastUpdated", "SystemCodeNumber", "Occupancy"], axis=1, inplace=True)
    

    return dataframe_to_distribution_shift_ds(
        "Parking Birmingham",
        data,
        "Percentage_Occupied",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


def get_folktables_data(states = ["MD"], config = {}):
    from folktables import (
        ACSDataSource,
        ACSIncome,
        ACSPublicCoverage,
        ACSMobility,
        ACSEmployment,
        ACSTravelTime,
    )

    # Create a dictionary to store all dataframes
    all_data = {
        "ACSIncome": None,
        "ACSPublicCoverage": None,
        #'ACSMobility': None,
        "ACSEmployment": None,
        #'ACSTravelTime': None
    }

    acs_data_2015_2019 = ACSDataSource(
        survey_year=2019,
        horizon="5-Year",
        survey="person",
        root_dir="./datasets/data/folktables",
    ).get_data(states=states, download=True)
    acs_data_2017_2021 = ACSDataSource(
        survey_year=2021,
        horizon="5-Year",
        survey="person",
        root_dir="./datasets/data/folktables",
    ).get_data(states=states, download=True)

    # Features that are in the 2015-2019 dataset but not in the 2017-2021 dataset
    # {'MLPI', 'MLPK'}
    # Features that are in the 2017-2021 dataset but not in the 2015-2019 dataset
    # {'MLPIK'}
    # Since our tasks dont use them, lets just drop them

    acs_data_2015_2019.drop(columns=["MLPI", "MLPK"], inplace=True)
    acs_data_2017_2021.drop(columns=["MLPIK"], inplace=True)

    # Build a year column that is the year an individual was surveyed
    acs_data_2015_2019["YEAR"] = (
        acs_data_2015_2019["SERIALNO"].apply(lambda x: x[:4]).astype(int)
    )
    acs_data_2017_2021["YEAR"] = (
        acs_data_2017_2021["SERIALNO"].apply(lambda x: x[:4]).astype(int)
    )

    # Merge the two datasets
    acs_data = pd.concat(
        [acs_data_2015_2019, acs_data_2017_2021[acs_data_2017_2021["YEAR"] > 2019]]
    )

    # Rename the columns that were renamed in 2019 by ACS to be consistent with the feature names as stated by folktables
    acs_data.rename(columns={"RELSHIPP": "RELP", "JWTRNS": "JWTR"}, inplace=True)

    used_cat_columns = {
        "DREM",
        "FER",
        "POWPUMA",
        "SCHL",
        "MIL",
        "MAR",
        "CIT",
        "TARGET",
        "DIS",
        "PUMA",
        "ANC",
        "NATIVITY",
        "ESP",
        "COW",
        "RELP",
        "RAC1P",
        "OCCP",
        "POBP",
        "DEAR",
        "ESR",
        "ST",
        "DEYE",
        "MIG",
        "GCL",
        "JWTR",
        "SEX",
    }

    # Data processing for each task
    tasks = [
        (ACSIncome, "ACSIncome"),
        (ACSPublicCoverage, "ACSPublicCoverage"),
        # (ACSMobility, 'ACSMobility'),
        (ACSEmployment, "ACSEmployment"),
    ]
    # (ACSTravelTime, 'ACSTravelTime')]

    random_state = 0
    for task, task_name in tasks:
        # We'd like to keep the new YEAR feature in our tasks
        if "YEAR" not in task.features:
            task.features.append("YEAR")

        features, labels, _ = task.df_to_pandas(acs_data)
        features["TARGET"] = labels

        # Convert the categorical features to be categorical and not numerical
        cat_columns = list(set.intersection(set(features.columns), used_cat_columns))
        features[cat_columns] = features[cat_columns].apply(
            lambda x: x.astype("category")
        )

        instances = 1300
        instances_per_year = round(instances / 7)  # 7 years of data

        # Subsample the data to make it more manageable
        subsampled_dfs = []
        for year in range(2015, 2021 + 1):
            domain_instances = features[features["YEAR"] == year].shape[0]
            # subsampled_instances = round(expected_instances_per_year * domain_instances / overall_instances)

            # Use stratified sampling to ensure that the distribution of the target is preserved
            subsampled_df = (
                features[features["YEAR"] == year]
                .groupby("TARGET", observed=False)
                .apply(
                    lambda x: x.sample(
                        frac=instances_per_year / domain_instances,
                        random_state=random_state,
                    )
                )
                .droplevel(0)
            )

            # Restore the order relative to each other
            subsampled_df.sort_index(inplace=True)

            subsampled_dfs.append(subsampled_df)

            random_state += 1

        # concatenate all subsampled DataFrames
        features = pd.concat(subsampled_dfs)

        all_data[task_name] = features

    # Reset index of each DataFrame
    for df in all_data.values():
        df.reset_index(drop=True, inplace=True)

    dataset_list = []

    for task, task_name in tasks:

        if config['mode'] == 'og':
            df = all_data[task_name]
            domain_name = "YEAR"
        else:
            config['data'] = all_data[task_name]
            df = master_shift(config)
            domain_name = "domain"

        dataset = dataframe_to_distribution_shift_ds(
            name=f"Folktables - {task_name} - {' | '.join(states)}",
            df=df,
            target="TARGET",
            domain_name=domain_name,
            task_type=TASK_TYPE_MULTICLASS,
            dataset_source="real-world",
            shuffled=False,
        )
        dataset_list.append(dataset)

    return dataset_list





