import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tabpfn.datasets.dist_shift_datasets import (
    dataframe_to_distribution_shift_ds,
    TASK_TYPE_MULTICLASS
)
from dynamic_domains import get_dynamic_shifts, get_dynamic_shifts_multifeature
# mode : og, pelt, binseg
# penalty : float
# model : rbf, linear


MODULE_DIR = "Drift-Resilient_TabPFN/tabpfn/datasets"


def get_housing_ames_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf" ,
        shift_col = "default",
        use_pca = False,
        n_components = 5 
):
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
    if mode == "og":
        df["domain"] = pd.cut(df["YearBuilt"], bins=bins_domain, right=False).cat.codes
    else:
        if shift_col == "default":
            df = get_dynamic_shifts(
                df, 
                shift_col="YearBuilt", 
                penalty=penalty, 
                method=mode, 
                model=model
            )
        elif shift_col == "all_numeric":
            df = get_dynamic_shifts_multifeature(
                data = df, 
                features = df.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
                )

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

def get_chess_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf",
        shift_col = "default",
        use_pca = False,
        n_components = 5
):
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
    if mode == "og":
        df["domain"] = df.index // 20
    else:
        if shift_col == "default":
            df['temp_col'] = df.index
            df = get_dynamic_shifts(
                df, 
                shift_col='temp_col', 
                penalty=penalty, 
                method=mode, 
                model=model
            )
            df.drop(columns=['temp_col'], inplace=True)
        elif shift_col == "all_numeric":
            df = get_dynamic_shifts_multifeature(
                data = df, 
                features = df.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
            )
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

def get_parking_birmingham_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf"
):
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

    if mode == "og":
        # Create 'Day', 'Week', and 'Domain' columns
        data["Hour"] = data["LastUpdated"].dt.hour
        data["Day"] = data["LastUpdated"].dt.day
        data["Month"] = data["LastUpdated"].dt.month
        data["domain"] = (
            data["LastUpdated"].dt.isocalendar().week
            - min(data["LastUpdated"].dt.isocalendar().week)
        ).astype(int)

    else:
        data["Month"] = data["LastUpdated"].dt.month
        data = get_dynamic_shifts(
            data, 
            shift_col = "Month", 
            penalty = penalty, 
            method=mode, 
            model= model
        )

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


def get_folktables_data(states):
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
        dataset = dataframe_to_distribution_shift_ds(
            name=f"Folktables - {task_name} - {' | '.join(states)}",
            df=all_data[task_name],
            target="TARGET",
            domain_name="YEAR",
            task_type=TASK_TYPE_MULTICLASS,
            dataset_source="real-world",
            shuffled=False,
        )
        dataset_list.append(dataset)

    return dataset_list

def get_cleveland_heart_disease_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf" ,
        shift_col = "default",
        use_pca = False,
        n_components = 5
):
    """
    @misc{misc_heart_disease_45,
        author       = {Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert},
        title        = {{Heart Disease}},
        year         = {1988},
        howpublished = {UCI Machine Learning Repository},
        note         = {{DOI}: https://doi.org/10.24432/C52P4X}
    }

    https://archive.ics.uci.edu/dataset/45/heart+disease
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/cleveland_heart_disease.csv"), sep=",",
        na_values="?", keep_default_na=False
    )

    # Drop the few nan rows
    data.dropna(inplace=True)

    # Cast all columns but oldpeak to be integer
    cols = data.columns.drop("oldpeak")
    data[cols] = data[cols].apply(pd.to_numeric, downcast="integer", errors="coerce")

    # Set the type of some columns to be categorical
    cat_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    data[cat_columns] = data[cat_columns].apply(lambda x: x.astype("category"))

    data.sort_values("age", inplace=True)

    data["domain"] = pd.cut(
            data["age"],
            bins=np.arange(
                data["age"].min() - data["age"].min() % 4, data["age"].max() + 5, 4
            ),
            right=False,
        ).cat.codes

    # Build the domain, which is an age category of 5 year intervals
    if mode == "og":
        data["domain"] = pd.cut(
            data["age"],
            bins=np.arange(
                data["age"].min() - data["age"].min() % 4, data["age"].max() + 5, 4
            ),
            right=False,
        ).cat.codes
    else:
        if shift_col == "default":
            data = get_dynamic_shifts(
                data, 
                shift_col="age", 
                penalty=penalty, 
                method=mode, 
                model=model
            )
        elif shift_col == "all_numeric":
            data = get_dynamic_shifts_multifeature(
                data = data, 
                features = data.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
            )

    # treat all deseases as one class
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

    return dataframe_to_distribution_shift_ds(
        "Cleveland Heart Disease",
        data,
        "target",
        "domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_absenteeism_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf" ,
        shift_col = "default",
        use_pca = False,
        n_components = 5
):
    """

    @misc{misc_absenteeism_at_work_445,
      author       = {Martiniano,Andrea and Ferreira,Ricardo},
      title        = {{Absenteeism at work}},
      year         = {2018},
      howpublished = {UCI Machine Learning Repository},
      note         = {{DOI}: https://doi.org/10.24432/C5X882}
    }

    https://archive.ics.uci.edu/dataset/445/absenteeism+at+work

    This dataset was licensed under the CC BY 4.0 license.
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/absenteeism_at_work.csv"), sep=",",
        na_values="?", keep_default_na=False
    )
    

    # Add a column to track the change in 'Month of absence'
    data["Season_Change"] = data["Seasons"].ne(data["Seasons"].shift()).astype(int)

    # Identify when the year changes based on 'Month of absence'
    data["Season_Domain"] = (data["Season_Change"] > 0).cumsum()

    # Remove the temporary columns
    data.drop(["Season_Change"], axis=1, inplace=True)

    data["Absenteeism time in hours"] = pd.qcut(
        data["Absenteeism time in hours"], 4, labels=False
        )

    # Build the domain, which is an age category of 5 year intervals
    if mode == "og":
        data["Absenteeism time in hours"] = pd.qcut(
        data["Absenteeism time in hours"], 4, labels=False
        )
    else:
        if shift_col == "default":
            data = get_dynamic_shifts(
                data, 
                shift_col="Seasons", 
                penalty=penalty, 
                method=mode, 
                model=model
            )
        elif shift_col == "all_numeric":
            data = get_dynamic_shifts_multifeature(
                data = data, 
                features = data.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
            )

    # Remove the last three rows, seem to not fit the data description (time jump)
    data = data.iloc[:-3]

    # Cast columns to be categorical
    cat_columns = [
        "ID",
        "Reason for absence",
        "Day of the week",
        "Seasons",
        "Disciplinary failure",
        "Education",
        "Social drinker",
        "Social smoker",
        "Month of absence",
        "Season_Domain",
    ]

    data[cat_columns] = data[cat_columns].astype("category")

    return dataframe_to_distribution_shift_ds(
        "Absenteeism",
        data,
        "Absenteeism time in hours",
        "Season_Domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_electricity_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf" ,
        shift_col = "default",
        use_pca = False,
        n_components = 5    
):
    """..
    @Book{ harries1999splice,
        author = { Harries, Michael},
        title = { Splice-2 comparative evaluation: Electricity Pricing},
        publisher = { University of New South Wales, School of Computer Science and Engineering [Sydney] },
        year = { 1999 },
        type = { Book, Online },
        url = { http://nla.gov.au/nla.arc-32869 },
        language = { English },
        subjects = { Machine learning },
        life-dates = { 1999 -  },
        catalogue-url = { https://nla.gov.au/nla.cat-vn3513275 },
    }
    """
    df = pd.read_csv(
        os.path.join(MODULE_DIR, "data/elec2.csv"), sep=",",
        na_values="?", skipinitialspace=True, keep_default_na=False
    )
    
    # Convert the 'date' column to string type
    df["date"] = df["date"].astype(str)

    # Extract the year, month, and day from the 'date' column
    df["date"] = (
        "19"
        + df["date"].str[:2]
        + "-"
        + df["date"].str[2:4]
        + "-"
        + df["date"].str[4:6]
    )

    # Date ranged from 7 May 1996 to 5 December 1998
    # Create a new 'date' column using the extracted year, month, and day
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Drop nan rows
    df.dropna(inplace=True)

    # Drop half of the rows to reduce the size of the dataset
    df = df[df["half_hour_interval"] % 4 == 0]

    # Create the domain column in which every two week period is considered a new domain
    # Create a grouper object based on 'date' and group by it
    grouper = pd.Grouper(key="date", freq="1W")

    # Create a new column 'domain', each unique group will have a unique identifier
    df["domain"] = df.groupby(grouper).ngroup()

    # Drop the first and last group, since those can be incomplete weeks
    df = df[df["domain"] != 0]
    df = df[df["domain"] != df["domain"].max()]

    # Change the data type of some columns
    df = df.astype(
        {
            "day_of_week": "category",
            "half_hour_interval": "int",
            "nsw_demand": "float",
            "v_demand": "float",
            "transfer": "float",
        }
    )

    # Drop the columns not intended for the model
    df.drop(["date", "nsw_prize", "v_prize"], axis=1, inplace=True)

    # Subsample a range of 15 domains to reduce the size of the dataset even further
    np.random.seed(0)  # Fixing the seed for reproducibility

    range = 15
    max_start = df["domain"].max() - range
    start = np.random.randint(1, max_start + 1)
    end = start + range

    df = df.sample(n=1000,replace=False).copy()

    if mode == "og":
        df = df[(df["domain"] >= start) & (df["domain"] < end)]
    else:
        if shift_col == "default":
            df = get_dynamic_shifts(
                df, 
                shift_col="domain", 
                penalty=penalty, 
                method=mode, 
                model=model
            )
        elif shift_col == "all_numeric":
            df = get_dynamic_shifts_multifeature(
                df = df, 
                features = df.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
            )

    return dataframe_to_distribution_shift_ds(
        name="Electricity",
        df=df,
        target="target",
        domain_name="domain",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )

def get_free_light_chain_mortality_data(
        mode = "og",
        penalty = 0.5,
        model = "rbf" ,
        shift_col = "default",
        use_pca = False,
        n_components = 5        
):
    """
    @article{dispenzieri2012nonclonal,
        author = {Dispenzieri, Angela and Katzmann, Jerry and Kyle, Robert and Larson, Dirk and Therneau, Terry and Colby, Colin and Clark, Raynell and Mead, Graham and Kumar, Shaji and Melton, L and Rajkumar, S},
        year = {2012},
        month = {06},
        pages = {517-23},
        title = {Use of Nonclonal Serum Immunoglobulin Free Light Chains to Predict Overall Survival in the General Population},
        volume = {87},
        journal = {Mayo Clinic proceedings. Mayo Clinic},
        doi = {10.1016/j.mayocp.2012.03.009}
    }

    https://www.kaggle.com/datasets/nalkrolu/assay-of-serum-free-light-chain
    """
    data = pd.read_csv(
        os.path.join(MODULE_DIR, "data/free_light_chain_mortality"), sep=",", keep_default_na=False
    )
    

    # Drop the chapter since it leaks information whether the patient died or not,
    # which is the target variable
    data.drop(["chapter"], axis=1, inplace=True)

    # Drop the first column since it is just an index
    data.drop(["Unnamed: 0"], axis=1, inplace=True)

    # Convert categorical variables
    cat_columns = ["death", "mgus", "flc.grp", "sex"]
    data[cat_columns] = data[cat_columns].astype("category")

    # Drop rows that are nan since we are subsampling the data anyway
    data.dropna(axis=0, inplace=True)

    subsampled_dfs = []
    for dom in data["sample.yr"].unique():
        domain_instances = data[data["sample.yr"] == dom].shape[0]

        # Use stratified sampling to ensure that the distribution of the target is preserved
        subsampled_df = (
            data[data["sample.yr"] == dom]
            .groupby("death", observed=False)
            .apply(
                lambda x: x.reset_index().sample(
                    frac=min(1.0, 80 / domain_instances), random_state=42
                )
            )
            .droplevel(0)
        )

        # Restore the order relative to each other
        subsampled_df.sort_index(inplace=True)

        subsampled_dfs.append(subsampled_df)

    # concatenate all subsampled DataFrames
    data = pd.concat(subsampled_dfs)


    if mode == "og":
        # Restore the order relative to each other
        data.sort_values("sample.yr", inplace=True)
    else:
        if shift_col == "default":
            data = get_dynamic_shifts(
                data, 
                shift_col="sample.yr", 
                penalty=penalty, 
                method=mode, 
                model=model
            )
        elif shift_col == "all_numeric":
            data = get_dynamic_shifts_multifeature(
                data = data, 
                features = data.select_dtypes(include=['number']).columns.tolist(), 
                penalty = penalty, 
                method=mode, 
                model=model,
                use_pca=use_pca,
                n_components=n_components
            )

    return dataframe_to_distribution_shift_ds(
        "Free Light Chain Mortality",
        data,
        "death",
        "sample.yr",
        task_type=TASK_TYPE_MULTICLASS,
        dataset_source="real-world",
        shuffled=False,
    )


