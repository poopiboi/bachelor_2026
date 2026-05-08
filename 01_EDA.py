from pathlib import Path
import platform
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

## We want to map the countries from ACLED onto the iso names from the Gravity dataset. I made this stupid dictionary BY HAND, please respect that.
country_iso_dict = {
    "Algeria": "DZA", "Angola": "AGO", 'Benin': "BEN", 'Botswana': "BWA", 'Burkina Faso': "BFA", "Burundi": "BDI", 'Cameroon': "CMR", 'Cape Verde': "CPV",
    "Central African Republic": "CAF", 'Chad': "TCD", 'Comoros': "COM", 'Democratic Republic of Congo': "COD", 'Djibouti': "DJI", 'Egypt': "EGY",
    'Equatorial Guinea': "GNQ", 'Eritrea': "ERI", 'Ethiopia': "ETH", 'Gabon': "GAB",'Gambia': "GMB", 'Ghana': "GHA", 'Guinea': "GIN", 'Guinea-Bissau': "GNB", 
    'Ivory Coast': "CIV", 'Kenya': "KEN", 'Lesotho': "LSO", 'Liberia': "LBR", 'Libya': "LBY", 'Madagascar': "MDG", 'Malawi': "MWI",'Mali': "MLI", 
    'Mauritania': "MRT", 'Mauritius': "MUS", 'Mayotte': "MYT", 'Morocco': "MAR",'Mozambique': "MOZ", 'Namibia': "NAM", 'Niger': "NER", 'Nigeria': "NGA", 
    'Republic of Congo': "COG", 'Reunion': "REU", 'Rwanda': "RWA", 'Saint Helena, Ascension and Tristan da Cunha': "SHN", 'Sao Tome and Principe': "STP", 
    'Senegal': "SEN", 'Seychelles': "SYC", 'Sierra Leone': "SLE", 'Somalia': "SOM", 'South Africa': "ZAF", 'South Sudan': "SSD", 'Sudan': "SDN", 
    'Tanzania': "TZA", 'Togo': "TGO", 'Tunisia': "TUN", 'Uganda': "UGA", 'Zambia': "ZMB", 'Zimbabwe': "ZWE", 'eSwatini': "SWZ"
}

def dataframe_prep(ACLED, Gravity):
    '''
    Function for getting the folders for the datasets. These should all be downloaded, and are accessible here: https://drive.google.com/drive/folders/1QDy30BzTSirBh7ndpf0HgvnI_bYFeVLU?usp=sharing

    input:
        ACLED = str datapath for the folder ACLED_sets
        Gravity = str datapath for the folder Gravity_sets

    output:
        df_a = dataframe for ACLED
        df_g = dataframe for Gravity
    '''

    ACLED_path = Path(ACLED).as_posix()
    Gravity_path = Path(Gravity).as_posix()

    acled_af = f"{ACLED_path}/ACLEDAfricaData_1997_2026-02-02.csv"
    gravity = f"{Gravity_path}/Gravity_V202211.csv"

    df_a = pd.read_csv(acled_af)
    df_g = pd.read_csv(gravity)

    return df_a, df_g

def acled_cleaner(acled_df: pd.DataFrame, lag_time:int = 0):
    '''
    Function for cleaning up and preparing the acled dataset for the ECOWAS frame, choosing columns and adding prefixes to the values.

    input:
        acled_df = pandas DataFrame that contains all the acled data.
        lag_time = int defining number of lag time to add to the ACLED years.

    output:
        acled_result = pandas DataFrame for the resulting dataframe.
    '''


    #   inter1 is the perpetrator, inter2 is the target
    df = acled_df[["country", "year", "disorder_type", "event_type", "inter1", "inter2", "fatalities"]
    ].copy()

    # We save a list of all unique possible values for the columns for future use and reference.
    country_list = df["country"].sort_values().unique()

    disorder_types = df["disorder_type"].unique()
    event_types = df["event_type"].unique()
    attack_groups = df["inter1"].unique()
    target_groups = df["inter2"].unique()

    # We create dummy values for each type of disorder, event, attackers and target
    dummies = pd.get_dummies(
        df[['disorder_type', 'event_type', 'inter1', 'inter2']],
        prefix=['disorder', 'event', 'perpetrator', 'target']
    )

    # We add the numeric columns back to the dummy dataset
    dummies['fatalities'] = df['fatalities']
    dummies['country'] = df['country']
    dummies['year'] = df['year']

    # Now we can group by country and year, and sum over the dummy categories. Perfect!
    acled_result = dummies.groupby(['country', 'year']).sum().reset_index()



    # We update the dataframe to have a new column for the iso-tags.
    acled_result["iso"] = acled_result["country"].map(country_iso_dict)

    # It is really just this simple!
    acled_result["year"] = acled_result["year"]+lag_time

    return acled_result


def gravity_cleaner(gravity_df:pd.DataFrame, narrow_columns: bool = True):
    '''
    Function that cleans up and prepares the gravity dataset. Columns can be defined narrowly or wide, depending on the needed amount of data.

    input:
        gravity_df = pandas DataFrame that contains the gravity dataset
        narrow_columns = bool for designating whether to keep only the few relevant columns (year, iso3_o/d, distw_arithmetic, contig, comlang_off, comlang_ethnic, gdp_d/o and the trade flows)

    output:
        df_g_filter = pandas Dataframe with implemented "dyad" column and a new combined_trade_baci.
    '''
    if narrow_columns:
        df_g = gravity_df[["year", "iso3_o", "iso3_d", "country_exists_o", "country_exists_d", "distw_arithmetic", "contig","comlang_off", "comlang_ethno", "gdp_o", "gdp_d", 
                            "tradeflow_baci", "manuf_tradeflow_baci", "tradeflow_comtrade_o", "tradeflow_comtrade_d", 
                            "tradeflow_imf_o", "tradeflow_imf_d"]]
    else:
        df_g = gravity_df[["year", "iso3_o", "iso3_d", "country_exists_o", "country_exists_d", "distw_harmonic", "distw_arithmetic", "dist", 
                            "distcap", "contig", "diplo_disagreement",  "comlang_off", "comlang_ethno", "comcol", "comleg_posttrans", "comrelig", 
                            "heg_o", "heg_d", "col_dep_ever", "col_dep",  "sibling_ever", "sibling", "sever_year", "pop_o", "pop_d", "gdp_o", "gdp_d", 
                            "gdpcap_ppp_o", "gdpcap_ppp_d",  "tradeflow_baci", "manuf_tradeflow_baci", "tradeflow_comtrade_o", "tradeflow_comtrade_d", 
                            "tradeflow_imf_o", "tradeflow_imf_d"]]
        
    df_g_filter = df_g[
        (df_g["year"] >= 1997) &
        (df_g["country_exists_o"] == 1) &
        (df_g["country_exists_d"] == 1) &
        (df_g["iso3_o"] != df_g["iso3_d"]) &
        (
            (df_g["iso3_o"].isin(country_iso_dict.values())) |
            (df_g["iso3_d"].isin(country_iso_dict.values()))
        )]
    

    ## NOTE: The tradeflow_baci in Gravity is only one way (from _o to _d), so to plot out the trade relations, we can create a new column
    # First we need to create a shared dyad key between two pairs (that is sorted, so it is always the same string)
    df_g_filter["dyad"] = df_g_filter.apply(
        lambda row: "_".join(sorted([row["iso3_o"], row["iso3_d"]])),
        axis=1
    )

    # Then we can aggregate total trade per dyad for each year
    combined = (
        df_g_filter.groupby(["dyad", "year"])["tradeflow_baci"]
        .sum()
        .reset_index()
        .rename(columns={"tradeflow_baci": "combined_trade_baci"})
    )

    # And we tack this onto the end of our dataframe. Our new column "combined_trade" can now be used
    df_g_filter = df_g_filter.merge(combined, on=["dyad", "year"], how="left")

    return df_g_filter


def ecowas_dataframe_merger(acled_df: pd.DataFrame, gravity_df: pd.DataFrame, printing_state:bool = True):
    '''
    A merger for the dataframes created for ACLED and Gravity for ECOWAS countries.

    input:
        acled_df = pandas DataFrame for the ACLED dataset (created by acled_cleaner)
        gravity_df = pandas DataFrame for the Gravity dataset (created by gravity_cleaner)
        printing_state = bool to print stats on the resulting DataFrame

    output:
        df_merged = pandas DataFrame for the final merged dataframe
    '''
    df_merged = pd.merge(
        acled_df, 
        gravity_df, 
        left_on=['iso', 'year'], 
        right_on=['iso3_o', 'year'], 
        how='inner'
    )
    if printing_state:
        print(f"Raw ACLED shape: {acled_df.shape}")
        print(f"Gravity filter shape: {gravity_df.shape}")
        print(f"Merged master shape: {df_merged.shape}")

    df_merged = df_merged.drop(columns=['iso', 'country_exists_o', 'country_exists_d'])


    # Keeping this beautiful code, because just look at it (It used to make sense, okay!! (when we had BACI integration))
    gambia_neighbors_ecowas = ["SEN", "BEN", "GHA", "GIN", "CIV", "NGA", "SLE", "TGO"]
    gambia_ecowas_previous_members = ["MLI", "BFA", "NER"]
    other_ecowas = ["LBR", "GNB"] # CPV is excluded, because of lacking Gravity data from before 2020
    whoops_forgot_gambia = ["GMB"]   # Please laugh
    combined_ecowas = set(gambia_neighbors_ecowas) | set(gambia_ecowas_previous_members) | set(other_ecowas) | set(whoops_forgot_gambia)



    ecowas_df_full = df_merged[
        (df_merged["iso3_o"].isin(combined_ecowas) & df_merged["iso3_d"].isin(combined_ecowas))
    ]
    if printing_state:
        print(f"Final dataframe shape: {ecowas_df_full.shape}")
    return ecowas_df_full


def check_nan(df:pd.DataFrame, print_state:bool = True):
    '''
    Small function for checking the number of missing values in the resulting dataframe.

    input:
        df = pandas DataFrame

    output:
        na_counts = a count of the missing values, sorted from highest to lowest
    '''
    # To check the NaN counts in our data   (See notes_for_work.md 04/03 entry to find reasons for removing columns)
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    
    if print_state:
        print(na_counts)
    return na_counts


def run_all(printing_state:bool = True):
    '''
    Function for running the whole thing.
    '''
    if platform.system() == "Windows":
        ACLED_folder_path_init = r"C:\Users\mhm25\Desktop\ITU\6thSemester\bachelorproj\data\acled_sets"
        Gravity_folder_path_init = r"C:\Users\mhm25\Desktop\ITU\6thSemester\bachelorproj\data\gravity_sets"

    else:
        ACLED_folder_path_init = r"/Users/zenrehda/Desktop/bachelor_2026/mydata/ACLED_sets"
        Gravity_folder_path_init = r"/Users/zenrehda/Desktop/bachelor_2026/mydata/gravity_sets"

    df_a, df_g = dataframe_prep(ACLED_folder_path_init, Gravity_folder_path_init)
    df_a_nil = acled_cleaner(df_a, 0)
    df_g = gravity_cleaner(df_g)
    comb_df = ecowas_dataframe_merger(df_a_nil, df_g, printing_state)


    comb_df.to_csv("data/ecowas_df_full_lag_zero.csv")

    df_a_one = acled_cleaner(df_a, 1)
    comb_df_one = ecowas_dataframe_merger(df_a_one, df_g, printing_state)
    comb_df_one.to_csv("data/ecowas_df_full_lag_one.csv")

    df_a_two = acled_cleaner(df_a, 2)
    comb_df_two = ecowas_dataframe_merger(df_a_two, df_g, printing_state)
    comb_df_two.to_csv("data/ecowas_df_full_lag_two.csv")
    
def main():
    run_all(False)

if __name__ == "__main__":
    run_all(False)
