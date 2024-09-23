import os
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycountry
import warnings
warnings.filterwarnings("ignore")


def load_ember_data():
    """
    Load the Ember data from a CSV file located in the data folder.

    Returns:
        pd.DataFrame: Ember data in long format.
    """
    ember_data_path = os.path.join(
        os.getcwd(), "data/ember_yearly_full_release_long_format.csv")
    ember_data = pd.read_csv(ember_data_path)
    return ember_data


def get_network_path(scenario_folder):
    """
    Get the full path to the PyPSA network file from the specified scenario folder.
    Assumes that only one network file exists in the folder.

    Args:
        scenario_folder (str): Folder containing the scenario data.

    Returns:
        str: Full path to the network file.

    Raises:
        AssertionError: If more than one network file exists in the folder.
    """
    foldername = f"{scenario_folder}/networks"
    results_dir = os.path.join(
        os.getcwd(), f"submodules/pypsa-earth/results/{scenario_folder}/networks")
    filenames = os.listdir(results_dir)

    # Ensure only one network file exists
    if len(filenames) == 1:
        print("Only 1 network per folder is allowed!")

    filepath = os.path.join(results_dir, filenames[0])
    return filepath


def create_results_dir():
    """
    Create a results directory if it does not already exist.
    This is where output data will be stored.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists("results/plots"):
        os.makedirs("results/plots")


def convert_two_country_code_to_three(country_code):
    """
    Convert a two-letter country code to a three-letter ISO country code.

    Args:
        country_code (str): Two-letter country code (ISO 3166-1 alpha-2).

    Returns:
        str: Three-letter country code (ISO 3166-1 alpha-3).
    """
    country = pycountry.countries.get(alpha_2=country_code)
    return country.alpha_3


def load_pypsa_network(scenario_folder):
    """
    Load a PyPSA network from a specific scenario folder.

    Args:
        scenario_folder (str): Folder containing the scenario data.

    Returns:
        pypsa.Network: The loaded PyPSA network object.
    """
    network_path = get_network_path(scenario_folder)
    network = pypsa.Network(network_path)
    return network


def get_demand_ember(data, country_code, horizon):
    """
    Get the electricity demand for a given country and horizon from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        country_code (str): Country code (ISO 3166-1 alpha-2).
        horizon (int): horizon of interest.

    Returns:
        float or None: Electricity demand if found, otherwise None.
    """
    demand = data[(data["Year"] == horizon)
                  & (data["Country code"] == country_code)
                  & (data["Category"] == "Electricity demand")
                  & (data["Subcategory"] == "Demand")]["Value"]

    if len(demand) != 0:
        return demand.iloc[0]
    return None


def get_demand_pypsa(network):
    """
    Get the total electricity demand from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        float: Total electricity demand in TWh.
    """
    demand_pypsa = network.loads_t.p_set.multiply(
        network.snapshot_weightings.objective, axis=0).sum().sum() / 1e6
    demand_pypsa = demand_pypsa.round(4)
    return demand_pypsa


def get_installed_capacity_ember(data, three_country_code, horizon):
    """
    Get installed capacity by fuel type for a given country and horizon from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        horizon (int): horizon of interest.

    Returns:
        pd.DataFrame: Installed capacity by fuel type.
    """
    capacity_ember = data[
        (data["Country code"] == three_country_code)
        & (data["Year"] == horizon)
        & (data["Category"] == "Capacity")
        & (data["Subcategory"] == "Fuel")][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    capacity_ember = capacity_ember[~capacity_ember["Variable"].isin(drop_row)]

    # Standardize fuel types
    capacity_ember = capacity_ember.replace({
        "Gas": "Fossil fuels",
        "Bioenergy": "Biomass",
        "Coal": "Fossil fuels",
        "Other Fossil": "Fossil fuels"})

    capacity_ember = capacity_ember.groupby("Variable").sum()
    capacity_ember.columns = ["Ember data"]

    return capacity_ember


def get_installed_capacity_pypsa(network):
    """
    Get installed capacity by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Installed capacity by fuel type.
    """
    gen_capacities = network.generators.groupby("carrier").p_nom.sum()
    storage_capacities = network.storage_units.groupby("carrier").p_nom.sum()
    capacity_pypsa = (
        pd.concat([gen_capacities, storage_capacities], axis=0) / 1e3).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT",
                    "OCGT", "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass"]

    # Reindex to include missing carriers
    capacity_pypsa = capacity_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    capacity_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass"}, inplace=True)

    # Aggregate fossil fuel and hydro capacities
    capacity_pypsa["Fossil fuels"] = capacity_pypsa[[
        "coal", "lignite", "CCGT", "OCGT"]].sum()
    capacity_pypsa["Hydro"] = capacity_pypsa[["hydro", "ror"]].sum()
    capacity_pypsa["Wind"] = capacity_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Filter and reformat
    capacity_pypsa = capacity_pypsa.loc[["Nuclear", "Fossil fuels", "Hydro",
                                         "PHS", "Solar", "Wind", "Biomass"]]
    capacity_pypsa.name = "PyPSA data"
    capacity_pypsa = capacity_pypsa.to_frame()

    return capacity_pypsa


def get_generation_capacity_ember(data, three_country_code, horizon):
    """
    Get electricity generation by fuel type for a given country and horizon from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        horizon (int): horizon of interest.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    generation_ember = data[
        (data["Category"] == "Electricity generation")
        & (data["Country code"] == three_country_code)
        & (data["Year"] == horizon)
        & (data["Subcategory"] == "Fuel")
        & (data["Unit"] == "TWh")
    ][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    generation_ember = generation_ember[~generation_ember["Variable"].isin(
        drop_row)]

    # Standardize fuel types
    generation_ember = generation_ember.replace({
        "Gas": "Fossil fuels",
        "Bioenergy": "Biomass",
        "Coal": "Fossil fuels",
        "Other Fossil": "Fossil fuels"})

    # Group by fuel type
    generation_ember = generation_ember.groupby("Variable").sum()
    generation_ember.loc["Load shedding"] = 0.0
    generation_ember.columns = ["Ember data"]

    return generation_ember


def get_generation_capacity_pypsa(network):
    """
    Get electricity generation by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    gen_capacities = (network.generators_t
                      .p.multiply(network.snapshot_weightings.objective, axis=0)
                      .groupby(network.generators.carrier, axis=1).sum().sum())

    storage_capacities = (network.storage_units_t
                          .p.multiply(network.snapshot_weightings.objective, axis=0)
                          .groupby(network.storage_units.carrier, axis=1).sum().sum())

    # Combine generator and storage generation capacities
    generation_pypsa = (
        (pd.concat([gen_capacities, storage_capacities], axis=0)) / 1e6).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT",
                    "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass", "load"]

    # Reindex to include missing carriers
    generation_pypsa = generation_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    generation_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass",
        "load": "Load shedding"}, inplace=True)

    # Aggregate fossil fuel, hydro, and wind generation
    generation_pypsa["Fossil fuels"] = generation_pypsa[[
        "CCGT", "OCGT", "coal", "lignite"]].sum()
    generation_pypsa["Hydro"] = generation_pypsa[["hydro", "ror"]].sum()
    generation_pypsa["Wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Adjust load shedding value
    generation_pypsa["Load shedding"] /= 1e3

    # Filter and reformat
    generation_pypsa = generation_pypsa.loc[["Nuclear", "Fossil fuels", "PHS",
                                             "Hydro", "Solar", "Wind", "Biomass",
                                             "Load shedding"]]
    generation_pypsa.name = "PyPSA data"
    generation_pypsa = generation_pypsa.to_frame()

    return generation_pypsa


def get_country_name(country_code):
    """ Input:
            country_code - two letter code of the country
        Output:
            country.name - corresponding name of the country
            country.alpha_3 - three letter code of the country
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name, country.alpha_3 if country else None
    except KeyError:
        return None


def get_data_EIA(data_path, country_code, horizon):
    """
    Retrieves energy generation data from the EIA dataset for a specified country and horizon.

    Args:
        data_path (str): Path to the EIA CSV file.
        country_code (str): Two-letter or three-letter country code (ISO).
        horizon (int or str): horizon for which energy data is requested.

    Returns:
        pd.DataFrame: DataFrame containing energy generation data for the given country and horizon, 
                    or None if no matching country is found.
    """

    # Load EIA data from CSV file
    data = pd.read_csv(data_path)

    # Rename the second column to 'country' for consistency
    data.rename(columns={"Unnamed: 1": "country"}, inplace=True)

    # Remove leading and trailing spaces in the 'country' column
    data["country"] = data["country"].str.strip()

    # Extract the three-letter country code from the 'API' column
    data["code_3"] = data.dropna(subset=["API"])["API"].apply(
        lambda x: x.split('-')[2] if isinstance(x,
                                                str) and len(x.split('-')) > 3 else x
    )

    # Get the official country name and three-letter country code using the provided two-letter code
    country_name, country_code3 = get_country_name(country_code)

    # Check if the three-letter country code exists in the dataset
    if country_code3 and country_code3 in data.code_3.unique():
        # Retrieve the generation data for the specified horizon
        result = data.query(
            "code_3 == @country_code3")[["country", str(horizon)]]

    # If not found by code, search by the country name
    elif country_name and country_name in data.country.unique():
        # Find the country index and retrieve generation data
        country_index = data.query("country == @country_name").index[0]
        result = data.iloc[country_index +
                           1:country_index+18][["country", str(horizon)]]

    else:
        # If no match is found, return None
        result = None

    # Convert the horizon column to float for numeric operations
    result[str(horizon)] = result[str(horizon)].astype(float)

    return result


def preprocess_eia_data(data):
    """
    Preprocesses the EIA energy data by renaming and filtering rows and columns.

    Args:
        data (pd.DataFrame): DataFrame containing EIA energy data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for analysis.
    """

    # Strip the last 13 characters (descriptive text) from the 'country' column
    data["country"] = data["country"].apply(lambda x: x[:-13].strip())

    # Set 'country' as the index of the DataFrame
    data.set_index("country", inplace=True)

    # Rename columns to provide clarity
    data.columns = ["EIA Data"]

    # Rename specific rows to match more standard terms
    data.rename(index={"Hydroelectricity": "Hydro",
                       "Biomass and waste": "Biomass",
                       "Hydroelectric pumped storage": "PHS"}, inplace=True)

    # Drop unwanted renewable energy categories
    data.drop(index=["Renewables", "Non-hydroelectric renewables",
                     "Geothermal", "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Fossil fuels",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass"], :]

    return data


def preprocess_eia_data_generation(data):
    """
    Preprocesses the EIA energy data by renaming and filtering rows and columns.

    Args:
        data (pd.DataFrame): DataFrame containing EIA energy data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for analysis.
    """

    # Strip the last 13 characters (descriptive text) from the 'country' column
    data["country"] = data["country"].apply(lambda x: x[:-13].strip())

    # Set 'country' as the index of the DataFrame
    data.set_index("country", inplace=True)

    # Rename columns to provide clarity
    data.columns = ["EIA Data"]

    # Rename specific rows to match more standard terms
    data.rename(index={"Hydroelectricity": "Hydro",
                       "Biomass and waste": "Biomass",
                       "Hydroelectric pumped storage": "PHS"}, inplace=True)

    # Drop unwanted renewable energy categories
    data.drop(index=["Fossil fuels", "Renewables", "Non-hydroelectric renewables",
                     "Geothermal", "Solar, tide, wave, fuel cell", "Tide and wave"], inplace=True)

    # Filter the DataFrame to only include relevant energy sources
    data = data.loc[["Nuclear", "Coal", "Natural gas", "Oil",
                     "Hydro", "PHS", "Solar", "Wind", "Biomass", ], :]
    return data


def get_generation_capacity_pypsa_detail(network):
    """
    Get electricity generation by fuel type from the PyPSA network.

    Args:
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    gen_capacities = (network.generators_t
                      .p.multiply(network.snapshot_weightings.objective, axis=0)
                      .groupby(network.generators.carrier, axis=1).sum().sum())

    storage_capacities = (network.storage_units_t
                          .p.multiply(network.snapshot_weightings.objective, axis=0)
                          .groupby(network.storage_units.carrier, axis=1).sum().sum())

    # Combine generator and storage generation capacities
    generation_pypsa = (
        (pd.concat([gen_capacities, storage_capacities], axis=0)) / 1e6).round(2)

    # Define all possible carriers
    all_carriers = ["nuclear", "coal", "lignite", "CCGT", "OCGT",
                    "hydro", "ror", "PHS", "solar", "offwind-ac",
                    "offwind-dc", "onwind", "biomass", "load"]

    # Reindex to include missing carriers
    generation_pypsa = generation_pypsa.reindex(all_carriers, fill_value=0)

    # Rename fuel types to match convention
    generation_pypsa.rename(index={
        "nuclear": "Nuclear",
        "solar": "Solar",
        "biomass": "Biomass",
        "load": "Load shedding"}, inplace=True)

    # Aggregate fossil fuel, hydro, and wind generation
    generation_pypsa["Natural gas"] = generation_pypsa[["CCGT", "OCGT"]].sum()
    generation_pypsa["Coal"] = generation_pypsa[["coal", "lignite"]].sum()
    generation_pypsa["Hydro"] = generation_pypsa[["hydro", "ror"]].sum()
    generation_pypsa["Wind"] = generation_pypsa[[
        "offwind-ac", "offwind-dc", "onwind"]].sum()

    # Adjust load shedding value
    generation_pypsa["Load shedding"] /= 1e3

    # Filter and reformat
    generation_pypsa = generation_pypsa.loc[["Nuclear", "Natural gas", "PHS", "Coal",
                                             "Hydro", "Solar", "Wind", "Biomass",
                                             "Load shedding"]]
    generation_pypsa.name = "PyPSA data"
    generation_pypsa = generation_pypsa.to_frame()
    return generation_pypsa


def get_generation_capacity_ember_detail(data, three_country_code, horizon):
    """
    Get electricity generation by fuel type for a given country and horizon from Ember data.

    Args:
        data (pd.DataFrame): Ember data.
        three_country_code (str): Country code (ISO 3166-1 alpha-3).
        horizon (int): horizon of interest.

    Returns:
        pd.DataFrame: Electricity generation by fuel type.
    """
    generation_ember = data[
        (data["Category"] == "Electricity generation")
        & (data["Country code"] == three_country_code)
        & (data["Year"] == horizon)
        & (data["Subcategory"] == "Fuel")
        & (data["Unit"] == "TWh")
    ][["Variable", "Value"]].reset_index(drop=True)

    # Drop irrelevant rows
    drop_row = ["Other Renewables"]
    generation_ember = generation_ember[~generation_ember["Variable"].isin(
        drop_row)]

    # Standardize fuel types
    generation_ember = generation_ember.replace({
        "Gas": "Natural gas",
        "Bioenergy": "Biomass",
        # "Coal": "Fossil fuels",
        # "Other Fossil": "Fossil fuels"
    })

    # Group by fuel type
    generation_ember = generation_ember.groupby("Variable").sum()
    generation_ember.loc["Load shedding"] = 0.0
    generation_ember.columns = ["Ember data"]

    return generation_ember
