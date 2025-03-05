import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
from util.plotting import savefig

def plot_demographics(config, df):

    # Assuming df is your DataFrame containing the data
    total_scans = len(df)

    # Gender distribution pie chart
    gender_counts = df['Gender'].value_counts()

    plt.figure(figsize=(5, 5))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=140)
    plt.title(f'Gender Distribution of CT Scans (Total Scans: {total_scans})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    save_path = os.path.join(config.base_dir, 'Data/figures/')
    os.makedirs(save_path, exist_ok=True)
    savefig(os.path.join(save_path, 'patient_sex.jpg'))

    # Assuming df is your DataFrame containing the data
    total_scans = len(df)

    # Gender distribution pie chart
    gender_counts = df['Pat Class'].value_counts()[0:4]

    plt.figure(figsize=(5, 5))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Patient Class Distribution of CT Scans (Total Scans: {total_scans})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    savefig(os.path.join(save_path, 'patient_class.jpg'))

    # Assuming df is your DataFrame containing the data
    total_scans = len(df)

    # Gender distribution pie chart
    gender_counts = df['Race'].value_counts()[0:5]

    plt.figure(figsize=(5, 5))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Patient Race Distribution of CT Scans (Total Scans: {total_scans})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    savefig(os.path.join(save_path, 'patient_race.jpg'))

    # Age distribution using patient_age column
    plt.figure(figsize=(16, 4))
    df['patient_age'].plot(kind='hist', bins=20, color='coral', edgecolor='black')
    plt.title(f'Age Distribtuion (at the Time of Scan) (Total Scans: {total_scans})')
    plt.xlabel('Age')
    plt.ylabel('Number of Scans')

    savefig(os.path.join(save_path, 'patient_age.jpg'))


    # Calculate the IQR to filter out outliers
    Q1 = df['Recent BMI'].quantile(0.25)
    Q3 = df['Recent BMI'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the BMI data to remove outliers
    df_filtered = df[(df['Recent BMI'] >= lower_bound) & (df['Recent BMI'] <= upper_bound)]

    # Box plot for BMI distribution without outliers
    plt.figure(figsize=(8, 6))
    df_filtered.boxplot(column='Recent BMI')
    plt.title(f'BMI Distribution of CT Scans (Total Scans: {total_scans})')
    plt.ylabel('BMI')
    savefig(os.path.join(save_path, 'patient_BMI.jpg'))


    # Calculate the IQR to filter out outliers
    Q1 = df['Recent Height cm'].quantile(0.25)
    Q3 = df['Recent Height cm'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 5 * IQR
    upper_bound = Q3 + 5 * IQR

    # Filter the BMI data to remove outliers
    df_filtered = df[(df['Recent Height cm'] >= lower_bound) & (df['Recent Height cm'] <= upper_bound)]

    # Box plot for heigh distribution
    plt.figure(figsize=(8, 6))
    df_filtered.boxplot(column='Recent Height cm')
    plt.title(f'Height Distribution of CT Scans (Total Scans: {total_scans})')
    plt.ylabel('Height (CM)')
    savefig(os.path.join(save_path, 'patient_height.jpg'))

def get_tabular_data(config):
    path = "/dataNAS/data/ct_data/ct_ehr/"
    df_crosswalk = pd.read_csv("/dataNAS/data/ct_data/priority_crosswalk_all.csv")
    folders = sorted(os.listdir("/dataNAS/data/ct_data/ct_ehr/"))

    dfs = []
    for folder in folders:
        dfs.append(pd.read_csv(os.path.join(path, folder, "radiology_report_meta.csv"), low_memory=False))

    df_radiology = pd.concat(dfs)

    dfs = []
    for folder in folders:
        dfs.append(pd.read_csv(os.path.join(path, folder, "demographics.csv"), low_memory=False))

    df_dem = pd.concat(dfs)
    df_dem = df_dem.drop(["Disposition", "Notes", "Tags", "Interpreter Needed", "Insurance Name", "Insurance Type", "Recent Encounter Date", "Zipcode", "Death Date SSA Do Not Disclose", "Comment"], axis=1)
    df_dem = df_dem.rename(columns={'Patient Id': 'patient_id'})

    dfs = []
    for folder in folders:
        dfs.append(pd.read_csv(os.path.join(path, folder, "adt.csv"), low_memory=False))

    df_adt = pd.concat(dfs)
    df_adt = df_adt[["Patient Id", "PatEncCsnId", "Pat Class", "Pat Service"]].rename(columns={'Patient Id': 'patient_id', 'PatEncCsnId': 'patient_csn'})
    df_adt = df_adt.drop_duplicates(subset=['patient_id', 'patient_csn']).reset_index(drop=True)

    # Step 1: Drop rows where either accession_number or patient_id is NA
    df_clean = df_radiology.dropna(subset=['Accession Number', 'Patient Id'])

    # Step 2: Remove duplicate pairs
    df_clean = df_clean.drop_duplicates(subset=['Accession Number', 'Patient Id'])
    df_crosswalk["accession"] = df_crosswalk["accession"].astype(str)
    df_crosswalk = df_crosswalk.set_index("accession").join(df_clean.set_index("Accession Number")).dropna(subset="Patient Id").reset_index()
    df_crosswalk = df_crosswalk.reset_index()

    # Retain anon_accession here
    df_crosswalk = df_crosswalk[["accession", "anon_accession", "filename", "Patient Id", "PatEncCsnId", "Age"]].rename(columns={'Patient Id': 'patient_id', 'PatEncCsnId': 'patient_csn', 'Age': 'patient_age'})
    df_crosswalk = pd.merge(df_crosswalk[["accession", "anon_accession", "filename", "patient_id", "patient_csn", "patient_age"]], df_adt, how='left', on=['patient_id', 'patient_csn'])
    df_crosswalk = pd.merge(df_crosswalk, df_dem, how='left', on=['patient_id'])

    df = df_crosswalk.copy().reset_index(drop=True)

    return df

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    df = get_tabular_data(config)
    plot_demographics(config, df)

if __name__ == "__main__":
    main()