import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import os

class TabularProcessor:
    def __init__(self, df: pd.DataFrame, max_categories=25):
        """
        Initializes the processor by:
        - Identifying numerical & categorical columns
        - Normalizing numeric features
        - Encoding categorical features (one-hot)
        - Limiting categorical values to the top `max_categories`
        - Keeping metadata for later reverse-mapping
        """
        self.df = df
        self.num_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        self.max_categories = max_categories
        self.coverage = {}  # Track category retention %
        self.value_counts = {}  # Store top category mappings
        self.impute_values = {}  # Store most frequent category per field
        self.feature_metadata = []  # Store field names for later reconstruction

        # Inclusion tracking (was the original value in the top `max_categories`?)
        self.inclusion_df = pd.DataFrame(index=df.index)

        # Process categorical columns
        for col in self.cat_cols:
            top_values = df[col].value_counts()
            self.value_counts[col] = top_values.index[:max_categories].tolist()

            # Compute retention percentage
            retained_count = top_values.iloc[:max_categories].sum()
            total_count = top_values.sum()
            self.coverage[col] = (retained_count / total_count) * 100

            # Most frequent category for imputation
            self.impute_values[col] = self.value_counts[col][0] if self.value_counts[col] else None

            # Track inclusion (1 = retained, 0 = replaced)
            self.inclusion_df[col + "_included"] = df[col].apply(lambda x: 1 if x in self.value_counts[col] else 0)

            # Replace rare categories with most frequent category
            df[col] = df[col].apply(lambda x: x if x in self.value_counts[col] else self.impute_values[col])

        # Initialize preprocessors
        self.num_scaler = StandardScaler() if self.num_cols else None
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore") if self.cat_cols else None

        # Fit preprocessors
        if self.num_cols:
            self.num_scaler.fit(df[self.num_cols])
        if self.cat_cols:
            self.cat_encoder.fit(df[self.cat_cols])

        # Generate feature metadata
        self.feature_metadata.extend(self.num_cols)  # Numeric column names
        if self.cat_cols:
            cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols)
            self.feature_metadata.extend(cat_feature_names)  # One-hot encoded names

        # Print field coverage
        for col, percent in self.coverage.items():
            print(f"Coverage for {col}: {percent:.2f}% retained.")

    def latent(self, df: pd.DataFrame):
        """
        Transforms the input DataFrame into a normalized numerical tensor and stores metadata.
        Returns:
            - `tensor_output`: The processed tensor
            - `inclusion_df`: Tracks which categorical values were included
            - `metadata`: Dictionary for reverse mapping field/category names
        """
        features = []
        inclusion_df = pd.DataFrame(index=df.index)

        # Process numeric features
        if self.num_cols:
            num_features = self.num_scaler.transform(df[self.num_cols])
            features.append(torch.tensor(num_features, dtype=torch.float32))

        # Process categorical features
        if self.cat_cols:
            for col in self.cat_cols:
                inclusion_df[col + "_included"] = df[col].apply(lambda x: 1 if x in self.value_counts[col] else 0)
                df[col] = df[col].apply(lambda x: x if x in self.value_counts[col] else self.impute_values[col])

            cat_features = self.cat_encoder.transform(df[self.cat_cols])
            features.append(torch.tensor(cat_features, dtype=torch.float32))

        # Concatenate final tensor
        output_tensor = torch.cat(features, dim=1) if len(features) > 1 else features[0]

        # Prepare metadata for reverse mapping
        metadata = {
            "feature_names": self.feature_metadata,  # Field names in tensor order
            "categorical_mappings": self.value_counts,  # Categorical mappings
        }

        return output_tensor, inclusion_df, metadata

    def reverse_map(self, tensor_idx):
        """
        Given a tensor index (e.g., from an attention map), returns the corresponding field/category.
        """
        if 0 <= tensor_idx < len(self.feature_metadata):
            return self.feature_metadata[tensor_idx]
        return None

import hydra
@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(config):
    # labs = pd.read_csv(os.path.join(config.ct_data_dir, '1/labs.csv')
    # encounters = pd.read_csv(os.path.join(config.ct_data_dir, '1/encounters.csv')
    # crosswalk = pd.read_csv('/dataNAS/data/ct_data/priority_crosswalk_all.csv')
    # radiology_report = pd.read_csv(os.path.join(config.ct_data_dir, '1/radiology_report.csv')
    # procedures = pd.read_csv(os.path.join(config.ct_data_dir, '1/procedures.csv')
    demographics = pd.read_csv(os.path.join(config.ct_data_dir, '1/demographics.csv'))

    # Sample dataframe
    # df = pd.DataFrame({
    #     'age': [25, 30, 35, 40, 45],  # Continuous
    #     'gender': ['M', 'F', 'F', 'M', 'M'],  # Categorical
    #     'diabetic': [1, 0, 0, 1, 1],  # Binary
    #     'blood_pressure': [120, 130, 125, 140, 135],  # Treated as continuous
    #     'hospital_grade': ['A', 'B', 'C', 'A', 'B'],  # Categorical
    #     'readmission_risk': [0, 1, 0, 1, 1]  # Target
    # })
    fields = ['Gender', 'Race', 'Ethnicity', 'Disposition', 'Marital Status', 'Recent Height cm', 'Recent Weight kg', 'Recent BMI', 'Smoking Hx', 'Alcohol Use']

    df = demographics[fields]
    processor = TabularProcessor(df)

    my_df = df.iloc[0:10]
    transformed = processor.latent(my_df)
    f = processor.reverse_map(3)
    x = 3


if __name__ == "__main__":
    main()