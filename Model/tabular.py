import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import os
import yaml

class TabularProcessor(torch.nn.Module):
    def __init__(self, config, max_categories=25):
        """
        Initializes the processor by:
        - Identifying numerical & categorical columns
        - Normalizing numeric features
        - Encoding categorical features (one-hot)
        - Limiting categorical values to the top `max_categories`
        - Keeping metadata for later reverse-mapping
        """
        super().__init__()
        self.max_categories = max_categories
        self.device = torch.device('cuda:0')

        with open("Data/raw_database/tabular_data.yaml", "r") as file:
            tabular_fields = yaml.safe_load(file)
        
        self.tabular_fields = tabular_fields['tabular_fields']

    def configure(self, df):
        self.tabular_df = df[self.tabular_fields]
        self.num_cols = self.tabular_df.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = self.tabular_df.select_dtypes(exclude=['number']).columns.tolist()

        # Fill missing values
        if self.num_cols:
            self.tabular_df[self.num_cols] = self.tabular_df[self.num_cols].fillna(self.tabular_df[self.num_cols].mean()).infer_objects(copy=False)
        if self.cat_cols:
            self.tabular_df[self.cat_cols] = self.tabular_df[self.cat_cols].fillna("Missing").infer_objects(copy=False)

        self.coverage, self.value_counts, self.impute_values, self.inclusion_df = {}, {}, {}, pd.DataFrame(index=self.tabular_df.index)

        for col in self.cat_cols:
            top_values = self.tabular_df[col].value_counts()
            self.value_counts[col] = top_values.index[:self.max_categories].tolist()
            self.coverage[col] = (top_values.iloc[:self.max_categories].sum() / top_values.sum()) * 100
            self.impute_values[col] = "Misc. (imputed)"  # Default imputation value
            self.inclusion_df[col + "_included"] = self.tabular_df[col].isin(self.value_counts[col]).astype(int)
            self.tabular_df[col] = self.tabular_df[col].apply(lambda x: x if x in self.value_counts[col] else self.impute_values[col])

        self.num_scaler = StandardScaler().fit(self.tabular_df[self.num_cols]) if self.num_cols else None
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(self.tabular_df[self.cat_cols]) if self.cat_cols else None

        self.feature_metadata = self.num_cols + (self.cat_encoder.get_feature_names_out(self.cat_cols).tolist() if self.cat_cols else [])

        for col, percent in self.coverage.items():
            print(f"Coverage for {col}: {percent:.2f}% retained.")

    def latent(self, data):
        """
        Transforms the input (NumPy array or DataFrame) into a normalized numerical tensor and stores metadata.
        
        Args:
            data (np.ndarray or pd.DataFrame): Input data with same columns as original DataFrame (or NumPy array with same order).
        
        Returns:
            - `tensor_output`: The processed tensor
            - `inclusion_mask`: Tracks which categorical values were included
            - `metadata`: Dictionary for reverse mapping field/category names
        """
        data = data[0]
        is_numpy = isinstance(data, np.ndarray)
        
        if is_numpy:
            df = pd.DataFrame(data, columns=self.tabular_df.columns)  # Convert to DataFrame for easier handling
        else:
            df = data.copy()

        features = []
        inclusion_mask = np.ones(df.shape[0], dtype=np.int32) if self.cat_cols else None

        # Ensure column order follows self.tabular_df.columns
        column_order = self.tabular_df.columns.tolist()
        
        # Process categorical features
        if self.cat_cols:
            cat_indices = [column_order.index(col) for col in self.cat_cols]
            inclusion_mask = np.ones((df.shape[0], len(self.cat_cols)), dtype=np.int32)

            for i, col in enumerate(self.cat_cols):
                df[col] = df[col].fillna("Missing")  # Correct approach
                valid_mask = df[col].isin(self.value_counts[col])
                inclusion_mask[:, i] = valid_mask.astype(np.int32)

                # Replace unseen categories with stored `impute_values`
                df[col] = df[col].where(valid_mask, self.impute_values[col])

            cat_features = self.cat_encoder.transform(df[self.cat_cols].astype(str))
            features.append(torch.tensor(cat_features, dtype=torch.float32))

        # Process numerical features
        if self.num_cols:
            num_indices = [column_order.index(col) for col in self.num_cols]

            # Ensure no new NaNs in input numerical data
            for col in self.num_cols:
                df[col] = df[col].fillna(self.tabular_df[col].mean())  # Correct approach

            num_features = self.num_scaler.transform(df[self.num_cols].astype(float))
            features.append(torch.tensor(num_features, dtype=torch.float32))

        # Concatenate final tensor
        output_tensor = torch.cat(features, dim=1) if len(features) > 1 else features[0]

        return output_tensor.to(self.device)


    def forward(self, x):
        return self.latent(x)

    def reverse_map(self, tensor_idx):
        """
        Given a tensor index (e.g., from an attention map), returns the corresponding field/category.
        """
        if 0 <= tensor_idx < len(self.feature_metadata):
            return self.feature_metadata[tensor_idx]
        raise Exception(f"Id {tensor_idx} not in tabular mapping")

import hydra
@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(config):
    # labs = pd.read_csv(os.path.join(config.ct_data_dir, '1/labs.csv')
    # encounters = pd.read_csv(os.path.join(config.ct_data_dir, '1/encounters.csv')
    # crosswalk = pd.read_csv('/dataNAS/data/ct_data/priority_crosswalk_all.csv')
    # radiology_report = pd.read_csv(os.path.join(config.ct_data_dir, '1/radiology_report.csv')
    # procedures = pd.read_csv(os.path.join(config.ct_data_dir, '1/procedures.csv')
    demographics = pd.read_csv(os.path.join(config.paths.ct_data_dir, '1/demographics.csv'))

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
    processor = TabularProcessor()
    processor.configure(df)

    my_df = df.iloc[0:10].to_numpy()
    transformed = processor.latent(my_df)
    f = processor.reverse_map(3)
    x = 3

if __name__ == "__main__":
    main()