import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def run_pipeline(input_csv):
    """
    This function runs the entire ETL pipeline:
    1. Extract raw data from CSV
    2. Transform data: handle missing values, encode categorical, scale numerical
    3. Load processed features and target to CSV files
    """

    # Step 1: Extract - Load dataset into a DataFrame
    df = pd.read_csv(input_csv)
    print("Raw data loaded successfully.")

    # Step 2: Separate features and target
    X = df.drop("Survived", axis=1)  # Features (all columns except target)
    y = df["Survived"]               # Target variable (Survival status)

    # Step 3: Define which columns are numerical and categorical
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    # Step 4: Build transformers for numerical and categorical columns

    # Numerical transformer: fill missing values with median, then scale features
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical transformer: fill missing values with 'missing', then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Step 5: Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    # Step 6: Apply transformations and fit on data
    X_processed = preprocessor.fit_transform(X)

    # Step 7: Convert the processed features back to a DataFrame for saving
    # Get one-hot encoded feature names
    cat_columns = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)
    all_columns = numeric_features + list(cat_columns)
    X_processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, columns=all_columns)

    # Step 8: Prepare output directory
    output_dir = "processed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 9: Save the processed features and target to CSV files
    features_path = os.path.join(output_dir, "features.csv")
    target_path = os.path.join(output_dir, "target.csv")

    X_processed_df.to_csv(features_path, index=False)
    y.to_csv(target_path, index=False)

    print(f"Processed features saved to {features_path}")
    print(f"Target saved to {target_path}")

if __name__ == "__main__":
    # Run the pipeline on the sample Titanic dataset
    run_pipeline("data/titanic.csv")
