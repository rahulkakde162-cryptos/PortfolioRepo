# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
#import mlflow

#mlflow.set_tracking_uri("file:./mlruns")
#mlflow.set_experiment("mlops-training-experiment")

import os

token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("HF_TOKEN environment variable not found.")

login(token=token)
api = HfApi(token=token)


from huggingface_hub import hf_hub_download

repo_id = "RahulKakde/PortfolioDataset"

Xtrain = pd.read_csv(
    hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
)

Xtest = pd.read_csv(
    hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
)

ytrain = pd.read_csv(
    hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
).squeeze()

ytest = pd.read_csv(
    hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")
).squeeze()


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
    ]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

numeric_transformer = make_pipeline(
    StandardScaler()
)

categorical_transformer = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

# Use ColumnTransformer directly, which takes 'transformers' as a list of tuples
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

# Create pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


    # Train model
model_pipeline.fit(Xtrain, ytrain)

    # Best model
best_model = model_pipeline

classification_threshold = 0.45

    # Predictions
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )
