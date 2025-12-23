import os
import sys
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    base_dir = "Exam_Score_preprocessing"
    
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    if not os.path.exists(train_path):
        print(f"Error: Dataset tidak ditemukan di '{base_dir}'.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["exam_score"])
    y_train = train_df["exam_score"]
    
    X_test  = test_df.drop(columns=["exam_score"])
    y_test  = test_df["exam_score"]

    # Nama eksperimen disamakan dengan main.yml
    #mlflow.set_experiment("Advance_Exam_Score_Tuning")

    print("Training model...")

    # training 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi 
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)
    rmse = mse ** 0.5 

    print(f"Done.")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    # LOGGING 
    mlflow.log_params({"n_estimators": 100, "random_state": 42})
    mlflow.log_metrics({
        "mse": mse, 
        "mae": mae, 
        "r2_score": r2, 
        "rmse": rmse
    })

    signature = infer_signature(X_test, y_pred)
    #input_example = X_test.iloc[0:1]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[0:1]
    )

    print("Success!")