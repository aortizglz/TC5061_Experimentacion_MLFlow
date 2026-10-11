import os
import warnings
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def load_and_preprocess_data(ratings_path: str, places_path:str):
    #carga
    df_ratings = pd.read_csv(ratings_path)
    df_places = pd.read_csv(places_path, encoding="latin-1")
    
    df = pd.merge(df_ratings, df_places, on="placeID",how="inner")
    df = df.drop_duplicates()
    #limpieza
    features = ["food_rating", "service_rating", "alcohol", "dress_code", "price"]
    target = "rating"
    df_clean = df[features + [target]].copy()
    df_clean = df_clean.replace("?", np.nan)
    df_clean = df_clean.dropna()
    #outliers
    vrange = ((df_clean["food_rating"]>=0)&(df_clean["food_rating"]<=2)&(df_clean["service_rating"]>=0)&(df_clean["service_rating"]>=0))
    df_clean = df_clean[vrange]
    #OHE
    df_clean=pd.get_dummies(df_clean, columns=["alcohol", "dress_code", "price"], drop_firts= True)
    X=df_clean.drop(columns=[target])
    y=df_clean[target]

    return X,y

def evaluate_model(model, X_test, y_test, output_fig_path="confusion_matrix.png"):
    y_pred=model.predict(X_test)
    #métricas diferenciadas
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_fig_path)
    plt.close()

    return macro_f1, weighted_f1, classification_report(y_test, y_pred)

def run_experiment(args):
    np.random.seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    X, y = load_and_preprocess_data(args.ratings_path, args.places_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    run_name = f"LR_C{args.c_param}_solver_{args.solver}_seed{args.seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", args.c_param)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.seed)
        mlflow.log_param("test_size", args.test_size)

        model = LogisticRegression(
            C=args.c_param,
            solver=args.solver,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        model.fit(X_train, y_train)
        fig_path = "confusion_matrix.png"
        macro_f1, weighted_f1, report = evaluate_model(model, X_test, y_test, fig_path)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_f1", weighted_f1)

        if os.path.exists(fig_path):
            mlflow.log_artifact(fig_path)
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        print(f"Macro F1-Score:    {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(report)

# pa que el script pueda recibir argumentos 
def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Entrenamiento parametrizado con MLFlow Tracking"
  )
  parser.add_argument(
      "--c_param",
      type=float,
      default=1.0,
      help="Parámetro inverso de regularización C de la Regresión Lógica",
  )
  parser.add_argument(
      "--solver",
      type=str,
      default="lbfgs",
      choices=["lbfgs", "liblinear", "saga", "newton-cg"],
      help="Algoritmo de optimización",
  )
  parser.add_argument(
      "--max_iter",
      type=int,
      default=500,
      help="Número máximo de iteraciones para que la operación converja",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Para reproducibilidad",
  )
  parser.add_argument(
      "--test_size",
      type=float,
      default=0.2,
      help="Fracción de datos que van para la prueba",
  )
  parser.add_argument(
      "--ratings_path",
      type=str,
      default="data/rating_final.csv",
      help="Ruta al archivo de rating d los restaurantes",
  )
  parser.add_argument(
      "--places_path",
      type=str,
      default="data/geoplaces2.csv",
      help="Ruta al archivo de geoplaces",
  )
  parser.add_argument(
      "--tracking_uri",
      type=str,
      default="http://localhost:5001",
      help="URI del servidor de MLflow",
  )
  parser.add_argument(
      "--experiment_name",
      type=str,
      default="TC5061_Model_Experimentation",
      help="Nombre del experimento en MLflow",
  )
  return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)