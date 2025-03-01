import h2o
from h2o.automl import H2OAutoML
import pandas as pd


print("Initializing H2O...")
h2o.init()
print("Initializing complete")

print("Importing Wine dataset...")
df = h2o.import_file('data/winequality-white.csv')
print("Import complete")

print("Training AutoML...")
x = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", 'alcohol']
y = "quality"

df[y] = df[y].asfactor()

train, test = df.split_frame([0.8])

aml = H2OAutoML(max_models=10, seed=0)
aml.train(x=x, y=y, training_frame=train)
print("Training complete")

# print("\n### AutoML Explainability ###")
# exa = aml.explain(test)
# print(exa)
# print("\n### AutoML Leader Explainability ###")
# exa = aml.leader.explain(test)
# print(exa)
print("\n### AutoML Leaderboard ###")
lb = h2o.get_leaderboard(aml, extra_columns='ALL')
print(lb)

model_path = h2o.save_model(model=aml.leader, path='.', force=True)
print(f"\nModel saved to: {model_path}")