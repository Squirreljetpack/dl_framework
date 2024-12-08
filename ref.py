import pandas as pd
import numpy as np
from numpy import mean
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    plot_confusion_matrix,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    LeaveOneOut,
    cross_validate,
    cross_val_score,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import seaborn as sns

# Loading Dataset
bioprint_df = pd.read_csv(
    "C:/Users/Shuyu/Desktop/20201229 Bioink Database/20210406/Final Database/20210429/Classification and Regression Database (617 instances) 20210429.csv"
)

# Setting references column in bioprint_df as the row indices
bioprint_df = bioprint_df.set_index(bioprint_df["Reference"])
bioprint_df = bioprint_df.drop(["Reference"], axis=1)

# Print the first 5 instances of data as well as general dataset array information and how many blank values there are per variable
# bioprint_df.head(5)
# bioprint_df.shape
bioprint_df.isna().sum()

# Data Preprocessing and Analysis
# Imputing mode temperatures
imputer_mode = SimpleImputer(
    missing_values=np.nan, strategy="most_frequent"
)  # imputing mode value into missing values for temperatures
bioprint_df.loc[:, ["Syringe_Temperature_(°C)", "Substrate_Temperature_(°C)"]] = (
    imputer_mode.fit_transform(
        bioprint_df.loc[:, ["Syringe_Temperature_(°C)", "Substrate_Temperature_(°C)"]]
    )
)

# Analyzing Numerical (Continuous) Data
# Dropping Variables and Instances
bioprint_df = bioprint_df.drop(
    ["Fiber_Diameter_(μm)"], axis=1
)  # drop for extrusion pressure dataset creation
bioprint_df = bioprint_df.drop(
    [
        "CaCl2_Conc_(mM)",
        "NaCl2_Conc_(mM)",
        "BaCl2_Conc_(mM)",
        "SrCl2_Conc_(mM)",
        "Physical_Crosslinking_Durantion_(s)",
        "Photocrosslinking_Duration_(s)",
    ],
    axis=1,
)  # drop these variables to create the extrusion pressure dataset from the cell viability dataset

# Variables where more than 50% of all instances have null values are dropped
bioprint_df = bioprint_df.dropna(
    axis=1, thresh=177
)  # Variables with more than 177 null instances are dropped

# Drop instances without cell viability values
bioprint_df = bioprint_df[bioprint_df["Viability_at_time_of_observation_(%)"].notna()]

# Drop nonprinting instances (instances where extrusion pressure is zero)
bioprint_df = bioprint_df.drop(
    bioprint_df[bioprint_df["Extrusion_Pressure (kPa)"] == 0].index
)
bioprint_df = bioprint_df[bioprint_df["Extrusion_Pressure (kPa)"].notna()]

bioprint_df.head(10)
bioprint_df.shape

# Feature Selection Through Correlation
corr = bioprint_df.corr()
# display(corr)
fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=0.1)
abs(bioprint_df.corr()["Viability_at_time_of_observation_(%)"])

# Create the independent variables (x) set and the dependent variable (y) set from the training dataset
bioprint_df = bioprint_df.drop(
    ["Final_PEGTA_Conc_(%w/v)", "Final_PEGMA_Conc_(%w/v)"], axis=1
)

# Imputing Values
bioprint_df.isna().sum()

# Imputation of numerical/continuous values databases
imputer_knn = KNNImputer(
    n_neighbors=30, weights="uniform"
)  # imputing mode value into missing values
bioprint_df.iloc[:, 0:22] = imputer_knn.fit_transform(
    bioprint_df.iloc[:, 0:22]
)  # used for extrusion pressure dataset preprocessing

# Imputation of categorical values in databases
bioprint_df = bioprint_df.fillna(
    bioprint_df["Conical_or_Straight_Nozzle"].value_counts().index[0]
)

bioprint_df.isna().sum()  # Check if the imputation code works by generating a list of the number of null values for each variable

# Drop categorical or numerical cell viability column depending on which type of prediction model is desired (regression versus classification)
bioprint_df = bioprint_df.drop(["Viability_at_time_of_observation_(%)"], axis=1)
bioprint_df = bioprint_df.drop(["Extrusion_Pressure (kPa)"], axis=1)

# Normalizing/Scaling and Encoding Continuous and Categorical Data
x = bioprint_df.drop("Acceptable_Pressure_(Yes/No)", axis=1)
y = bioprint_df["Acceptable_Pressure_(Yes/No)"].values

# Use MinMaxScaler() function to normalize input values for performance metric evaluation
x.iloc[:, 0:22] = MinMaxScaler().fit_transform(
    x.iloc[:, 0:22]
)  # Used for extrusion pressure dataset

# One-hot encoding for categorical variables
x_ohencoded = pd.get_dummies(
    x,
    columns=[
        "Cell_Culture_Medium_Used?",
        "DI_Water_Used?",
        "Precrosslinking_Solution_Used?",
        "Saline_Solution_Used?",
        "EtOH_Solution_Used?",
        "Photoinitiator_Used?",
        "Enzymatic_Crosslinker_Used?",
        "Matrigel_Used?",
        "Conical_or_Straight_Nozzle",
        "Primary/Not_Primary",
    ],
)

y_ohencoded = pd.get_dummies(y)
y_ohencoded.isna().sum()
x_ohencoded.shape

# Machine Learning Algorithms for Regression Modeling


# 1. Random Forest Regressor
def rfr_model_optimization(x, y):
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid={
            "max_depth": range(3, 7),
            "n_estimators": [10, 50, 100],
        },
        cv=10,
        scoring="r2",
        verbose=0,
        n_jobs=-1,
    )  # verbose controls how many messages are returned
    grid_result = gsc.fit(x, y)
    best_params = grid_result.best_params_
    rfr = RandomForestRegressor(
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        random_state=42,
        verbose=False,
    )
    scores = cross_val_score(rfr, x, y, cv=10, scoring="r2")
    return best_params, scores


rfr_model_optimization(x_ohencoded, y)

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=42
)

# Initialize RandomForestRegressor model
rfr = RandomForestRegressor(max_depth=6, random_state=42, n_estimators=10)
rfr.fit(x_train, y_train)
pred_rfr = rfr.predict(x_test)  # runs label prediction on the test set
rfr_score = rfr.score(
    x_test, y_test
)  # returns the coefficient of determination of the model
print(rfr_score)  # coefficient of determination scoring

# Feature importance ranking graph
features = x_train.columns
importances = rfr.feature_importances_
indices = np.argsort(importances)

num_features = 10
plt.barh(
    range(num_features), importances[indices[-num_features:]], color="b", align="center"
)
plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
plt.xlabel("Relative Importance")
plt.xlim(0, 0.6)
plt.show()


# Calculates for coefficient of determination (r2) and mean squared error values based on the number of cross-validation folds
def rfr_model():
    model = RandomForestRegressor(max_depth=6, random_state=42, n_estimators=10)
    return model


def rfr_model_performance(
    cv,
):  # cv is the cross-validation type ex: 10fold, loocv, stratified, etc
    model = rfr_model()
    scores = cross_val_score(model, x_ohencoded, y, scoring="r2", cv=cv, n_jobs=-1)
    return mean(scores), scores.min(), scores.max()


rfr_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

# Evaluate each k value
for k in rfr_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    k_mean, k_min, k_max = rfr_model_performance(cv)
    print("> folds=%d, r2=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# Line plot of k mean values with min/max error bars
plt.errorbar(
    rfr_folds,
    means,
    yerr=[mins, maxs],
    fmt="o",
    markersize=5,
    color="black",
    linewidth=3,
)
plt.title("Number of Cross Validation Folds vs R2", fontsize=20)
plt.xlabel("Folds tested on", fontsize=20)
plt.ylabel("R2", fontsize=20)
plt.rcParams["figure.figsize"] = (10, 7)
plt.show()

# 1. Linear Regression
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=42
)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)  # Runs label prediction on the test set
lr_score = lr.score(
    x_test, y_test
)  # Returns the coefficient of determination of the model
print(lr_score)  # Returns coefficient of determination (r2)


def lr_model():
    model = LinearRegression()
    return model


def lr_model_performance(
    cv,
):  # crossval is the cross-validation type ex: 10fold, loocv, stratified, etc
    model = lr_model()
    scores = cross_val_score(model, x_ohencoded, y, scoring="r2", cv=cv, n_jobs=-1)
    return np.mean(scores), scores.min(), scores.max()


# Define folds to test
lr_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

# Evaluate each k value
for k in lr_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)  # cv is the number of folds
    k_mean, k_min, k_max = lr_model_performance(cv)
    print("> folds=%d, r2=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# Line plot of k mean values with min/max error bars
plt.errorbar(
    lr_folds,
    means,
    yerr=[mins, maxs],
    fmt="o",
    markersize=5,
    color="black",
    linewidth=3,
)
plt.title("Number of Cross Validation Folds vs R2")
plt.xlabel("Folds tested on")
plt.ylabel("R2")
plt.rcParams["figure.figsize"] = (10, 7)
plt.show()

# 2. Support Vector Regression
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=1
)
svr = SVR(kernel="poly")
svr.fit(x_train, y_train)
pred_svr = svr.predict(x_test)  # Runs label prediction on the test set
svr_score = svr.score(
    x_test, y_test
)  # Returns the coefficient of determination of the model
print(svr_score)  # Coefficient of determination scoring


def svr_model():
    model = SVR(kernel="rbf")
    return model  # Model already defined


def evaluate_svr_model(
    cv,
):  # Crossval is the cross-validation type ex: 10fold, loocv, stratified, etc
    model = svr_model()
    scores = cross_val_score(
        model, x_ohencoded, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
    )
    return np.mean(scores), scores.min(), scores.max()


# Define folds to test
svr_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

# Evaluate each k value
for k in svr_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)  # cv is the number of folds
    k_mean, k_min, k_max = evaluate_svr_model(cv)
    print("> folds=%d, mse=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

# Line plot of k mean values with min/max error bars
plt.errorbar(
    svr_folds,
    means,
    yerr=[mins, maxs],
    fmt="o",
    markersize=5,
    color="black",
    linewidth=3,
)
plt.title("Number of Cross Validation Folds vs Mean Squared Error")
plt.xlabel("Folds tested on")
plt.ylabel("Mean Squared Error")
plt.rcParams["figure.figsize"] = (10, 7)
plt.show()


# 3. Random Forest Classifier
def rfc_model(x, y):
    gsc = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={"max_depth": range(3, 7), "n_estimators": [10, 50, 100, 1000]},
        cv=10,
        scoring="accuracy",
        verbose=0,
        n_jobs=-1,
    )
    grid_result = gsc.fit(x, y)
    best_params = grid_result.best_params_
    rfr = RandomForestClassifier(
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        random_state=42,
        verbose=False,
    )
    scores = cross_val_score(rfr, x, y, cv=10, scoring="accuracy")
    return best_params, scores


rfc_model(x_ohencoded, y)
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=42
)
rfc = RandomForestClassifier(max_depth=6, random_state=42, n_estimators=10)
rfc.fit(x_train, y_train)
pred_rfc = rfc.predict(x_test)
rfc_score = rfc.score(x_test, y_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features = x_train.columns
num_features = 10
plt.barh(
    range(num_features), importances[indices[-num_features:]], color="b", align="center"
)
plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
plt.xlabel("Relative Importance")
plt.xlim(0, 0.6)
plt.show()


def rfc_model():
    model = RandomForestClassifier(max_depth=6, random_state=42, n_estimators=10)
    return model


def rfc_model_performance(cv):
    model = rfc_model()
    scores = cross_val_score(
        model, x_ohencoded, y, scoring="accuracy", cv=cv, n_jobs=-1
    )
    return np.mean(scores), scores.min(), scores.max()


rfc_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

for k in rfc_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    k_mean, k_min, k_max = rfc_model_performance(cv)
    print("> folds=%d, accuracy=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

plt.errorbar(
    rfc_folds,
    means,
    yerr=[mins, maxs],
    fmt="o",
    markersize=20,
    color="black",
    linewidth=10,
)
plt.plot(
    rfc_folds, [1.0 for _ in range(len(rfc_folds))], color="r", label="Ideal accuracy"
)
plt.title("Fold vs Accuracy")
plt.xlabel("Fold tested on")
plt.ylabel("Accuracy score")
plt.legend(loc="upper left")
plt.rcParams["figure.figsize"] = (20, 20)
plt.show()

# 4. Logistic Regression
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=42
)
logr = LogisticRegression()
logr.fit(x_train, y_train)
pred_logr = logr.predict(x_test)
logr_score = logr.score(x_test, y_test)
print(classification_report(y_test, pred_logr))
print(confusion_matrix(y_test, pred_logr))


def lr_model():
    model = LogisticRegression()
    return model


def lr_model_performance(cv):
    model = lr_model()
    scores = cross_val_score(model, x_ohencoded, y, scoring="recall", cv=cv, n_jobs=-1)
    return np.mean(scores), scores.min(), scores.max()


lr_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

for k in lr_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    k_mean, k_min, k_max = lr_model_performance(cv)
    print("> folds=%d, recall=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

plt.errorbar(lr_folds, means, yerr=[mins, maxs], fmt="o", color="black")
plt.plot(
    lr_folds, [1.0 for _ in range(len(lr_folds))], color="r", label="Ideal accuracy"
)
plt.title("Fold vs Recall")
plt.xlabel("Fold tested on")
plt.ylabel("Accuracy score")
plt.legend(loc="upper left")
plt.rcParams["figure.figsize"] = (15, 10)
plt.show()

# 5. Support Vector Classification
x_train, x_test, y_train, y_test = train_test_split(
    x_ohencoded, y, test_size=0.1, random_state=42, shuffle=False
)
svc = SVC(kernel="rbf")
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
svc_score = svc.score(x_test, y_test)
print(classification_report(y_test, pred_svc))
print(confusion_matrix(y_test, pred_svc))

disp = plot_confusion_matrix(
    svc,
    x_test,
    y_test,
    display_labels=["FD out of tolerance", "FD within tolerance"],
    cmap=plt.cm.Blues,
)
plt.show()


def svc_model():
    model = SVC(kernel="rbf")
    return model


def svc_model_performance(cv):
    model = svc_model()
    scores = cross_val_score(model, x_ohencoded, y, scoring="recall", cv=cv, n_jobs=-1)
    return np.mean(scores), scores.min(), scores.max()


svc_folds = [2, 5, 10]
means, mins, maxs = list(), list(), list()

for k in svc_folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    k_mean, k_min, k_max = svc_model_performance(cv)
    print("> folds=%d, recall=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    mins.append(k_mean - k_min)
    maxs.append(k_max - k_mean)

plt.errorbar(svc_folds, means, yerr=[mins, maxs], fmt="o", color="black")
plt.plot(
    svc_folds, [1.0 for _ in range(len(svc_folds))], color="r", label="Ideal accuracy"
)
plt.title("Fold vs Accuracy")
plt.xlabel("Fold tested on")
plt.ylabel("Accuracy score")
plt.legend(loc="upper left")
plt.rcParams["figure.figsize"] = (15, 10)
plt.show()

# Generating Value Predictions
predict_df = pd.read_csv(
    "C:/Users/Shuyu/Desktop/20201229 Bioink Database/20210406/Final Database/Filament Diameter Prediction Set 340 Instances.csv"
)

svc.fit(x_ohencoded, y)
xnew = predict_df.drop(["Filament_Diameter_(µm)"], axis=1)
ynew = svc.predict(xnew)
xnew["Filament_Diameter_(µm)"] = ynew
export_df = pd.DataFrame(xnew)
export_df.to_csv(
    r"C:/Users/Shuyu/Desktop/export_dataframe_FD.csv", index=False, header=True
)
