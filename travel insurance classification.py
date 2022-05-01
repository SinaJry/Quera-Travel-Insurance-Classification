import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from matplotlib import pyplot as plt
import joblib
from xgboost import XGBClassifier

def read_data(path,print_desc):
    df = pd.read_csv(path)
    if print_desc :
        print('Data Shape : {}'.format(df.shape))
        print("Data contains of :")
        print(df.info())
        print("*"*40)
        print('Total Nulls : {}'.format(df.isnull().sum()))
        print("*"*40)
        print('Total Duplicates : {}'.format(df.duplicated().sum()))

    return df
   

path = r'path\travel_insurance\train.csv'

df = read_data(path,False)    

def clean_data(df,cols_to_drop,cat_cols):
    y_n_cat = {"No":0, "Yes":1}
    o_cat = {'Private Sector/Self Employed':0 , 'Government Sector':1}
    
    for i in cat_cols:
        if i == 'Employment Type':
            df[i] = df[i].map(o_cat)
        else:
            df[i] = df[i].map(y_n_cat)
    
    df = df.drop(columns = cols_to_drop)

    return df
    
cat_cols = ['EverTravelledAbroad','FrequentFlyer', 'GraduateOrNot' , 'Employment Type','TravelInsurance']
cols_to_drop = ['Customer Id']

df = clean_data(df,cols_to_drop ,cat_cols)

print(sorted(list(pd.qcut(df.Age , 2).unique())))

### Winsorizing numerical data

df.loc[(df.Age >= 24) & (df.Age < 30) , "Age"] = 0
df.loc[df.Age >= 30 ,"Age"] = 1


print(sorted(list(pd.cut(df.FamilyMembers , 3).unique())))

df.loc[(df.FamilyMembers >= 1) & (df.FamilyMembers < 4) , "FamilyMembers"] = 1
df.loc[(df.FamilyMembers >= 4) & (df.FamilyMembers < 6) , "FamilyMembers"] = 2
df.loc[(df.FamilyMembers >= 6)  , "FamilyMembers"] = 3


print(sorted(list(pd.cut(df.AnnualIncome , 3).unique())))

df.loc[(df.AnnualIncome >= 298500) & (df.AnnualIncome < 800000) , "AnnualIncome"] = 1
df.loc[(df.AnnualIncome >= 800000) & (df.AnnualIncome < 1300000) , "AnnualIncome"] = 2
df.loc[(df.AnnualIncome >= 1300000) , "AnnualIncome"] = 3

print(df.shape)
print('\n')
print('Columns :{}'.format(df.columns.tolist()))



def train_test(df,y,test_size):
    
    Y = df.loc[: , y]
    X = df.loc[:,~df.columns.isin([y])]
    
    return train_test_split(X,Y,test_size = test_size , random_state = 42, shuffle = True)
    

X_train , X_test , y_train , y_test = train_test(df,'TravelInsurance',test_size = 0.2)

def dummy_classifier():
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train,y_train)
    y_train_proba =model.predict_proba(X_train)[:,1]
    y_valid_proba = model.predict_proba(X_test)[:,1]
    
    return model , y_train_proba , y_valid_proba

def train_pred_gs(alg_name,alg,parameters,cv,scoring):

    model = GridSearchCV(alg,param_grid=parameters,cv= cv,scoring =scoring)
    model.fit(X_train,y_train)
    
    if hasattr(model , 'best_estimator'):
        best = model.best_estimator_.named_steps[alg_name]

    else:
        best =model
        
    y_train_proba =best.predict_proba(X_train)[:,1]
    y_valid_proba = best.predict_proba(X_test)[:,1]
        
    return best , y_train_proba,y_valid_proba

def calc_auc_roc(y, prob_pred):
      return roc_auc_score(y, prob_pred)



### Making pipelines

seed = 42

pipelines = {
    'Dummy': Pipeline([('Dummy', DummyClassifier(strategy='most_frequent'))]), # base line
    'Log': Pipeline([('Log',LogisticRegression(random_state=seed,  penalty='l2'))]),
    'Forrest': Pipeline([('Forrest', RandomForestClassifier(random_state=seed, oob_score=True))]),
    'xgb': Pipeline([('xgb', XGBClassifier(random_state=seed))]),
    'lgbm': Pipeline([('lgbm', LGBMClassifier(random_state=seed))]),
}

xgb_hyperparameters = {
    'xgb__max_depth': np.arange(2,12,2),  # the maximum depth of each tree
    'xgb__learning_rate': [0.1,0.3],  # the training step for each iteration
    'xgb__n_estimators': np.arange(1,80,10),
}

lgbm_hyperparameters = {
    'lgbm__n_estimators': np.arange(10,140,20),
    'lgbm__min_data_in_leaf': np.arange(100,1000,100),
    'lgbm__max_depth': np.arange(2,10,2),

}


log_params = {'Log__solver':['liblinear' , 'lbfgs' , 'saga'],
             'Log__C' : [0.01,0.05,0.1] , 
             "Log__max_iter" : [500 , 1000]
    }


forrest_params = {'Forrest__criterion':['gini' , 'entropy'],
             'Forrest__n_estimators' : [100,1000] , 
             "Forrest__max_depth" : [5 , 10 , 20]}


hyperparameters = {'Log' : log_params , 
                   'Forrest':forrest_params,
                   'lgbm':lgbm_hyperparameters,
                   "xgb":xgb_hyperparameters}


model_names = {"Dummy" : 'DummyClassifier','Log' : 'LogisticRegression' , 'Forrest' : 'RandomForestClassifier'
               ,"xgb": "XGBoost", "lgbm": "Light Gradient Boosting"}




results = []

cv , scoring = 5 , 'accuracy'



fig, ax =  plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for key,pipe in tqdm(pipelines.items()):
    if key == 'Dummy':
        best ,y_train_proba,y_val_proba= dummy_classifier()

        
    else:
        best,y_train_proba,y_val_proba = train_pred_gs(key,pipe,hyperparameters[key],cv,scoring)
    
    result ={}
    
    result['model'] = key
    result['training auc score'] = calc_auc_roc(y_train,y_train_proba)
    result['testing auc score'] = calc_auc_roc(y_test,y_val_proba)
    result["Brier score"] = brier_score_loss(y_test, y_val_proba)
    
    results.append(result)
    
    if key == 'Dummy':
      continue
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_val_proba, n_bins=10)
    
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (model_names[key], ))
    plt.legend(loc = 'upper left')
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")
    ax.set_title('Calibration plot')

    filename = r'path'.format(model_names[key])
    joblib.dump(best ,filename )


result_df = pd.DataFrame(results)


model = joblib.load(r'path')





### Reading test data

path2 = r'path\travel_insurance\test.csv'

test = read_data(path2,False)   
 
cat_cols = ['EverTravelledAbroad','FrequentFlyer', 'GraduateOrNot' , 'Employment Type']
cols_to_drop = ['Customer Id']

test2 =clean_data(test ,cols_to_drop,cat_cols )

print(sorted(list(pd.qcut(test2.Age,2).unique())))


test2.loc[(test2.Age >= 25) & (test2.Age <= 30) , "Age"] = 1
test2.loc[test2.Age > 30 ,"Age"] = 0

print(sorted(list(pd.qcut(test2.FamilyMembers,3).unique())))

test2.loc[(test2.FamilyMembers >= 1.99) & (test2.FamilyMembers < 4) , "FamilyMembers"] = 1
test2.loc[(test2.FamilyMembers >= 4) & (test2.FamilyMembers < 5) , "FamilyMembers"] = 2
test2.loc[(test2.FamilyMembers >= 5) , "FamilyMembers"] = 3

print(sorted(list(pd.qcut(test2.AnnualIncome,3).unique())))

test2.loc[(test2.AnnualIncome >= 299999) & (test2.AnnualIncome < 750000) , "AnnualIncome"] = 1
test2.loc[(test2.AnnualIncome >= 750000) & (test2.AnnualIncome < 1200000) , "AnnualIncome"] = 2
test2.loc[(test2.AnnualIncome >= 1200000) & (test2.AnnualIncome < 1800000) , "AnnualIncome"] = 3
test2.loc[(test2.AnnualIncome >= 1800000)  , "AnnualIncome"] = 4

### Loading best model
best_model_name = r'path\Light Gradient Boosting'
selected_model = joblib.load(f"{best_model_name}.joblib")
print(f"selected model is {best_model_name[8:]}.\n")
print("Its parameters are:")
selected_model.get_params()


fig, ax =  plt.subplots(figsize=(7, 7))
ax.set_title('AUC ROC Curve of the light gbm Model')
ax.plot([0,1],[0,1],linestyle = '--',c='black')
plot_roc_curve(selected_model, X_test, y_test, ax=ax)


###prob prediction of test data
features = test.columns.tolist()
features.remove('Customer Id')
test['Prediction'] = selected_model.predict_proba(test[features].values)[:,1]



