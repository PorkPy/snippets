

## Training data file name
training_data = 'flattened_data_v2'


## imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from pprint import pprint
from sklearn.utils import shuffle
from sklearn.base import clone

## Feature Selection
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, chi2 

## Preprocessing 
from sklearn.preprocessing import (StandardScaler, 
                                   RobustScaler, 
                                   PolynomialFeatures,
                                  )
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import (make_pipeline, 
                             make_union, 
                             FeatureUnion,
                             )
from sklearn.model_selection import (train_test_split, 
                                     cross_validate, 
                                     GridSearchCV, 
                                     RandomizedSearchCV,
                                    )
## Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (VotingClassifier, 
                             StackingClassifier,
                             BaggingClassifier,
                             RandomForestClassifier, 
                             AdaBoostClassifier, 
                             ExtraTreesClassifier, 
                             GradientBoostingClassifier,
                             )
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

## Metrics
from sklearn.metrics import (confusion_matrix, 
                             ConfusionMatrixDisplay, 
                             precision_score, 
                             recall_score, 
                             roc_auc_score,
                             f1_score,
                             accuracy_score,
                             classification_report,
                            )

## Extrenal Transformers
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE
## External Models
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

def get_data(data, _SMOTE=True): 
    
    np.random.seed(42)
    df = data
    ## convert datetimes.
    #data['sr_open_date'] = pd.to_datetime(data['sr_open_date'])

    #Features identified as detrimental to model.
    df = df.drop(columns=['master_case_number', ])
#                           'changecontact/securityinfo', 
#                           'reviewcase&decideonaction', 
#                           'annualreview', 
#                           'changeaddressdetails', 
#                           'generalenquiry', 
#                           'outofarrears', 
#                           'changedeoemploymentdetails'])

    #df = shuffle(df)
    df.reset_index(drop=True, inplace=True)

    
    ## define features and targets
    X = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    
    ## Train Test split. 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.10, 
                                                       shuffle=True,
                                                       stratify = y,
                                                       random_state=42)
    ## balence targets
    if _SMOTE == True:
        #ada = SMOTE(random_state=42)
        smote_on_1 = int(len(X_train)*0.5)  ## make minority up to 1/2 the majority. Even dist leads to over-fitting. 
        ada = SMOTE( random_state=42)#, sampling_strategy={ 1: smote_on_1} ) ## SMOTENC cat variables [30,31,32,33,34,35,36]
        X_train, y_train = ada.fit_resample(X_train, y_train)
        
    X_train = pd.DataFrame(X_train)
    X_train.columns = df.columns.drop(['target'])
    #y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    X_test.columns = df.columns.drop(['target'])
    #y_test = pd.DataFrame(y_test)

    ## find proportion of positives and negatives.
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)


    ## reshape targets or get moaned at by the model.
    ## not needed if converted to dfs
    y_train = y_train.reshape(len(y_train))
    y_test = y_test.reshape(len(y_test))
    
    print("\n Target unique counts \n", np.asarray((train_unique, train_counts)).T)
    print("\n Test unique counts \n", np.asarray((test_unique, test_counts)).T)
    print("\n Data shape", X_train.shape, X_test.shape, y_train.shape, y_test.shape,)
    return X_train, X_test, y_train, y_test, df

#----------------------------------------------------------------------------


def build_model(model='RFC', grid_search=False):
    print("Model:",model) 
   
    
    XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1.0,
                  enable_categorical=False, eta=0.4, gamma=0.5, gpu_id=-1,
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.400000006, max_delta_step=0, max_depth=6,
                  min_child_weight=1,monotone_constraints='()',
                  n_estimators=100, n_jobs=12, num_parallel_tree=4,
                  predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
                  #scale_pos_weight=1, 
                        subsample=0.8, tree_method='exact',
                  validate_parameters=1, verbosity=None, num_class=1)

    ETC = ExtraTreesClassifier(bootstrap=True, 
                                 criterion="gini", 
                                 max_features=0.4, 
                                 min_samples_leaf=5, 
                                 min_samples_split=16, 
                                 n_estimators=100)

    RFC = RandomForestClassifier(bootstrap=False, 
                                 criterion="gini",
                                 max_depth=24,
                                 max_features=0.25, 
                                 min_samples_leaf=1, 
                                 min_samples_split=4, 
                                 n_estimators=50)


    GBC = GradientBoostingClassifier(learning_rate=0.1, 
                                   max_depth=8, 
                                   max_features=0.15000000000000002, 
                                   min_samples_leaf=16, 
                                   min_samples_split=10, 
                                   n_estimators=100, 
                                   subsample=0.25)

    SVC_ = SVC()
    ABC = AdaBoostClassifier()
    DTC = DecisionTreeClassifier()
    KNC = KNeighborsClassifier()
    LR  = LogisticRegression()
    MNB = MultinomialNB()
    #CAT = CatBoostClassifier(silent=True)
    MLP = MLPClassifier()
    
    models_dict = { 'MLP': MLP,
                    'XGB': XGB,
                    #'CAT': CAT,
                    'MNB': MNB,
                    'KNC': KNC,
                    'DTC': DTC,
                    'ABC': ABC,
                    'SVC': SVC,
                    'GBC': GBC,
                    'RFC': RFC,
                    'ETC': ETC,
                    'LR': LR,
                    }

    clf = models_dict.get(model)

    """  Transformers. """
    Column_Transformer = make_column_transformer(
                        (
                        #OneHotEncoder(),
                        RobustScaler(),
                        make_column_selector(dtype_include=float))) 
    
    Transformers = [ ('col_trans', Column_Transformer),
                    #('ohe', OneHotEncoder()),
                    #('Scaler', RobustScaler()),
                    #('kernel_pca', KernelPCA()),
                    #('reduce_dim', PCA(.98)),
                    #('KBest', SelectKBest(chi2, k=35)), #.fit_transform(X, y)),
                    #('FEATURE', PolynomialFeatures(degree=2,include_bias=False))
    ]
    ## Combine transformed features
    Transformer_Union = FeatureUnion(Transformers)
    """********************************************"""

    """ Estimators """
    # Create Base Learners for stacking. XGB+RFC+CAT = 0.948
    base_learners = [
                      ('XGB', XGB), # 0.941 # 0.945
                      #('ETC', ETC), # 0.930 # 0.938
                      ('RFC', RFC), # 0.943 # 0.947
                      #('DTC', DTC), # 0.916 # 0.914
                      #('GBC', GBC), # 0.932 # 0.938
                      #('SVC', svc), # 0.830
                      #('KNC', KNC), # 0.848
                      #('ABC', ABC), # 0.906
                      #('LR', LR),   # 0873
                      #('MNB', MNB),  # 0.812
                      #('CAT', CAT),  # ----- # 0.945
    ]

    # # Initialize Stacking Classifier with the Meta Learner
    stack = StackingClassifier(estimators      = base_learners, 
                               final_estimator = LR,
                              verbose=2) # Base=XGB=0.941. XGB+RFC+CAT = 0.950
    ## Voting Classifier
    vote = VotingClassifier(estimators = base_learners, #XGB=0.941
                            voting     ='soft');
    ## Bagging Classifier
    bag = BaggingClassifier(base_estimator = XGB, # XGB=0.940 RFC=0.941
                            n_estimators   = 100, # maybe n_estimatoers is too small. 
                            random_state   = 0)
    """****************************************"""

    """ Clf PIPELINE """
    clf = make_pipeline(None, clf); ## use * infront of Transformers to unpack list when not using union.
    
    """******************************************"""

    """  GRID SEARCH. """
    if grid_search:
        
        X_train, X_test, y_train, y_test, df = get_data(_SMOTE=_SMOTE)
        clf = RandomizedSearchCV(estimator = clf, param_distributions = get_gridsearch(), 
                                                   n_iter = 100, #100
                                                   cv = 3,#3 
                                                   verbose=2, 
                                                   random_state=42, 
                                                   n_jobs = -1)
        ## Retrain on best estimator 
        clf.fit(X_train, y_train)
        print("best_estimator_ \t\n:", clf.best_estimator_)
        print("best_score_ \t\n",      clf.best_score_)
        print("best_params_ \t\n",     clf.best_params_)
        clf = clf.best_estimator_
        print(clf) 
    """************************************************************"""

    return clf

#------------------------------------------------------------------------------


    ### Randomised Grid Search

def get_gridsearch():
    
    """
        Params example for nested classifiers.
        Use 'clf.get_params()'' to get a list of the different parameters available.
        
        params = [{'votingclassifier__XGB__learning_rate':    [0.1, 0.01, 0.001, 0.0001],
                   'votingclassifier__XGB__min_child_weight': [1,2,3,4],
                   'votingclassifier__XGB__subsample':        [0.2, 0.4, 0.6, 0.8]}]
        """
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 200, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'randomforestclassifier__n_estimators': n_estimators,
                   'randomforestclassifier__max_features': max_features,
                   'randomforestclassifier__max_depth': max_depth,
                   'randomforestclassifier__min_samples_split': min_samples_split,
                   'randomforestclassifier__min_samples_leaf': min_samples_leaf,
                   'randomforestclassifier__bootstrap': bootstrap}
    #pprint(random_grid)

    return random_grid

#----------------------------------------------------------------------------

def train(model='RFC', X=None,y=None, data=None, grid_search=False, _SMOTE=False):
    
    if data is not None:
        print("Using user defined data")
        X_train, X_test, y_train, y_test, df = get_data( data, _SMOTE=_SMOTE)

    if X is not None:
        print("Using user defined X and y")
        ## Train Test split. 
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.10, 
                                                        shuffle=True,
                                                        stratify = y,
                                                        random_state=42)

    """ get model. """
    clf = build_model(model, grid_search)

    """ Train. """
    print("Training...")
    np.random.seed(4)
    clf.fit(X_train, y_train)
    
    """ Test. """
    print("Testing...")
    preds = clf.predict(X_test)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print("Train Score \t", train_score)
    print("Test Score \t", test_score)
    ## of all the cases predicted positive, how many were actually positive?
    #print("precision_score ", precision_score(y_test, preds))
    ## of all the positive cases, how may did the model identify?
    #print("recall_score \t", recall_score(y_test, preds))
        
    return  preds, clf, X_test, y_test, X_train, y_train



if __name__ == "__main__":

    ## get data location PATH from path.txt file
    fileObject = open("path.txt", "r")
    PATH = fileObject.read()
    data = pd.read_csv(PATH +'/'+ training_data +'.csv')
    
    ## convert datetimes to int
    data['sr_open_date'] = pd.to_datetime(data['sr_open_date'])
    data['sr_open_date'] = data['sr_open_date'].apply(lambda x: x.value)

    ## Train and test model.
    train(model='XGB',data=data, grid_search=False, _SMOTE=False)
    print("Training complete")


