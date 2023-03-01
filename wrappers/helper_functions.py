import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,roc_curve,auc,classification_report

#returns dataset with top variables
def feature_selection(data, num_features, target):
    X = data.drop(target, axis=1)
    y = data[target]
    et_model = ExtraTreesClassifier(n_estimators=50)
    et_model.fit(X, y)
    feature_importances = et_model.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1][:num_features]
    selected_features = X.columns[sorted_idx]
    X_selected = X[selected_features]
    y = data[target]
    train = pd.merge(X_selected, y, left_index=True, right_index=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    fprs = []
    for train_index, test_index in skf.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        et_model.fit(X_train, y_train)
        # et_probs = et_model.predict_proba(X_test)
        # et_preds = et_probs[:,1]
        # et_fpr, et_tpr, et_threshold = roc_curve(y_test, et_preds)
        # optimal_idx = np.argmax(et_tpr - et_fpr)
        # et_optimal_threshold = et_threshold[optimal_idx]
        y_pred = (et_model.predict_proba(X_test)[:, 1] > 0.17).astype('float')
        cnf_matrix = confusion_matrix(y_test, y_pred)
        
        TN, FP, FN, TP = cnf_matrix.ravel()
        
        tprs.append(TP / (TP + FN))
        fprs.append(FP / (FP + TN))
    avg_tpr = round(sum(tprs) / len(tprs),4)*100
    avg_fpr = round(sum(fprs) / len(fprs),3)*100
    print (f"For {num_features} variables TPR is ",avg_tpr)
    print (f"For {num_features} variables FPR is ",avg_fpr)
    return train

#returns regularized encoded variables
def regularized_mean_encoding(train,test,column,alpha):
    global_mean = train['Attrition'].mean()
    encoded_values = {}
    for category in train[column].unique():
        category_size = train[train[column] == category].shape[0]
        category_mean = train[train[column] == category]['Attrition'].mean()
        reg_mean = ((category_mean * category_size) + (global_mean * alpha)) / (category_size + alpha)
        encoded_values[category] = reg_mean

    train[column] = train[column].map(encoded_values)
    test[column] = test[column].map(encoded_values)

    return train,test

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.viridis):
    plt.grid(None)
    plt.clf
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    sns.set(rc={'figure.figsize':(8.7,4.27)})
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.ylim([1.5, -.5])
    plt.tight_layout()
    
    width, height = cm.shape
 
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center',color='black',fontsize=22)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#returns classification stats
def Eval_Statistics(confusion_matrix):
    TN = confusion_matrix[0,0]
    TP = confusion_matrix[1,1]
    FN = confusion_matrix[1,0]
    FP = confusion_matrix[0,1]

    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    Specificity = TN/(TN+FP)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)

    print('TPR :',TPR)
    print('FPR :',FPR)
    print('TNR :',TNR)
    print('FNR :',FNR)
    
    return(Recall,Specificity)

#calculates corrs and plots
def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap.values)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'blue','red')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(15,10),legend=False)
    ax.set_xlabel("SHAP Value (Blue = Positive Impact)")

#calculate per decile values
def Decile_Analysis(df,Trained_model):
    Decile_set = df.copy(deep = True)
    predict_set = df.copy(deep = True)
    predict_set = predict_set.drop(['Attrition'],axis = 1)
    Decile_set['Probs'] = Trained_model.predict_proba(predict_set)[:,1]
    Decile_set['Decile'] = pd.qcut(Decile_set['Probs'], 10, labels=[10,9,8,7,6,5,4,3,2,1])
    Responders = {'1st Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 1)),
              '2nd Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 2)),
              '3rd Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 3)),
              '4th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 4)),
              '5th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 5)),
              '6th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 6)),
              '7th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 7)),
              '8th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 8)),
              '9th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 9)),
              '10th Decile' : sum((Decile_set.Attrition == 1) & (Decile_set.Decile == 10))}
    


    customers_per_decile = {'1st Decile' : sum(Decile_set.Decile == 1),
         '2nd Decile' : sum(Decile_set.Decile == 2),
         '3rd Decile' : sum(Decile_set.Decile == 3),
         '4th Decile' : sum(Decile_set.Decile == 4),
         '5th Decile' : sum(Decile_set.Decile == 5),
         '6th Decile' : sum(Decile_set.Decile == 6),
         '7th Decile' : sum(Decile_set.Decile == 7),
         '8th Decile' : sum(Decile_set.Decile == 8),
         '9th Decile' : sum(Decile_set.Decile == 9),
         '10th Decile': sum( Decile_set.Decile == 10)
        }


    
    customers_per_decilee = list(customers_per_decile.values())
    customers_per_decilee
    cumulative_customers =[]
    iterator = 0
    for i in range(0,len(customers_per_decilee)):
        if iterator == 0:
            iterator = 1
            cumulative_customers.append(customers_per_decilee[i])
        else:        cumulative_customers.append(cumulative_customers[i-1] + customers_per_decilee[i])
    cumulative_customers
    Cumulative_percentage_customers = [10,20,30,40,50,60,70,80,90,100]
    Responderss = list(Responders.values())
    Responders

    Response_rate = np.divide(Responderss,customers_per_decilee)*100


    perc_responsders = np.divide(Responderss,sum(Responderss))*100
    perc_responsders

    Gain = []
    iterator = 0
    for i in range(0,len(perc_responsders)):
        if iterator == 0:
            iterator = 1
            Gain.append(perc_responsders[i])
        else:        
            Gain.append(Gain[i-1] + perc_responsders[i])

    Gain

    Lift = np.divide(Gain,Cumulative_percentage_customers)
    max_probs = Decile_set.groupby(['Decile'])['Probs'].max()
    min_probs = Decile_set.groupby(['Decile'])['Probs'].min()
    max_probs = np.flip(max_probs)
    min_probs = np.flip(min_probs)
    
    Decilee = [1,2,3,4,5,6,7,8,9,10]
    Decile_analysis = {'Decile': Decilee,
                  'No. of Customers':customers_per_decilee,
                   'Cumulative Customers':cumulative_customers,
                   'Cumulative % customers': Cumulative_percentage_customers,
                   'Responders' : Responderss,
                   'Response Rate': Response_rate,
                   'percentage_responders' : perc_responsders,
                   'Gain' : Gain,
                   'lift' : Lift,
                   'Max Prob': max_probs,
                   'Min Prob': min_probs
                  }
    analysis = pd.DataFrame(Decile_analysis) 
    analysis=analysis.reset_index(drop =True)

    return(analysis)