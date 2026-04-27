#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score


#%%
file_path = 'citrus.csv'
df = pd.read_csv(file_path) 
print("Dataset berhasil dimuat!")

df.head()


#%%
df.describe() 


#%%
le = LabelEncoder()
df['name_encoded'] = le.fit_transform(df['name']) # 0: grapefruit, 1: orange

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='diameter', hue='name', kde=True)
plt.title('Distribusi Diameter Buah (Jeruk vs Anggur)')
plt.show()


#%%
plt.figure(figsize=(8, 6))
df_numeric = df.drop(columns=['name']) 
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Fitur')
plt.show()


#%%
X = df.drop(columns=['name', 'name_encoded'])
y = df['name_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"Jumlah Data Latih: {len(X_train)}")
print(f"Jumlah Data Uji: {len(X_test)}")


#%%
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

y_pred_proba_dict = {}

for name, model in models.items():
    print(f"\n================ {name.upper()} ================")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba_dict[name] = model.predict_proba(X_test)[:, 1] 
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plt.figure(figsize=(5, 4))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


#%%
plt.figure(figsize=(8, 6))

for name in models.keys():
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_dict[name])
    plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.show()


#%%
plt.figure(figsize=(8, 6))

for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_dict[name])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# %%
