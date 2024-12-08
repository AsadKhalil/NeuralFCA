import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import LineVizNx
import matplotlib.pyplot as plt
import torch
import os
import neural_lib as nl
plt.rcParams['figure.facecolor'] = (1,1,1,1)


file_path = 'income-data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
df.head()

# Replace '?' with NaN for proper cleaning
data_cleaned = df.replace('?', pd.NA)

# Remove rows with any missing values (including replaced '?')
data_cleaned = data_cleaned.dropna()
data_cleaned.drop(columns = ['capital-gain','capital-loss'], inplace= True)
# Separate X and Y again after cleaning
X = data_cleaned.drop(columns=['income'])
Y = data_cleaned['income']

# Display the updated shapes of X and Y
print(X.shape, Y.shape)
# Count unique values for each column in X
unique_values_count = X.nunique()

# Display the unique values count for each column
print(unique_values_count)
Y_binarized = Y.apply(lambda x: 1 if x.strip() == '>50K' else 0)  # Encode target variable
X_binarized = X.copy()
age_bins = [0, 17, 24, 34, 44, 54, 64, np.inf]
age_labels = ['under_18', '18_24', '25_34', '35_44', '45_54', '55_64', '65_plus']
X['age_group'] = pd.cut(X['age'], bins=age_bins, labels=age_labels, right=False)
for label in age_labels:
    X['age_' + label] = (X['age_group'] == label).astype(int)

workclass_categories = X['workclass'].unique()
for category in workclass_categories:
    X['workclass_' + category] = (X['workclass'] == category).astype(int)

    # Binarize 'fnlwgt'
fnlwgt_quartiles = pd.qcut(X['fnlwgt'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    X['fnlwgt_' + quartile] = (fnlwgt_quartiles == quartile).astype(int)

educational_num_values = X['educational-num'].unique()

for value in educational_num_values:
    X['educational_num_' + str(value)] = (X['educational-num'] == value).astype(int)


educational_num_values = X['educational-num'].unique()

for value in educational_num_values:
    X['educational_num_' + str(value)] = (X['educational-num'] == value).astype(int)


marital_status_categories = X['marital-status'].unique()

for category in marital_status_categories:
    X['marital_status_' + category] = (X['marital-status'] == category).astype(int)


occupation_categories = X['occupation'].unique()

for category in occupation_categories:
    X['occupation_' + category] = (X['occupation'] == category).astype(int)


relationship_categories = X['relationship'].unique()

for category in relationship_categories:
    X['relationship_' + category] = (X['relationship'] == category).astype(int)


race_categories = X['race'].unique()

for category in race_categories:
    X['race_' + category] = (X['race'] == category).astype(int)

X['gender_male'] = (X['gender'] == 'Male').astype(int)

# Define hours per week ranges and labels
hours_bins = [0, 34, 40, np.inf]
hours_labels = ['part_time', 'full_time', 'over_time']

# Create hours group categories
X['hours_group'] = pd.cut(X['hours-per-week'], bins=hours_bins, labels=hours_labels, right=True)

# Create binary columns for each hours group
for label in hours_labels:
    X['hours_' + label] = (X['hours_group'] == label).astype(int)


# Define country to region mapping
country_to_region = {
    'United-States': 'North America',
    'Canada': 'North America',
    'Outlying-US(Guam-USVI-etc)': 'North America',
    # Central America
    'Mexico': 'Central America',
    'Puerto-Rico': 'Central America',
    'Honduras': 'Central America',
    'Cuba': 'Central America',
    'Jamaica': 'Central America',
    'Haiti': 'Central America',
    'Dominican-Republic': 'Central America',
    'Guatemala': 'Central America',
    'Nicaragua': 'Central America',
    'El-Salvador': 'Central America',
    'Trinadad&Tobago': 'Central America',
    # South America
    'Columbia': 'South America',
    'Ecuador': 'South America',
    'Peru': 'South America',
    # Asia
    'India': 'Asia',
    'Japan': 'Asia',
    'China': 'Asia',
    'Iran': 'Asia',
    'Philippines': 'Asia',
    'Vietnam': 'Asia',
    'Hong': 'Asia',
    'Thailand': 'Asia',
    'Cambodia': 'Asia',
    'Laos': 'Asia',
    'Taiwan': 'Asia',
    # Europe
    'England': 'Europe',
    'Germany': 'Europe',
    'Greece': 'Europe',
    'Italy': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Ireland': 'Europe',
    'France': 'Europe',
    'Scotland': 'Europe',
    'Yugoslavia': 'Europe',
    'Hungary': 'Europe',
    'Holand-Netherlands': 'Europe',
    # Other
    'South': 'Other',
    # Add any additional mappings if necessary
}

# Map countries to regions
X['region'] = X['native-country'].map(country_to_region)

# Fill NaN values with 'Other'
X['region'] = X['region'].fillna('Other')

# Get unique regions
region_categories = X['region'].unique()

# Create binary columns for each region
for region in region_categories:
    X['native_country_' + region] = (X['region'] == region).astype(int)


columns_to_drop = [
    'age', 'age_group',
    'workclass',
    'fnlwgt',
    'education',
    'educational-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'hours-per-week', 'hours_group',
    'native-country',
    'region'
]

# Drop the columns
X_binarized = X.drop(columns=columns_to_drop)


# Convert the dataframe to boolean (True/False)
X_binarized = X_binarized.astype(bool)
X_binarized.index = X_binarized.index.astype(str)
# Check the data types to confirm

print(X_binarized.dtypes)
print(X_binarized.index.dtype)

X_train, X_test, y_train, y_test = train_test_split(X_binarized, Y_binarized, test_size=0.2, random_state=42)


K_train = FormalContext.from_pandas(X_train)
# K_train = FormalContext.from_pandas(X_binarized)
print(K_train)

# Generate the monotone concept lattice
L = ConceptLattice.from_context(K_train, is_monotone=True)
print(f'Lattice has {len(L)} concepts.')

from sklearn.metrics import f1_score

# Calculate F1 scores for each concept in the lattice
f1_scores = []
for c in L:
    y_preds = np.zeros(K_train.n_objects)
    y_preds[list(c.extent_i)] = 1
    f1 = f1_score(y_train, y_preds)
    f1_scores.append(f1)

# Display the top 7 best concepts
best_concepts = np.argsort(f1_scores)[::-1][:7]
print(f'Best concepts: {best_concepts}')

import neural_lib as nl  # Assuming neural_lib is available

# Create a concept network from the lattice and selected best concepts
targets = sorted(set(y_train))  # The unique values in the target variable
cn = nl.ConceptNetwork.from_lattice(L, best_concepts, targets)

# Ensure that selected concepts cover all training objects
assert len({g_i for c in L[best_concepts] for g_i in c.extent_i}) == K_train.n_objects, "Selected concepts do not cover all train objects"

# Setup visualization for the neural network architecture
vis = LineVizNx(node_label_font_size=14, node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes)) + '\n\n')


fig, ax = plt.subplots(figsize=(15, 5))
vis.draw_poset(
    cn.poset, ax=ax,
    flg_node_indices=False,
    node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True) + '\n\n'
)
plt.title('Neural Network based on 7 best concepts from monotone concept lattice', loc='left', x=0.05, size=24)
plt.tight_layout()
save_path = os.path.join("save_dir", f"concept_lattice_best_concept_.png")

# Save the figure to the specified directory
plt.savefig(save_path)
print(f"Saved visualization to {save_path}")
plt.show()

# Step 5: Train the network on the training data
cn.fit(X_train, y_train)

y_pred = cn.predict(X_test)
y_pred_proba = cn.predict_proba(X_test).detach().numpy()

print('Class predictions:', y_pred.numpy())
print('Class predictions with probabilities:', y_pred_proba)
print('True class labels:', y_test.values)

edge_weights = cn.edge_weights_from_network()

import networkx as nx
fig, ax = plt.subplots(figsize=(15, 5))
vis.draw_poset(
    cn.poset, ax=ax,
    flg_node_indices=False,
    node_label_func=lambda el_i, P: nl.neuron_label_func(el_i, P, set(cn.attributes), only_new_attrs=True) + '\n\n',
    edge_color=[edge_weights[edge] for edge in cn.poset.to_networkx().edges],
    edge_cmap=plt.cm.RdBu,
)
nx.draw_networkx_edge_labels(cn.poset.to_networkx(), vis.mover.pos, {k: f"{v:.1f}" for k, v in edge_weights.items()}, label_pos=0.7)

plt.title('Neural Network with fitted edge weights', size=24, x=0.05, loc='left')
plt.tight_layout()
save_path = os.path.join("save_dir", f"concept_lattice_.png")

# Save the figure to the specified directory
plt.savefig(save_path)
plt.show()