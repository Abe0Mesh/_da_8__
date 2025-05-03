import pandas as pd
import matplotlib.pyplot as plt





unclean = pd.read_csv("strong.csv")
cleaned = unclean.drop(columns=["Distance", "Seconds", "Notes", "RPE", "Workout Notes"])
cleaned['Date'] = pd.to_datetime(cleaned['Date'])
missing_data = cleaned.isnull().sum()

cleaned.to_csv("cleaned.csv",index=False)


#function below plots my pulll up progression within october

def plot_average_pullups_per_day():
    pullup = cleaned[(cleaned['Exercise Name'] == "Pull Up") & (cleaned['Date'] >= "2024-10-01") &  (cleaned['Date'] <= "2024-10-31")]

    daily_pullup = pullup.groupby('Date').agg(
        total_rep=('Reps', 'sum'),
        total_set=('Set Order', 'count')
    )
    daily_pullup['average_reps'] = daily_pullup['total_rep'] / daily_pullup['total_set']

    plt.figure()
    plt.bar(daily_pullup.index, daily_pullup['average_reps'])
    plt.xlabel("Date")
    plt.ylabel("Average Reps")
    plt.title("Average daily pullup set reps ")
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.show()

# function below plots my daily volume  
def plot_daily_volume_2024():

    data_2024 = cleaned[cleaned['Date'].dt.year == 2024]

    data_2024['Date'] = data_2024['Date'].dt.date

    daily_volume = data_2024.groupby('Date').size()

    plt.figure()
    plt.bar(daily_volume.index, daily_volume.values)
    plt.xlabel("Date")
    plt.ylabel("Number of daily sets")
    plt.title("Daily Volume in 2024")
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.show()
#function below plots weight pusher per day
def plot_total_weight_pushed_per_day_2024():

    data_2024 = cleaned[cleaned['Date'].dt.year == 2024].copy()

    data_2024 = data_2024[data_2024['Weight'] > 0]

    data_2024['Volume'] = data_2024['Weight'] * data_2024['Reps']

    daily_volume = data_2024.groupby(data_2024['Date'].dt.date)['Volume'].sum()

    plt.figure()
    plt.bar(daily_volume.index, daily_volume.values)
    plt.xlabel("Date")
    plt.ylabel("Total weight pushed")
    plt.title("Total weight pushed in 2024")
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()
    plt.show()
#Function below plots my average weight per set 
def plot_average_weight_per_set_2024():
    data_2024 = cleaned[cleaned['Date'].dt.year == 2024].copy()
    data_2024 = data_2024[data_2024['Weight'] > 0]

    data_2024['Volume'] = data_2024['Weight'] * data_2024['Reps']

    daily_stats = data_2024.groupby(data_2024['Date'].dt.date).agg(
        total_volume=('Volume', 'sum'),
        total_sets=('Set Order', 'count')
    )
    daily_stats['average_weight_per_set'] = daily_stats['total_volume'] / daily_stats['total_sets']

    plt.figure()
    plt.plot(daily_stats.index, daily_stats['average_weight_per_set'], marker='o', label='Avg weight')
    plt.xlabel("Date")
    plt.ylabel("Average weight per set")
    plt.title("Average weight per set in 2024")
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.show()
#Function below is used to plot my daily steps data
def plot_daily_steps(daily_steps_df):
    plt.figure()
    plt.bar(daily_steps_df['Date'], daily_steps_df['Step Count'])
    plt.xlabel('Date')
    plt.ylabel('Daily Step Count')
    plt.title('Step count throughout first semester')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Function below is used to do a t test on my weight pushed data

from scipy.stats import ttest_ind
from datetime import datetime

def t_test_weight_pushed(data):

    data['Date'] = pd.to_datetime(data['Date'])

    start_e = datetime.strptime("2024-10-01", "%Y-%m-%d").date()
    end_e = datetime.strptime("2024-10-31", "%Y-%m-%d").date()
    start_l= datetime.strptime("2024-12-01", "%Y-%m-%d").date()
    end_l = datetime.strptime("2024-12-26", "%Y-%m-%d").date()

    data_2024 = data[data['Date'].dt.year == 2024].copy()
    data_2024 = data_2024[data_2024['Weight'] > 0]
    data_2024['Volume'] = data_2024['Weight'] * data_2024['Reps']
    daily_volume = data_2024.groupby(data_2024['Date'].dt.date)['Volume'].sum()

    early_semester = daily_volume[(daily_volume.index >= start_e) & (daily_volume.index <= end_e)]
    late_semester = daily_volume[(daily_volume.index >= start_l) & (daily_volume.index <= end_l)]

    t_stat= ttest_ind(early_semester, late_semester)

    return t_stat



#Function below is used to do a t test on my daily steps data

from scipy.stats import ttest_rel
from datetime import datetime

def t_test_daily_steps(data):
  
    data['Date'] = pd.to_datetime(data['Date'])

    first_half = data[(data['Date'] >= "2024-10-06") & (data['Date'] <= "2024-11-15")]
    second_half = data[(data['Date'] >= "2024-11-15") & (data['Date'] <= "2024-12-26")]

    first_half_steps = first_half['Step Count'].values
    second_half_steps = second_half['Step Count'].values

    minl = min(len(first_half_steps), len(second_half_steps))
    first_half_steps = first_half_steps[:minl]
    second_half_steps = second_half_steps[:minl]

    t_stat = ttest_rel(second_half_steps, first_half_steps)
    return t_stat





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import numpy as np

# This function bellow is used to classify my workout name data
def run_classification_and_plot_tree(cleaned_df):

    test_size = 0.2
    random_state = 17
    k_neighbors = 5
    tree_depth = 3

    workout = ['Evening Workout', 'Afternoon Workout', 'Midday Workout', 'Morning Workout']
    wd = cleaned_df[cleaned_df['Workout Name'].isin(workout)].copy()

   
    wd['Hour'] = wd['Date'].dt.hour

    X = wd[['Hour']]

    y_lab = wd['Workout Name']
    lab = LabelEncoder()
    y = lab.fit_transform(y_lab)
    cp = lab.classes_
    fp = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size,random_state=random_state)

    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print("Knn Accurracy:", knn_accuracy)
    #desision tree
    tree_model = DecisionTreeClassifier(random_state=random_state, max_depth=tree_depth)
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_predictions)
    print("Decision Tree:", tree_accuracy)

    plt.figure(figsize=(12, 8)) 
    plot_tree(tree_model, class_names=list(cp),  feature_names=fp, fontsize=10)
    plt.title("Decision Tree)")
    plt.show() 


