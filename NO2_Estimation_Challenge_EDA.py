import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=PerformanceWarning)

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
df = train._append(test, ignore_index=True)

##################################
# 1. Capturing Numeric and Categorical Variables
##################################
def grab_col_names(dataframe, cat_th=12, car_th=20):
    """
    Extract column names for a given dataframe.

    param dataframe: The dataframe to analyze.
    param cat_th: Threshold for numerical columns to be considered categorical.
    param car_th: Threshold for categorical columns to be considered as cardinal.
    return: Lists of categorical columns, categorical but cardinal columns, and numerical columns.
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() <= cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = number of variables.
    # num_but_cat is already included in cat_cols.
    # Therefore, all variables will be selected with these three lists: cat_cols + num_cols + cat_but_car.
    # num_but_cat is provided only for reporting purposes.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

##################################
# 2. Data Overview
##################################
train.head()
test.head()

train.ID.nunique()
test.ID.nunique()

df["site_ID"] = df["LAT"].astype(str) + "_" + df["LON"].astype(str)
df.site_ID.nunique()

df["ID"].nunique()

df["ID_Zindi"].nunique()

##################################
# 3. Handling Missing Values
##################################
train.isnull().sum()

for col in num_cols:
    for site_id in train.ID.unique():
      train.loc[train["ID"] == site_id, col] = train.loc[train["ID"] == site_id, col].bfill().ffill()


train.isnull().sum()

##################################
# 4. Descriptive Statistics and Visualization for Numerical Variables
##################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("###########################################")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols[:-1]:
    num_summary(train, col, plot=True)

num_summary(train, "GT_NO2", plot=True)

##################################
# 5. Feature Engineering with Date Variables
##################################
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
train['NEW_dayofyear'] = train['Date'].dt.day_of_year
train['NEW_month'] = train['Date'].dt.month
train['NEW_dayofMonth'] = train['Date'].dt.day
train["NEW_dayofweek"] = train["Date"].dt.dayofweek
train["NEW_week_of_year"] = train['Date'].dt.isocalendar().week


##################################
# 6. Visualizations for Date Variables
##################################
train.groupby("NEW_dayofyear").agg({"GT_NO2": ["mean", "std"]})

sns.lineplot(y=train['GT_NO2'],x=train['NEW_dayofyear'])

sns.lineplot(x=train['NEW_month'],y=train['GT_NO2'])

sns.lineplot(x=train['NEW_dayofMonth'],y=train['GT_NO2'],hue=train['NEW_month'])

sns.lineplot(x=train['NEW_dayofMonth'],y=train['GT_NO2'])

sns.lineplot(y=train['GT_NO2'],x=train['NEW_dayofweek'])

sns.lineplot(y=train['GT_NO2'],x=train['NEW_week_of_year'])

sns.lineplot(x=train['Date'].dt.year,y=train['GT_NO2'])


##################################
# 7. Spatial Analysis
##################################
sns.scatterplot(x=train['LAT'],y=train['LON'],hue=train['GT_NO2'])


##################################
# 8. Boxplot Analysis
##################################
for col in num_cols[:-1]:
    sns.boxplot(x=df[col], data=df)
    plt.title(f'Boxplot of {col}')
    plt.show()

sns.boxplot(x=df["GT_NO2"], data=df)
plt.title(f'Boxplot of GT_NO2')

##################################
# 9. Correlation Analysis
##################################
train[num_cols].corrwith(train["GT_NO2"]).sort_values(ascending=False)

num_cols = [col for col in num_cols if col not in "GT_NO2"]

def high_correlated_cols(df, plot=False, cor_th=0.80):
    corr = df[num_cols].corr()
    cor_matrix = corr. abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > cor_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize":(15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(train, plot=True, cor_th=0.8)

##################################
# 10. Scatter Plots for Highly Correlated Variables
##################################
sns.scatterplot(x=train['NO2_trop'],y=train['GT_NO2'])

sns.scatterplot(x=train['NO2_total'],y=train['GT_NO2'])

sns.scatterplot(x=train['NO2_strat'],y=train['GT_NO2'])

sns.scatterplot(x=train['LST'],y=train['GT_NO2'])

