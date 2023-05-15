'''
Wrangle Zillow Data

Functions:
- wrangle_zillow
- split_zillow
- mm_zillow
- std_zillow
- robs_zillow
- encode_county
'''

### IMPORTS ###

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from env import user,password,host

### ACQUIRE DATA ###

def get_zillow(user=user,password=password,host=host):
    """
    This function acquires data from a SQL database of 2017 Zillow properties and caches it locally.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `aq_zillow` is returning a dirty pandas DataFrame
    containing information on single family residential properties
    """
    # name of cached csv
    filename = 'zillow.csv'
    # if cached data exist
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    # wrangle from sql db if not cached
    else:
        # read sql query into df
        # 261 is single family residential id
        df = pd.read_sql('''select *
                            from properties_2017
                            left join predictions_2017 using (parcelid)
                            where propertylandusetypeid = 261'''
                            , f'mysql+pymysql://{user}:{password}@{host}/zillow')
        # cache data locally
        df.to_csv(filename, index=False)
    return df

def wrangle_zillow():
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.
    
    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, property value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    df = get_zillow()
    # nulls account for less than 1% so dropping
    df = df.dropna()
    # rename columns
    df = df.rename(columns=({'yearbuilt':'year'
                            ,'bedroomcnt':'beds'
                            ,'bathroomcnt':'baths'
                            ,'calculatedfinishedsquarefeet':'area'
                            ,'taxvaluedollarcnt':'prop_value'
                            ,'taxamount':'prop_tax'
                            ,'fips':'county'}))
    # map county to fips
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    # make int
    ints = ['year','beds','area','prop_value']
    for i in ints:
        df[i] = df[i].astype(int)
    # handle outliers
    df = df[df.area < 25000].copy()
    df = df[df.prop_value < df.prop_value.quantile(.95)].copy()
    df = df[(df['beds'] != 0) & (df['baths'] != 0)].copy()
    return df

### SPLIT DATA ###

def split_data(df):
    '''Split into train, validate, test with a 60/20/20 ratio'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=42)
    return train, validate, test

### SCALERS ###

def mm_zillow(train,validate,test,scale=None):
    """
    The function applies the Min Max Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    mm_scale = MinMaxScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(mm_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(mm_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(mm_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

def std_zillow(train,validate,test,scale=None):
    """
    The function applies the Standard Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    std_scale = StandardScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(std_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

def robs_zillow(train,validate,test,scale=None):
    """
    The function applies the RobustScaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    rob_scale = RobustScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(rob_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(rob_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(rob_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

### ENCODE ###

def encode_county(df):
    '''
    Encode county column from zillow dataset
    '''
    df['Orange'] = df.county.map({'Orange':1,'Ventura':0,'LA':0})
    df['Ventura'] = df.county.map({'Orange':0,'Ventura':1,'LA':0})
    return df