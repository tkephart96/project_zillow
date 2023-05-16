# Project Zillow

Predict the tax assessed property value of Zillow Single Family Residential properties with transaction date in 2017

### Project Description

As the most-visited real estate website in the United States, Zillow and its affiliates offer customers an on-demand experience for selling, buying, renting and financing with transparency and nearly seamless end-to-end service. I have decided to look into the different elements that determine tax assessed property value.

### Project Goal

* Discover drivers of property value
* Use drivers to develop a machine learning model to predict property value
* This information could be used to further our understanding of how Single Family properties are tax assessed for their property value

### Initial Thoughts

My initial hypothesis is that drivers of tax assessed property value will be the elements like number of rooms, square feet, and location.

## The Plan

* Acquire data from Codeup MySQL DB
* Prepare data
  * Create Engineered columns from existing data
* Explore data in search of drivers of property value
  * Answer the following initial questions
    * Is there a correlation between area and property value?
    * Is there a correlation between beds and property value?
    * Is there a correlation between baths and property value?
    * Is there a correlation between rooms and property value?
* Develop a Model to predict property value
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Original                     | Feature    | Type    | Values              | Definition                                               |
| :--------------------------- | :--------- | :------ | :------------------ | :------------------------------------------------------- |
| yearbuilt                    | year       | Year    | 1801-2016           | The year the principal residence was built               |
| bedroomcnt                   | beds       | Numeric | 1-25                | Number of bedrooms in home                               |
| bathroomcnt                  | baths      | Numeric | 0.5-32              | Number of bathrooms in home including fractional         |
| calculatedfinishedsquarefeet | area       | SqFt    | 1~1mil              | Calculated total finished living area                    |
| taxvaluedollarcnt (target)   | prop_value | USD     | 1~98mil             | The total tax assessed value of the parcel/home          |
| taxamount                    | prop_tax   | USD     | 1.85~1.3mil         | The total property tax assessed for that assessment year |
| fips                         | county     | County  | LA, Orange, Ventura | Federal Information Processing Standard (these 3 in CA)  |
| Additional Features          |            | Numeric | 1=True, 0=False     | Encoded categorical variables                            |

FIPS County Codes:

* 06037 = LA County, CA
* 06059 = Orange County, CA
* 06111 = Ventura County, CA

## Steps to Reproduce

1) Clone this repo
2) If you have access to the Codeup MySQL DB:
   - Save **env.py** in the repo w/ `user`, `password`, and `host` variables
   - Run notebook
3) If you don't have access:
   - Request access from Codeup
   - Do step 2

# Conclusions

#### Takeaways and Key Findings

* There is a correlation between each feature (area, beds, baths, and rooms) with property value
* Area seemed to be the most correlated feature and models worked best with it included

#### Recommendations

* Make sure that area is included and accurate when a property is added for best predictions

#### Next Steps

* Given more time I could check other features such as location to hopefully get more insights and build a better model
