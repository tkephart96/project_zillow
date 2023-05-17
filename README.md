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
    * Is there a correlation between age and property value?
    * Is there a correlation between the room count and property value?
    * Is there a difference in average property value between counties?
* Develop a Model to predict property value
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

| Original                     | Feature    | Type    | Definition                                              |
| :--------------------------- | :--------- | :------ | :------------------------------------------------------ |
| yearbuilt                    | year       | Year    | The year the principal residence was built              |
| bedroomcnt                   | beds       | Numeric | Number of bedrooms in home                              |
| bathroomcnt                  | baths      | Numeric | Number of bathrooms in home including fractional        |
| roomcnt                      | roomcnt    | Numeric | Total number of rooms in the property                   |
| calculatedfinishedsquarefeet | area       | SqFt    | Calculated total finished living area                   |
| taxvaluedollarcnt (target)   | prop_value | USD     | The total tax assessed value of the parcel/home         |
| fips                         | county     | County  | Federal Information Processing Standard (these 3 in CA) |
| latitude                     | latitude   | Numeric | Latitude coordinates of property                        |
| longitude                    | longitude  | Numeric | Longitude coordinates of property                       |
| Additional Features          |            | Numeric | Encoded categorical variables                           |
|                              | age        | Year    | How many years from 2017 since it was built             |

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

* The younger the property the better for property value
* The bigger the living area the bigger the property value
* Location matters for property value
* Model still needs improvement

### Recommendations and Next Steps

* It would nice to have the data to check if the included appliances or the type of heating services (gas or electric) of the property would affect property value
* More time is needed to work on features to better improve the model
    - latitude and longitude could hopefully give insights into cities and neighborhoods with higher or lower property values
    - pools and garages could also be looked into
