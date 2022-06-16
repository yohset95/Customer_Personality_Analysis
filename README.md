# Customer Personality Analysis using K-Means Clustering (Customer Segmentation)
* Written by Yohanes Setiawan
* This is my final project from Coursera's IBM Machine Learning: Clustering
* For further reading of the code, please open the `.ipynb` file

# **Business Understanding**
* Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers
* Customer Personality Analysis in Customer Segmentation helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment

## Main Objective
Cluster Analysis for Customer Segmentation in order to have a good understanding of Customer Personality Analysis

## Analytical Approach
* Descriptive analysis
* Graph analysis
* Cluster analysis (K-Means Clustering)
* Dimensionality reduction for showing clustering graph

# **Data Understanding**
## Dataset
* The dataset for this project is provided by Dr. Omar Romero-Hernandez.
* Taken from https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

### Dataset Information
![](https://drive.google.com/uc?export=view&id=1yujeeCBby5_7BssMWm-P7dq6cAEC6NbY) </br>

### Checking Missing Values
![](https://drive.google.com/uc?export=view&id=1yo-Ebcu5ptrdMjkN96jyvcj_eKZGe_4S) </br>

### Checking Duplicated Data
![](https://drive.google.com/uc?export=view&id=1NMchDbLWN7Mxnd1P7hzezRDIKSF49kB7) </br>

## Descriptive Statistics

### Numerical Features
![](https://drive.google.com/uc?export=view&id=1stk2V6_1Uu0t4f9wXmMeU3yZ1qd0BqN_) </br>
![](https://drive.google.com/uc?export=view&id=1SpwNSGQmc8EMdqa1wO7yzWA8C8wPRbov) </br>
![](https://drive.google.com/uc?export=view&id=1B7Sx10c2KyDXEIFBDLYth-bEYp7P9kX1) </br>

### Categorical Features
![](https://drive.google.com/uc?export=view&id=1JXIa0Y6HPph8RQ3ybtoo5j1sJLJqtQeT) </br>

# **Data Cleaning**
* Handling Missing Value : I deleted rows with missing value
* Drop Columns : Drop `Z_CostContact` and `Z_Revenue`

# **Feature Engineering**

## Create new Columns for Analysis
* Accepted Campaign = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5
* Amount of Purchases = MntFishProducts + MntFruits + MntGoldProds + MntMeatProducts + MntSweetProducts + MntWines
* Number of Purchases = NumCatalogPurchases + NumDealsPurchases + NumStorePurchases + NumWebPurchases

## Estimating Customer's Age
Estimating customer's age by subtracting the year of `Dt_Customer` and `Year_Birth`

## Categorical Encoding
* I am going to use `Education` in my cluster analysis. Therefore, I need to transform it into numerical column
* For the categorical encoding, I used Label Encoding since educational status is an ordinal variable

Assumption:
* Graduation equals to bachelor's degree
* 2n cycle equals to master's degree

## Choosing Appropriate Columns/Features for Analysis
I am going to choose these features for my customer personal analysis in the field of clustering:
* Age
* Education
* Kidhome
* Teenhome
* Income
* Recency
* Accepted Campaigns
* Amount of Purchases
* Number of Purchases

### Univariate Analysis
![](https://drive.google.com/uc?export=view&id=1XsOv0wjTwaSXUnC1RB7Sh_HTPKjbP0RG) </br>
Skewed:
* `Age`
* `Income`
* `Number of Purchases`
* `Amount of Purchases`

#### Box Plot
![](https://drive.google.com/uc?export=view&id=1KkU3Zct-5m-KEfG1akF5S1jR1AURR_dG) </br>
Each of columns has outliers but not severe

## Correlation Analysis
![](https://drive.google.com/uc?export=view&id=1yqPIUVdIdIYhl9KVk0MrcyHImfPl3zli) </br>
Strong Correlation (> 0.5):
* `Number of Purchases` with `Amount of Purchases` (0.76)
* `Income` with `Number of Purchases` (0.67)
* `Income` with `Amount of Purchases` (0.57)

Insight:
* The higher income of the customers, the higher number of purchases and amount spent to the products

Feature Selection for Cluster Analysis:
* `Income` and `Number of Purchases` should be removed to avoid redundant features

## Log Transformation for Skewed Columns
K-Means performs
well if:
* Data’s distribution if not skewed
* Standardized

![](https://drive.google.com/uc?export=view&id=1J8QocUt4D3Dm6SlJOKShmgytE8MsyZv9) </br>

## Feature Scaling
Feature Scaling is done by standardization

# **Get Insights from Exploratory Data Analysis (EDA)**

#### What is customer's most favorite product?
![](https://drive.google.com/uc?export=view&id=18-otnPiOG02wuJjybPohWMVX2Cg6937U) </br>
Insight:
* Wines and Meat are considered as our most favorite products
* Sweet and Fruits are our less amount of products

#### How is the evaluation of our campaigns?
![](https://drive.google.com/uc?export=view&id=1YV9ZLOQWb7pB621hVyvar9eCXjv4Q0Y8) </br>
Insight:
* Our best accepted campaign is on our last campaign
* Campaign 2 has been our lowest accepted campaign

#### How is customer's favorite purchase place since the day they enroll the company?
![](https://drive.google.com/uc?export=view&id=1EnOBYcCtHeBTEqYO-tNQi_bDk50rpoBq) </br>
Insight:
* Most of customers (46.2%) are likely to purchase directly from our store

#### How is the growth of customer through years?
![](https://drive.google.com/uc?export=view&id=1Fi-mAFDGcTt_uWZrQQ6RKUtsy8VAQI3e) </br>
Insight:
* The highest growth comes from July to August in 2012 before went down
* 2013 has similar customer enrollment as August 2012 on October. 2013 likely can be called as our stable year in customer enrollment
* The drop of our customer enrollment happens in 2014: our customer enrollment went down from May to June which has been the lowest customer enrollment throughout the dataset

# **Modelling**
Used machine learning algorithms:
* K-Means Clustering for Cluster Analysis
* Principle Component Analysis (PCA) for Reducing Dimensionality to plot Clustering Result

### Cluster Analysis using K-Means Clustering
![](https://drive.google.com/uc?export=view&id=1IiYQq5ynid1z60pC5Sd-9OT5udzpmM0r) </br>
Based on Elbow Method, the candidate for number of clusters:
* k = 2
* k = 3
* k = 4

Then, I compared those k's with their Silhouette Scores

| Number of Cluster (k) | Silhouette Score |
|:-------------:|:-------------:|
| 2 | 0.19 |
| **3** | **0.21** |
| 4 | 0.19 |
Therefore, I choose the number of clusters (k) = 3 because it has the highest Silhouette Score

## Dimensionality Reduction for Clustering Plot
I used Principle Component Analysis (PCA) to perform plotting for my clustering result from previous cluster analysis because of limitation of dimensional plotting

![](https://drive.google.com/uc?export=view&id=1QdxB2Sapi2WUfQnqXY5Jodst-FVj4eJA) </br>

# **Interpretation**

## Descriptive Analysis for each Cluster
![](https://drive.google.com/uc?export=view&id=1guNV8DPI_8ZXo53qRmW5228JXLXZd3AT) </br>
![](https://drive.google.com/uc?export=view&id=1HJXC1qkQLvWVbUrRtUSQzbC4X0ttomQF) </br>

## Customer Segmentation for Customer Personality Analysis
Cluster 0
* Count of population: 1125 (51%) - - the highest population
* Average of age: 49 years old -- the oldest population
* Education: 2n cycle or Master's degree
* Kidhome: 0 (There is no kid at home)
* Teenhome: 1 (There is a teenager home)
* Average of recency: 50 days -- the highest average of recency
* Average of accepted campaign: 0 time(s)
* Average of Amount of Purchases: 870

Cluster 1
* Count of population: 911 (41%)
* Average of age: 39 years old -- the youngest population
* Average of education: Bachelor's degree
* Kidhome: 1 (There is a kid at home)
* Teenhome: 0 (There is no teenager at home)
* Average of recency: 49 days
* Average of accepted campaign: 0 time(s)
* Average of Amount of Purchases: 103 - the lowest amount of purchases

Cluster 2
* Count of population: 180 (8%) - the lowest population
* Average of age: 45 years old
* Average of education: 2n cycle or Master's degree / Bachelor's degree
* Kidhome: 0 (There is no kid at home)
* Teenhome: 0 (There is no teenager at home)
* Average of recency: 46 days - the lowest recency
* Average of accepted campaign: 2 times - the highest accepted campaign
* Average of Amount of Purchases: 1510 - the highest amount of purchases

## Business Insights
* Cluster 0 is our intermediate loyal customers since this cluster has intermediate amount of purchases (not as high as Cluster 2, but not too low than Cluster 1). It has the highest population and oldest customers. They mostly have teenagers at home, thus they are likely to manage their financial spending by by not too often for shopping. It is proved that they have the highest recency (number of days since customer's last purchase), which means they are rarely to shop in our store. Therefore, we should give them special flash sales or promotions to attract them spending more as soon as possible and make sure that they always come to us periodically
* Cluster 1 is our lost customers. It has the lowest amount of purchases and high average of recency which achieves small difference compared by Cluster 0. We should increase our quality services for those in this cluster by evaluating from their complaints. In addition, they mostly have kids at home. Therefore, we can try to attract them by giving them special prices for products for kid's daily needs
* Cluster 2 is our best loyal customers. It is proved by its lowest recency and the highest amount of purchases. They mostly accepted our campaigns and spent more into their daily needs. We need to appreciate them by giving them certain promotions, e.g. discount, cashback vouchers, birthday gift(s). They do not have either kid or teenager at home. Therefore, certain promotion in products for their daily needs is useful to maintain their loyalty to our market

# **Summary of Key Findings**
* This dataset has 24 missing values which have been removed for analysis. Furthermore, Inefficient columns, such as `Z_CostContact` and `Z_Revenue` also be removed
* Wines and Meat are considered as our most favorite products, while
Sweet and Fruits are our less amount of products
* Our best accepted campaign is on our last campaign
In contrast, Campaign 2 has been our lowest accepted campaign
* Most of customers are likely to purchase directly from our store, but not less customers has purchased from our website. Therefore, website development to attract customers should be gradually improved
* The highest growth comes from July to August in 2012 before went down. 2013 has similar customer enrollment as August 2012 on October. 2013 likely can be called as our stable year in customer enrollment.The drop of our customer enrollment happens in 2014: our customer enrollment went down from May to June which has been the lowest customer enrollment throughout the dataset. We should find strategy to increase customer enrollment by cluster analysis
* I created new columns for cluster analysis, such as accepted campaign, amount of products, and number of purchases 
* I used K-Means Clustering for Cluster Analysis and Principle Component Analysis (PCA) as my dimensionality reduction algorithm to visualize my clustering result from K-Means (reducted into 2-D)
* K-Means Clustering performs its best if: data's distribution is not skewed and be standardized. I performed feature selection to avoid redundant features, handling skewed features by performing log transformation, and feature scaling by standardization
* The best number of clusters has been achieved through the highest Silhouette Score: 3 clusters of customers
* Cluster 0 is our intermediate loyal customers, Cluster 1 is our lost customers, and Cluster 2 is our best loyal customers
* Recommendation: Cluster 0 should be given special flash sales or promotions. Cluster 1 should be evaluated by checking their complaint forms and given special prices for kid's daily needs. Cluster 2 should be maintained by giving them certain promotions as appreciation 

# **Suggestions**
* Attempt for another clustering algorithm, e.g. Fuzzy C-Means Clustering
* Perform different steps of data cleaning and or feature engineering to get variant results