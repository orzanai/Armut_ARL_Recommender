
#########################
# Business Problem
#########################

# Armut, which is Turkey's largest online service platform, brings together service providers and those who want to receive services.
# It enables easy access to services such as cleaning, renovation, and transportation with just a few touches on a computer or smartphone.
# By using the dataset that includes the users who received services and the categories of services they received, it is aimed to
# create a product recommendation system using Association Rule Learning.


#########################
# Dataset
#########################
# The dataset consists of the services that customers have received and their categories.
# It includes the date and time information for each received service.

# UserId: Customer ID
# ServiceId: Anonymized services belonging to each category. (Example: Under the cleaning category, sofa washing service)
# A ServiceId can be found under different categories and represents different services under different categories
# (Example: The service with CategoryId 7 and ServiceId 4 is radiator cleaning, while the service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories (Example : Cleaning, logistic, renovation categories)
# CreateDate: Date of purchase of the service




#########################
# Data Processing
#########################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv("/Case Studies/ArmutARL-221114-234936/armut_data.csv")


# Create a new variable representing the services by combining ServiceID and CategoryID with "_"

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# The dataset consists of the dates and times when services are purchased, there is no basket definition (such as invoice, etc.).
# To be able to apply Association Rule Learning, a basket definition (such as an invoice) must be created.
# Here, the basket definition is the services each customer received monthly.
# For example; services 9_4 and 46_4 received by customer with ID 7256 in August 2017 represent one basket,
# and services 9_4 and 38_4 received in October 2017 represent another basket.
# The baskets need to be identified with a unique ID. For this, we first create a new date variable containing only year and month.
# Then, we combine the UserID and the newly created date variable with "_" and assign it to a new variable called ID.

df["Monthly_Date"] = pd.to_datetime(df["CreateDate"]).dt.to_period("M")

df["BasketId"] = df["UserId"].astype(str) + "_" + df["Monthly_Date"].astype(str)

#########################
# Association Rules
#########################

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketId
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


basked_service_matrix = df.groupby(["BasketId", "Service"])["Service"].\
                        count().unstack().\
                        fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Create Association Rules

freq_results = apriori(basked_service_matrix, min_support= 0.01, low_memory=False, use_colnames=True)

support_rules = association_rules(freq_results, metric="support", min_threshold=0.01).sort_values("lift", ascending=False)

# confidence_rules = association_rules(freq_results, metric="confidence", min_threshold=0.1).sort_values("lift", ascending=False)

# arl_recommender function to make service recommendations for a user who has last received the 2_0 service

rules = support_rules[(support_rules["support"] > 0.005) &
                      (support_rules["confidence"] > 0.1) &
                      (support_rules["lift"] > 2)]

def arl_recommender(rules, service, rec_count = 1):
    recommendation_list = []
    for i, p in enumerate(rules["antecedents"]):
        if service in list(p) and len(list(p)) == 1:
            recommendation_list.append(list(rules.iloc[i]["consequents"]))
    return recommendation_list[:rec_count]


arl_recommender(rules, "2_0", 3)


