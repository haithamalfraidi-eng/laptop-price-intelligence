import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "D:/Downloads/laptops_test.csv"

df = pd.read_csv(file, header=0)
pd.options.display.float_format = '{:.2f}'.format

#print("Shape:" ,df.shape)

#print("\n columns:" ,df.columns)

#print("\n missing Data:\n",df.isnull().sum())
#print("\n duplicated Data=",df.duplicated().sum())
#print(df.shape)

df=df.drop_duplicates().reset_index(drop=True)

df.columns=df.columns.str.strip()

#print(df["Operating System"].isin(['No OS']).sum())

#print(df["Operating System Version"].isnull().sum())

df.loc[df["Operating System"] == "No OS", "Operating System Version"] = "N/A"
df.loc[df["Operating System"].isin(["Linux","Chrome OS","Android"])]


df.loc[df["Operating System"].isin(["Linux", "Chrome OS", "Android"]),"Operating System Version"] = "Not Specified"
print(df["Operating System Version"].isnull().sum())
#for index,row in df.iterrows():
    #if  pd.isna(row["Operating System Version"]) and row["Operating System"]!="No OS":
       # print(row["Operating System"])
        #print(row["Operating System Version"])


df["Screen Size"]=df["Screen Size"].str.replace('"',"")
df["Screen Size"]=df["Screen Size"].astype("float")

df["Weight"]=df["Weight"].str.replace("kg","")

df.loc[df["Weight"]=="4s","Weight"]=np.nan
avgWeight = df["Weight"].astype("float").mean()
df["Weight"]=df["Weight"].replace(np.nan,avgWeight)
df["Weight"]=df["Weight"].astype("float")

df["RAM"]=df["RAM"].str.replace("GB","")
df["RAM"]=df["RAM"].astype("float")

df["Storage"]=df["Storage"].str.replace("1TB","1000GB")
df["Storage"]=df["Storage"].str.replace("2TB","1000GB")
#print(df["Price"].isna().sum())
#print(df["Storage"].value_counts())

#print(df[["Price","RAM","Screen Size","Weight"]].describe().loc[['mean','min','max','50%']])

highest_price_index = df["Price"].idxmax()

print(df.loc[highest_price_index,["Manufacturer", "Model Name", "Category", "RAM","Storage", "GPU", "Weight", "Price"]])

print(df["Manufacturer"].value_counts())


print(df.groupby("Category")['Price'].agg('mean').sort_values(ascending=False).round(3))
print(df.groupby("Manufacturer")['Price'].agg('mean').sort_values(ascending=False).round(2))
RamPriceAvg=df.groupby("RAM")['Price'].agg('mean').sort_values(ascending=False).round(2)
print(df.groupby("RAM")['Price'].agg('mean').sort_values(ascending=False).round(2))

print(df.groupby("RAM").size())
CatWeiAvg=df.groupby("Category")["Weight"].agg('mean').sort_values(ascending=False)
print(CatWeiAvg)

CatWeiAvg.plot(kind="bar")
plt.ylabel("Average Weight (kg)")

df.plot(kind="scatter", x="RAM", y="Price")

plt.ylabel("Price")
plt.xlabel("RAM (GB)")
plt.title("Price VS RAM")

plt.show()
highest_price_index = df["Price"].idxmax()

df.to_csv("Laptop.CSV", index=False)




