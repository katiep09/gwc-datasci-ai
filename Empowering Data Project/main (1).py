# import libraries
import pandas as pd 
import matplotlib.pyplot as plt
import GWCutilities as util
# read CSV into variable using pandas
lwd=pd.read_csv("livwell175.csv")


# Part 1: Printing out background information & proposal
print("Guatemala is a country in central America with a population of 17.8 million people. 1.9 million of these people are in between the ages of 15-20, around half of which are girls. \n")
print("Benin is a country in west Africa with a population of 13.3 million people. In Benin, 1.4 million people are in between the ages of 15-20, and like Guatemala, half of them are also girls. \n")
print("At these ages, many people choose to complete high school and pursue a higher education by going to college or university. \n")
print("But in Guatemala and Benin, the average age of marriage for women has consistently fallen into this range! \n")


input("Press return to continue\n")

print("While exploring the LivWell175 dataset, I discovered that for both countries, as the marriage age for women decreased, so did their average years of schooling. \n\n")
print("Based on this correlation, my proposed research question is: \n\n")
print(" \"Why do the average years of schooling for women decrease as the average age of marriage goes down? How can we ensure that the average years of schooling for women continue to increase over time?\"")



# Part 2: Filtering Data
# creating dataset for Benin
beninRows = lwd["country_name"] == "Benin"
beninData = lwd.loc[beninRows]
# creating dataset for Guatemala
guatemalaRows = lwd["country_name"] == "Guatemala"
guatemalaData = lwd.loc[guatemalaRows]


#Part 3: Graphing the Results
# create a scatter plot
plt.scatter(beninData["DM_age_marr_mean"],beninData["ED_educ_years_mean"],label="Benin",color="purple")
plt.scatter(guatemalaData["DM_age_marr_mean"],guatemalaData["ED_educ_years_mean"],label="Guatemala",color="gold")
# add a title to the plot
plt.title("Average Age of Marriage vs Average Years of Schooling")
# label the x-axis
plt.xlabel("Average Age of Marriage")
# label the y-axis
plt.ylabel("Average Years of Schooling")
# display labels
plt.legend()
# show the graph
plt.show()