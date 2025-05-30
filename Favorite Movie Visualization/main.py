import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

movieData = pd.read_csv('./rotten_tomatoes_movies.csv')
favMovie = "McFarland USA"

print("My favorite movie is "+favMovie)


#print(movieData.head())
#print(movieData["movie_title"])


print("\nThe data for my favorite movie is:\n")
#Create a new variable to store your favorite movie information
favMovieBooleanList = movieData["movie_title"] == favMovie
#print(favMovieBooleanList)

favMovieData = movieData.loc[favMovieBooleanList]
print(favMovieData)


print("\n\n")

#Create a new variable to store a new data set with a certain genre
dramaMovieBooleanList = movieData["genres"].str.contains("Drama")

dramaMovieData = movieData.loc[dramaMovieBooleanList]


numOfMovies = dramaMovieData.shape[0]

print("We will be comparing " + favMovie +
      " to other movies under the genre Drama in the data set.\n")
print("There are " + str(numOfMovies) + " movies under the category Drama.")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
input("Press enter to see more information about how " + favMovie +
      " compares to other movies in this genre.\n")

#find min
min = dramaMovieData["audience_rating"].min()
print("The min audience rating of the data set is: " + str(min))
print(favMovie + " is rated 84 points higher than the lowest rated movie.")
print()

#find max
max = dramaMovieData["audience_rating"].max()
print("The max audience rating of the data set is: " + str(max))
print(favMovie + " is rated 12 points lower than the highest rated movie.")
print()

#find mean
mean = dramaMovieData["audience_rating"].mean()
print("The mean audience rating of the data set is: " + str(mean))
print(favMovie + " is higher than the mean movie rating.")

#find median
median = dramaMovieData["audience_rating"].median()
print("The median audience rating of the data set is: " + str(median))
print(favMovie + " is higher than the median movie rating.")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
input("Press enter to see data visualizations.\n")

#Part 6 Create graphs
#Create histogram
plt.hist(dramaMovieData["audience_rating"], range = (0,100), bins = 20)

#Adds labels and adjusts histogram
plt.grid(True)
plt.title("Audience Ratings of Drama Movies Histogram")
plt.xlabel("Audience Ratings")
plt.ylabel("Number of Movies")

#Prints interpretation of histogram
print(
  "According to the histogram, the rating range with the greatest number of drama movies is 75-80. The distribution of the graph is skewed left."
)
print("Close the graph by pressing the 'X' in the top right corner.")
print()

#Show histogram
plt.show()

#Create scatterplot
plt.scatter(data = dramaMovieData, x = "audience_rating", y = "critic_rating", label = "Drama Movies")

#Adds labels and adjusts scatterplot
plt.grid(True)
plt.title("Audience Rating vs Critic Rating")
plt.xlabel("Audience Rating")
plt.ylabel("Critic Rating")
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.scatter(data = favMovieData, x = "audience_rating", y = "critic_rating", label = favMovie)
plt.legend
#Prints interpretation of scatterplot
print(
  "According to the scatter plot, there is a positive correlation between audience ratings and critic ratings."
)
print(favMovie + " follows the scatter plot's overall pattern. The critic rating is slightly lower than the audience rating.")

print("Close the graph by pressing the 'X' in the top right corner.")

#Show scatterplot
plt.show()

print("\nThank you for reading through my data analysis!")
