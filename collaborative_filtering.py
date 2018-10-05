import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def load_movie_names():
    movieNames = {}
    with open("ml-1m/movies.dat") as f:
        for line in f:
            fields = line.split("::")
            movieNames[int(fields[0])] = fields[1]
    return movieNames

def make_pairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filter_duplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def pearson_correlation_based_similarity(ratingPairs):
    numPairs = 0
    sum_x = sum_y = 0
    for ratingX, ratingY in ratingPairs:
        sum_x+=ratingX
        sum_y+=ratingY
        numPairs += 1
    avg_x=sum_x/float(numPairs)
    avg_y=sum_y/float(numPairs)
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xy += (ratingX - avg_x) * (ratingY - avg_y)
        sum_xx += (ratingX - avg_x) * (ratingX - avg_x)
        sum_yy += (ratingY - avg_y) * (ratingY - avg_y)

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)

print("\nLoading movie names...")
nameDict = load_movie_names()

data = sc.textFile("ml-1m//ratings.dat")

# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split("::")).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))
ratingsPartitioned = ratings.partitionBy(100)

# Filter out bad ratings (i.e., ratings less than 3)
filteredRatings = ratingsPartitioned.filter(lambda l: l[1][1] >= 3)

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = filteredRatings.join(filteredRatings)

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filter_duplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(make_pairs)

# Now collect all ratings for each movie pair 
moviePairRatings = moviePairs.groupByKey()

# Compute similarities.
moviePairSimilarities = moviePairRatings.mapValues(pearson_correlation_based_similarity).persist()

# Filter out movies with strength <=50
moviePairSimilaritiesFiltered = moviePairSimilarities.filter(lambda pairSim: pairSim[1][1] > 50)

# Extract similarities for the movie we care about that are "good".
if (len(sys.argv) > 1):

    movieID = int(sys.argv[1])

    # Results for required movieID
    filteredResults = moviePairSimilaritiesFiltered.filter(lambda pairSim: (pairSim[0][0] == movieID or pairSim[0][1] == movieID) )

    # Sort by quality score
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))
