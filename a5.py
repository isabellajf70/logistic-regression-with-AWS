import numpy as np
import re
import math


##########################
##########################

# Task 1: data preparation


# load up all of the documents in the corpus
#corpus = sc.textFile("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")
#corpus = sc.textFile("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")
#corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")



# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
# keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2 : x.index('</doc>')]))
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2 : ]))


# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top(20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map(lambda x: (topWords[x][0], x))




# give us the frequency position of the words “applicant”, “and”, “attack”,
# “protein”, and “car”.
task1test = ['applicant', 'and', 'attack', 'protein', 'car']
for i in task1test:
	cur = dictionary.lookup(task1test)
	if cur and cur[0] >= 0 and cur[0] <= 19999:
		print(cur[0])
	else:
		print(-1)



##########################
##########################

# Task 2: learning

# sub-task 1:
# convert each of document to TF-IDF
############
# reference: the section below is modified from the solution of A4 provided by Professor Jermaine on Piazza for helping students complete A5
###########
allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allDictionaryWords = dictionary.join (allWords)
justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
#regex = re.compile('/.*?/')
#allDictionaryWordsInEachDocWithNewsgroup = allDictionaryWordsInEachDoc.map (lambda x: ((x[0], regex.search(x[0]).group (0)), x[1]))
def buildArray (listOfIndices):
        returnVal = np.zeros (20000)
        for index in listOfIndices:
                returnVal[index] = returnVal[index] + 1
        return returnVal
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map (lambda x: (x[0], buildArray (x[1])))
t2tf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.divide(x[1], np.sum(x[1]))))
zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.where(x[1] > 0, 1, 0)))
dfArray = zeroOrOne.reduce (lambda x1, x2: (("", np.add (x1[1], x2[1]))))[1]
totaldoc = t2tf .count()
multiplier = np.full(20000, totaldoc)
idfArray = np.log (np.divide (multiplier, dfArray))
allDocsAsNumpyArrays = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply (x[1], idfArray)))



# sub-task 2:
# For better accuracy, center and normalize data

# mean
tfsum = allDocsAsNumpyArrays.values().sum()
tfcount = allDocsAsNumpyArrays.values().count()
mean = tfsum / tfcount

# standard deviation
sd = allDocsAsNumpyArrays.values().sampleStdev()

# normalize
sd[sd == 0] = 1
normalizedtf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.divide(np.subtract(x[1], mean), sd)))

# following the instruction on piazza:
# " The labels are 0/1, where 0 is false and 1 is true, not 1/-1, since 1/-1 doesn't work for Bernoulli Distribution"
regex = re.compile('AU')
doctflabel = normalizedtf.map(lambda x: (x[0], (x[1], (1 if bool(regex.match(x[0][0:2])) else 0))))



# sub-task 3:
# use gradient descent algorithm to learn logistic regression model


learningrate = 0.001
penalty = 0.001
r = np.zeros((20000,))


while abs(llhnew - llhold) >= 0.0000001:


	# (1)  objective function

	# compute hypothesis: x * regression coefficients
	hypothesis = doctflabel.map(lambda x: (x[0], (x[1][0], x[1][1], np.dot(x[1][0], r))))

	# compute cost function
	cost = hypothesis.map(lambda x: (x[0], x[1][1] * x[1][2] - np.log(1 + np.exp(x[[1][2]))))



	# (2)  adding penalty term to objective function: using l2 regularization
	penaltyterm = penalty * np.sum(np.square(r))
	llhnew = cost.reduce(lambda x1, x2: (('', x1[1] + x2[1])))[1] - penaltyterm
	llhnew = llhnew / totaldoc




	# (3) update r

	#  gradient of complete function we obtained above
	gradient = hypothesis.map(lambda x: (x[0], (-1) * x[1][0] * x[1][1] + x[1][0] * (np.exp(x[1][2])) / ( 1 + np.exp(x[1][2]))))

	# sum up
	gradient = gradient.reduce(lambda x1, x2: (('', np.add(x1[1], x2[1]))))[1]

	# update r
	r = r - learningrate * ( gradient + 2 * penalty * r ) / totaldoc


	# (4) adjust learning rate

	if llhnew > llhold:
		learningrate *= 0.5
	else:
		learningrate *= 1.05

	llhold = llhnew



# sub-task 4:
# getting the result: fifty words with the largest regression coefficients
dictionary1 = dictionary.map(lambda x: (x[1], x[0]))
for i in r.argsort()[-50:]:
	word = dictionary1.lookup(i)
	print(word[0])



##########################
##########################


# Task 3: Evaluation of the learned model

#######
# using small data set, use the same values we obtained from above, the same normalization parameters
#######

hypothesis1 = doctflabel.map(lambda x: (x[0], (x[1][0], x[1][1], 1 / ( 1+ math.exp(np.sum((np.dot(x[1][0], r))))))))
hypothesis1 = doctflabel.map(lambda x: (x[0], (x[1][1], x[1][2])))

# actual positive
actualpositive = hypothesis1.map(lambda x: (x[0], 1 if (x[1][0] == 1) else 0)).values().sum()

# cut off value
cutoff = 0.20

# true positive and claimed positive
truepositive = hypothesis1.map(lambda x: (x[0], 1 if(x[1][0] == 1 and x[1][1] > cutoff) else 0)).values().sum()
claimedpositive = hypothesis1.map(lambda x: (x[0], 1 if(x[1][0] == 0 and x[1][1] > cutoff) else 0)).values().sum()
precision = (trupositive * 1.0) / (truepositive + claimedpositive)
recall = (truepositive * 1.0) / actualpositive

# calculating f1 score
f1score = 2 * precision * recall / (precision + recall)
print(f1score)



# look at actual context of the three false positives

k = 3
allfalsepositive = hypothesis1.map(lambda x: (x[0], 1 if(x[1][0] == 0 and x[1][1] > cutoff) else 0))
threefp = allfalsepositive.top(k, lambda x: x[1])

ind = 0
while ind < k:
	keyAndText.lookup(threefp[ind][0])
	ind += 1





