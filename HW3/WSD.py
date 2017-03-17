# Assignment 3
# Robin Mehta
# robimeht

from __future__ import division, print_function
import numpy as np 
import sys
import string
import operator
from collections import defaultdict
from string import punctuation
import math

#method found online (https://www.quora.com/How-do-I-remove-punctuation-from-a-Python-string)
def strip_punctuation(s):
	return ''.join(c for c in s if c not in punctuation)

#regex in this method found online (http://stackoverflow.com/questions/3368969/find-string-between-two-substrings)
def findMiddleText(start, end, line):
	foundWord = ""
	if line.find(start):
		startAndword = line[line.find(start):line.rfind(end)]
		foundWord = startAndword[len(start):]
		return foundWord

def parse_file(path):
	with open(path,'r') as data:
		ambiguousWord = ""
		numInstances = 0
		for line in data.read().split('\n'):
			if line.startswith("<instance id="):
				numInstances += 1
			start = '<head>'
			end = '</head>'
			if ambiguousWord == "":
				ambiguousWord = findMiddleText(start, end, line)

	remainderInstances = numInstances % 5
	numInstances += (5 - remainderInstances)
	numInstancesPerFold = numInstances / 5
	return numInstances, remainderInstances, numInstancesPerFold, ambiguousWord
# training data = numInstancesPerFold + 1 until end of file
#test data = 1st instance up until numInstancesPerFold (first fold of file)

def parse_data(path, numInstances, remainderInstances, numInstancesPerFold, ambiguousWord, numFold):
	testOutFileName = ambiguousWord + 'testData.out'
	trainOutFileName = ambiguousWord + 'trainData.out'
	testData = open(testOutFileName, 'w')
	trainData = open(trainOutFileName, 'w')

	startInstance = numFold * numInstancesPerFold
	endInstance = startInstance + numInstancesPerFold

	with open(path,'r') as data:
		countInstances = 0
		for paragraph in data.read().split("\n\n"):
			if paragraph.startswith("<instance id="):
				countInstances += 1
				if countInstances >= startInstance and countInstances < endInstance:
					testData.write(paragraph)
					testData.write("\n\n")
				else:
					trainData.write(paragraph)
					trainData.write("\n\n")
	testData.close()
	trainData.close()
	return testOutFileName, trainOutFileName

def parse_training_data(trainOutFileName):
	# get all types of senses
	# get # of each type of sense, store in dict of vectors
	# store all context words for each sense in a vector from dict
	# get unique words 
	sensesDict = dict() #dict where key=sense, value=list of all context words
	numSensesDict = dict() #dict where key=sense, value=# of times it appears in training data
	with open(trainOutFileName, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<answer"):
					start = "senseid=\""
					end = "\"/>"
					sense = findMiddleText(start, end, line)
					if not sense in sensesDict:
						sensesDict[sense] = list()
						numSensesDict[sense] = 0
					else:
						numSensesDict[sense] += 1
				if line.startswith("<context>"):
					text = findMiddleText("<context>", "</context>", paragraph)
					for word in text.split(" "):
						word = word.strip('\n')
						word = word.lower()
						word.strip()
						word = strip_punctuation(word)
						if word != "":
							sensesDict[sense].append(word)
	uniqueSensesDict = dict() #dict where key=sense, value=list of unique context words
	countUniqueSensesDict = dict() #dict where key=sense, value=# of unique context words
	for sense in sensesDict:
		countUniqueSensesDict[sense] = 0
		uniqueSensesDict[sense] = list()
		for word in sensesDict[sense]:
			if not word in uniqueSensesDict[sense]:
				uniqueSensesDict[sense].append(word)
		countUniqueSensesDict[sense] = len(uniqueSensesDict[sense])
	return sensesDict, numSensesDict, uniqueSensesDict, countUniqueSensesDict

def probabilitiesOfSenses(numSensesDict):
	probSensesDict = dict()
	totalNumSenses = 0
	for sense in numSensesDict:
		totalNumSenses += numSensesDict[sense]
	for sense in numSensesDict:
		probSensesDict[sense] = numSensesDict[sense] / totalNumSenses
	return probSensesDict

def extractTestData(testOutFileName):
	wordsAndTestIDsDict = dict() #dict where key=instanceID, value=list of context words (TEST DATA)
	with open(testOutFileName, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<instance"):
					start = "=\""
					end = "\" docsrc ="
					ID = findMiddleText(start, end, line)
					if not ID in wordsAndTestIDsDict:
						wordsAndTestIDsDict[ID] = list()
				if line.startswith("<context>"):
					text = findMiddleText("<context>", "</context>", paragraph)
					for word in text.split(" "):
						word = word.strip('\n')
						word = word.lower()
						word.strip()
						word = strip_punctuation(word)
						if word != "": #and word not in stopwords.words()
							wordsAndTestIDsDict[ID].append(word)
	return wordsAndTestIDsDict

#iterates a dict of scores, IDs, and senses, finds argmax and best sense for ID
def keyOfMaxValue(scoreDict):
	solvedDict = dict()
	argMax = -999999999999
	labelSense = ""
	for ID in scoreDict:
		for scoreList in scoreDict[ID]:
			for sense in scoreList:
				score = scoreList[sense]
				if score > argMax:
					argMax = score
					labelSense = sense
		solvedDict[ID] = labelSense
	return solvedDict

def naiveBayesAddOneSmoothing(wordsAndTestIDsDict, sensesDict, numSensesDict, uniqueSensesDict, probSensesDict, outFile):
	scoreDict = dict()
	solvedDict = dict()
	outFile.write("Naive Bayes Add-One Smoothing")
	outFile.write("\n\n")
	for ID in wordsAndTestIDsDict:
		for sense in sensesDict:
			total = 0
			for word in wordsAndTestIDsDict[ID]:
				numerator = sensesDict[sense].count(word) + 1
				denominator = numSensesDict[sense] + len(uniqueSensesDict[sense])
				total += math.log((numerator / denominator), 2)
			score = total * math.log(probSensesDict[sense], 2)
			if ID not in scoreDict:
				scoreDict[ID] = list()
			scoreDict[ID].append({sense:score})
	solvedDict = keyOfMaxValue(scoreDict)
	for ID in solvedDict:
		outFile.write(str(ID))
		outFile.write(" ")
		outFile.write(str(solvedDict[ID]))
		outFile.write("\n")
	return solvedDict

def calculateAccuracies(solvedDict, testOutFileName, outFile):
	solutionsDict = dict() #dict where key=instanceID, value=list of context words (TEST DATA)
	with open(testOutFileName, 'r') as data:
		for paragraph in data.read().split("\n\n"):
			for line in paragraph.split("\n"):
				if line.startswith("<instance"):
					start = "=\""
					end = "\" docsrc ="
					ID = findMiddleText(start, end, line)
				if line.startswith("<answer"):
					start = "senseid=\""
					end = "\"/>"
					sense = findMiddleText(start, end, line)
					solutionsDict[ID] = sense
	numCorrect = 0
	numTotal = 0
	for key in solvedDict:
		if solvedDict[key] == solutionsDict[key]:
			numCorrect += 1
		numTotal += 1
	outFile.write("\n")
	outFile.write("Accuracy: ")
	outFile.write(str((numCorrect/numTotal)*100))
	outFile.write("%")
	outFile.write("\n\n")
	return numCorrect/numTotal

def main():
	path = sys.argv[1]
	numInstances, remainderInstances, numInstancesPerFold, ambiguousWord = parse_file(path)
	outputFileName = ambiguousWord+".wsd.out"
	outFile = open(outputFileName, 'w')
	accuraciesDict = dict() #key=fold, val=accuracy
	for x in range(0, 5): #folds
		outFile.write("Fold ")
		outFile.write(str(x))
		outFile.write("\n")

		testOutFileName, trainOutFileName = parse_data(path, numInstances, remainderInstances, numInstancesPerFold, ambiguousWord, x) #writes to new test and training files
		sensesDict, numSensesDict, uniqueSensesDict, countUniqueSensesDict = parse_training_data(trainOutFileName) #parse training file
		probSensesDict = probabilitiesOfSenses(numSensesDict) #probabilities of each sense
		wordsAndTestIDsDict = extractTestData(testOutFileName)

		solvedDict = naiveBayesAddOneSmoothing(wordsAndTestIDsDict, sensesDict, numSensesDict, uniqueSensesDict, probSensesDict, outFile)
		accuracy = calculateAccuracies(solvedDict, testOutFileName, outFile)
		accuraciesDict[x] = accuracy
	avgAccuracy = 0
	for fold in accuraciesDict:
		avgAccuracy += accuraciesDict[fold]
	avgAccuracy = avgAccuracy/5

	outFile.write("\n")
	outFile.write("Average Accuracy: ")
	outFile.write(str(avgAccuracy*100))
	outFile.write("%\n")
	outFile.close()

if __name__ == '__main__':
	main()