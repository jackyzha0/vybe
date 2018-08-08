import glob

import csv

import pandas as pd

import librosa

def get_data(): #dir

	files = {}

# import glob, os
# os.chdir("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\data")
# for file in glob.glob("*.wav"):
#     print(file)

# import csv
# 	reader = csv.reader(open("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv", 'r'))
# 	data_label = {}
# 	for a, h in reader:
# 		# k, v = row
# 		data_label[a] = h

	with open('C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv', mode='r') as infile:
		reader = csv.reader(infile)
		with open('coors_new.csv', mode='w') as outfile:
			writer = csv.writer(outfile)
			mydict = {rows[0]:rows[7] for rows in reader}

	for x in mydict.keys():
		value = mydict[x]
		if value == 'siren':
			mydict[x] = "0"

	for y in mydict.keys():
		value = mydict[y]
		if value == 'car_horn':
			mydict[y] = "1"

	# mydict["siren"] = 0
	# mydict["car_horn"] = 1

	# mydict.update({"siren": 0, "car_horn" : 1})

	return mydict

	# print(mydict)

def get_label(): #filepath

	listing_stuff = []

	mydict_2 = get_data()

	for x in mydict_2:

		listing_stuff.append(mydict_2[x])

		# print(mydict_2[x])

	return(listing_stuff)


def next_minibatch(): #

	y, sr = librosa.load("C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\audio\\fold1\\7061-6-0-0.wav") #, offset = 30, duration = 5
	mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13)

	x = []

	for i in mfccs:
		x.append(i)

	# labels = get_label()

	# new_dict = {mfccs : labels}

	"""
	Return dictionary of next minibatch in this format
	(arr of mfcc) : label
	"""

	# from io import StringIO
	# s = io.StringIO(text)
	# with open('C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv', 'w') as f:
	# with open("C:\\Users\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\test_icle.csv", 'w') as f:
	#     for o in x:
	#         f.write(o)

 #    import csv
 #    with open("C:\\Users\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\test_icle.csv") as csvfile:
 #    reader = csv.DictReader(csvfile)
 #    for row in reader:
 #             print(row['first_name'], row['last_name'])

	# with open('coors_new.csv', mode='w') as outfile:
	# 	writer = csv.writer(outfile)
	# 	mydict = {rows[0]:rows[2] for rows in reader}

	# with open('C:\\Users\\Imam\\Documents\\PROGRAMMING PROJECTS\\LaunchX\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv', mode='r') as infile:
	# 	reader = csv.reader(infile)
	# 	with open('coors_new.csv', mode='w') as outfile:
	# 		writer = csv.writer(outfile)
	# 		mydict = {rows[0]:rows[7] for rows in reader}


# 	print(mydict)
# next_minibatch()

	res = {}
	values = get_label()
	keys = mfccs 
	for i in range(len(values)):
		print(keys[i])
		print(values[i])
		print(keys[i].type)
		print(values[i].type)
		res[keys[i]] = values[i]

	print(res)

next_minibatch()