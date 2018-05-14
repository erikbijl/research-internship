import numpy as np
import json
import pandas as pd
from scipy import spatial

def getDistanceTable():

	filename = 'data.json'

	if filename:
	    with open(filename, 'r') as f:	
	        data = json.load(f)

	nrof_images = len(data["faces"])
	matrix = np.zeros((nrof_images, nrof_images))

	for i in range(nrof_images):
		for j in range(nrof_images):	
			dist = np.sqrt(np.sum(np.square(np.subtract(data["faces"][i]["embedding"],data["faces"][j]["embedding"]))))
			matrix[i][j] = dist
	d = []
	for face in data["faces"]:
	    d.append({'file': face["file"], 'embedding': face["embedding"]})

	column_names = []
	for num in map(str, range(128)):
	    column_names.append(num)

	df = pd.DataFrame(d)

	df[['dimensions']] = pd.DataFrame(df.embedding.values.tolist(), index= df.index)
	df[column_names] = pd.DataFrame(df.dimensions.values.tolist(), index= df.index)
	df = df.drop(['embedding', 'dimensions'], axis=1)

	distances = spatial.distance.cdist(df.iloc[:,1:], df.iloc[:,1:], metric='euclidean')
	return [data, distances]