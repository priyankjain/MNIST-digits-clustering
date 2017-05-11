###############################################################################
#Author: Priyank Jain (@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import csv
from random import shuffle
import scipy
import matplotlib.cm as cm
from sklearn import metrics
import scipy
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import gc

dists = None
class selector(object):
	def __init__(self, file):
		if not os.path.exists(file):
			raise FileNotFoundError(file + ' does not exist')
		self.file = file
		
	def read(self, percentage):
		self.percentage = percentage
		fin = open(self.file, 'r')
		csvreader = csv.reader(fin)
		self.data = []
		for row in csvreader:
			row = [float(r) for r in row]
			self.data.append({'class': int(row[1]), 'features': row[2:]})
		divider = int(percentage*len(self.data)/100)
		fin.close()
		return self.data[:divider], self.data[divider:]	

	def filter(self, labels, data):
		newData = []
		for datum in data:
			if datum['class'] in labels:
				newData.append(datum)
		return newData

def explore():
	sel = selector('digits-raw.csv')
	rawData, _ = sel.read(100)
	onePtPerClass = dict()
	numClasses = 0
	for datum in rawData:
		if datum['class'] in onePtPerClass:
			continue
		onePtPerClass[datum['class']] = datum
		numClasses += 1
		if numClasses == 10:
			break
	sel = selector('digits-embedding.csv')
	twoDdata, _ = sel.read(5)
	for key, pt in onePtPerClass.items():
		mat = np.array(pt['features']).reshape(28, 28)
		plotGreyscale(mat, str(key))
	title = 'Visualization of class labels'
	plotScatterPlot(twoDdata, title)

def NMI(data):
	counts = dict()
	for datum in data:
		counts.setdefault(datum['class'], dict())
		counts[datum['class']].setdefault(datum['cluster'], 0)
		counts[datum['class']][datum['cluster']] += 1
	clsCounts = dict()
	clusterCounts = dict()
	for cls, dct in counts.items():
		clsCounts.setdefault(cls, 0)
		for cluster, cnt in dct.items():
			clusterCounts.setdefault(cluster, 0)
			clusterCounts[cluster] += cnt
			clsCounts[cls] += cnt
	clsProbs = {idx: x/sum(clsCounts.values()) for idx, x in clsCounts.items()}
	clusterProbs = {idx: x/sum(clusterCounts.values()) for idx, x in clusterCounts.items()}
	clsH = sum([-x*math.log(x) for x in clsProbs.values()])
	clusterH = sum([-x*math.log(x) for x in clusterProbs.values()])
	num = 0
	n = sum(clsCounts.values())
	for cls, dct in counts.items():
		for cluster, cnt in dct.items():
			prob = cnt/n
			num += prob * math.log(prob/(clsProbs[cls]*clusterProbs[cluster]))
	nmi = num/math.sqrt(clsH * clusterH)
	#print('My MI', num)
	#print('Scipy MI', metrics.mutual_info_score([datum['class'] for datum in data], [datum['cluster'] for datum in data]))
	#print('Class entropy', clsH)
	#print('Scipy class entropy', scipy.stats.entropy(list(clsCounts.values())))
	#print('Cluster entropy', clusterH)
	#print('Scipy cluster entropy', scipy.stats.entropy(list(clusterCounts.values())))
	text_nmi = num/(clsH + clusterH)
	return text_nmi

def SC(data):
	coeffs = []
	dcopy = copy.deepcopy(data)
	counts = dict()
	centroids = {i:list() for i in set([datum['cluster'] for datum in data])}
	oneFeature = dcopy[0]['features']
	for i, croid in centroids.items():
		centroids[i] = [0 for x in oneFeature]
		counts.setdefault(i, 0)
	for datum in data:
		centroids[datum['cluster']] = [x+y for x,y in zip(\
			centroids[datum['cluster']], datum['features'])]
		counts[datum['cluster']] += 1
	for i, croid in centroids.items():
		centroids[i] = [x/counts[i] for x in croid]
	for idx1, copyDatum in enumerate(dcopy):
		min_distance = None
		closest_centroid = None
		for key, value in centroids.items():
			if key == copyDatum['cluster']:
				continue
			else:
				curDist = distance(copyDatum['features'], centroids[key])
				if closest_centroid is None or curDist < min_distance:
					min_distance = curDist
					closest_centroid = key
		a = 0
		b = 0
		wc_count = 0
		bc_count = 0
		for idx2, datum in enumerate(data):
			if datum['id'] == copyDatum['id']:
				continue
			elif copyDatum['cluster'] == datum['cluster']:
				#hashKey = tuple(sorted((copyDatum['id'], datum['id'])))
				#if hashKey not in dists:
				#	dists[hashKey] = \
				#	distance(copyDatum['features'], datum['features'])
				a += dists[idx1][idx2]
				wc_count += 1					
			elif closest_centroid == datum['cluster']: 
				#hashKey = tuple(sorted((copyDatum['id'], datum['id'])))
				#if hashKey not in dists:
				#	dists[hashKey] = \
				#	distance(copyDatum['features'], datum['features'])
				b += dists[idx1][idx2]
				bc_count += 1
		if wc_count == 0:
			coeffs.append(1)
			continue
		else:
			a /= (wc_count)
			b /= bc_count
			sc = (b-a)/max([a, b])
			coeffs.append(sc)
	sc = sum(coeffs)/len(coeffs)
	return sc

def distance(p1, p2):
	return sum([(x-y)**2 for x, y in zip(p1,p2)])

def WCSSD(data):
	counts = dict()
	centroids = {i:list() for i in set([datum['cluster'] for datum in data])}
	oneFeature = data[0]['features']
	for i, croid in centroids.items():
		centroids[i] = [0 for x in oneFeature]
		counts.setdefault(i, 0)
	for datum in data:
		centroids[datum['cluster']] = [x+y for x,y in zip(\
			centroids[datum['cluster']], datum['features'])]
		counts[datum['cluster']] += 1
	for i, croid in centroids.items():
		centroids[i] = [x/counts[i] for x in croid]
	wcssd = 0
	for datum in data:
		wcssd += distance(datum['features'], centroids[datum['cluster']])
	return wcssd

def SC2(data):
	global dists
	coeffs = []
	dcopy = copy.deepcopy(data)
	counts = dict()
	for idx, datum in enumerate(data):
		counts.setdefault(datum['cluster'], 0)
		counts[datum['cluster']] += 1
	for idx1, copyDatum in enumerate(dcopy):
		distances = [0 for x in set([datum['cluster'] for datum in data])]
		for idx2, datum in enumerate(data):
			distances[datum['cluster']] += dists[idx1][idx2]
		a = None
		if counts[copyDatum['cluster']] == 1:
			coeffs.append(1)
			continue
		else:
			a = distances[copyDatum['cluster']]/(counts[copyDatum['cluster']] - 1)
		distances = {idx: dist/counts[idx] for idx, dist in enumerate(distances)}
		distances.pop(copyDatum['cluster'])
		b = min(list(distances.values()))			
		coeff = (b-a)/max([a, b])
		coeffs.append(coeff)
	sc2 = sum(coeffs)/len(coeffs)
	return sc2

def SC3(data):
	global dists
	coeffs = []
	clusters = dict()
	for idx, datum in enumerate(data):
		clusters.setdefault(datum['cluster'], list())
		clusters[datum['cluster']].append(idx)
	debug = dict()
	coeffs = []
	for cluster1, indices1 in clusters.items():
		Bs = None
		As = None
		indices1 = np.array(indices1)		
		for cluster2, indices2 in clusters.items():
			if cluster2 == cluster1:
				As = np.sum(dists[indices1[:,None], indices2], axis=1)
				if len(indices2) == 1:
					As.fill(0)
				else:
					As /= (len(indices2)-1)
			else:
				newCol = np.sum(dists[indices1[:,None], indices2], axis=1)
				newCol /= len(indices2)
				if Bs is None:
					Bs = newCol
				else:
					if len(Bs.shape) == 1:
						Bs = Bs.reshape((Bs.shape[0], 1))
					newCol = newCol.reshape((newCol.shape[0],1))
					Bs = np.hstack((Bs, newCol))
		if len(Bs.shape) == 2:
			Bs = np.amin(Bs, axis=1)
		As = list(As)
		Bs = list(Bs)
		for a, b in zip(As, Bs):
			if indices1.shape[0]==1:
				coeffs.append(0)				
			else:
				coeffs.append((b-a)/max([a, b]))
	return sum(coeffs)/len(coeffs)

class Kmeans(object):
	def __init__(self, k, max_iterations=50):
		self.k = k
		self.centroids = dict()
		self.dists = dict()
		self.max_iterations = max_iterations

	def cluster(self, data):
		self.data = data
		#Initial centroid assignment
		centroids = np.random.choice(data, size=self.k, \
			replace=False)
		for i, croid in enumerate(centroids):
			self.centroids[i] = croid['features']
		for datum in data:
			datum['cluster'] = None
		iteration = 0
		while iteration < self.max_iterations:
			assg_change = False
			#Cluster Assignment Step
			for datum in self.data:
				cluster = 0
				min_distance = None
				for i, croid in self.centroids.items():
					dist = distance(croid, datum['features'])
					if min_distance is None or dist < min_distance:
						cluster = i
						min_distance = dist
				if cluster != datum['cluster']:
					assg_change = True
				datum['cluster'] = cluster
			#If no change in assignment, we converged!
			if assg_change == False:
				break
			#Centroid calculation step
			counts = dict()
			for i, croid in self.centroids.items():
				self.centroids[i] = [0 for x in croid]
				counts.setdefault(i, 0)
			for datum in self.data:
				self.centroids[datum['cluster']] = [x+y for x,y in zip(\
					self.centroids[datum['cluster']], datum['features'])]
				counts[datum['cluster']] += 1
			for i, croid in self.centroids.items():
				if counts[i]!=0:
					self.centroids[i] = [x/counts[i] for x in croid]
			iteration += 1

def hierarchicalClustering(X, type, classes):
	Z = linkage(X, type)
	# calculate full dendrogram
	plt.figure(figsize=(25, 10))
	title = 'Hierarchical Clustering Dendrogram for {} linkage with {} classes'.format(type, classes)
	plt.title(title)
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(
	    Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=8.,  # font size for the x axis labels
	)
	plt.savefig(title + '.png')
	plt.clf()
	plt.cla()
	plt.close()
	return Z

def plotErrorBar(X, means, stds, xlabel, ylabel, title):
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.errorbar(X, means, yerr=stds)
	plt.savefig(title + '.pdf')  
	plt.clf()
	plt.cla()
	plt.close()

def plotScatterPlot(twoDdata, title):
	colors = plt.cm.Set3(np.linspace(0, 1, 10))
	classData = dict()
	for datum in twoDdata:
		classData.setdefault(datum['class'],[list(),list()])
		classData[datum['class']][0].append(datum['features'][0])
		classData[datum['class']][1].append(datum['features'][1])
	count = 0
	for key, val in classData.items():
		plt.scatter(val[0], val[1], color=colors[count])
		count += 1
	plt.legend([str(i) for i in range(0, 10)])
	plt.title(title)
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.savefig(title + '.png')
	plt.clf()
	plt.cla()
	plt.close()

def kmeansWithNMI(data, num_classes, k, suffix=''):
	model = Kmeans(k)
	model.cluster(data)
	nmi = NMI(data)
	print('NMI for {} classes and k={} is {}'.format(num_classes, k, nmi))
	shuffle(data)
	title = 'Visualization of cluster labels with {} classes{}'.\
	format(num_classes, suffix)
	plotScatterPlot(data[:100], title)

def kmeansAnalysis(data, name, runs=10, suffix=''):
	global dists
	K = [2, 4, 8, 16, 32]
	bestK = None
	bestSC = None
	dists = squareform(pdist([d['features'] for d in data]))
	means = []
	stds = []
	firstScores = []
	for k in K:
		curMeasures = []
		for run in range(0, runs):
			print('K:', k, 'run:', run)
			model = Kmeans(k)
			model.cluster(data)
			wcssd = WCSSD(data)
			sc = SC3(data)
			curMeasures.append((wcssd, sc))
			if run == 0:
				firstScores.append((wcssd, sc))		
				if bestK is None or bestScore < sc:
					bestK = k
					bestScore = sc
		means.append(tuple(np.mean(curMeasures, axis=0)	))	
		stds.append(tuple(np.std(curMeasures, axis=0)))
	for metIdx, metric in enumerate(["WCSSD", "SC"]):
		mean = [mean[metIdx] for mean in means]
		std = [std[metIdx] for std in stds]
		firstScore = [fs[metIdx] for fs in firstScores]
		title = "{} vs K for {} classes{}".format(metric, name, suffix)
		plotLineChart(K, firstScore, "K", metric, title)
		if runs==10:
			plotErrorBar(K, mean, std, "K", metric,  "Average " +title)
	return bestK, bestScore

def plotLineChart(X, Y, xlabel, ylabel, title):
	plt.plot(X, Y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(title + '.png')
	plt.clf()
	plt.cla()
	plt.close()

def plotGreyscale(data, title):
	plt.imshow(data, cmap='gray', interpolation='nearest') #, vmin=0, vmax=255)
	plt.title(title)
	plt.savefig(title + '.png')
	plt.clf()
	plt.cla()
	plt.close()

def comparison(data, classes):
	global dists
	counts = {cls:0 for cls in classes}
	num_classes = len(classes)
	tenImages = []
	shuffle(data)
	for datum in data:
		if counts[datum['class']] < 10:
			counts[datum['class']] += 1
			tenImages.append(datum)
		if sum(counts.values()) == 10*len(counts.values()):
			break
	dists = squareform(pdist([d['features'] for d in tenImages]))	
	X = [d['features'] for d in tenImages]
	labels = [d['class'] for d in tenImages]
	for type in ['single', 'complete', 'average']:
		Z = hierarchicalClustering(X, type, len(classes))
		max_num_clusters = len(tenImages)
		scores = []
		bestK = None
		bestSC = None
		for k in range(2, max_num_clusters+1):
			labels = fcluster(Z, k, criterion='maxclust')
			for idx, datum in enumerate(tenImages):
				datum['cluster'] = labels[idx]-1
			sc3 = SC3(tenImages)
			scores.append((k, WCSSD(tenImages), sc3))
			X = [datum['features'] for datum in tenImages]
			labels = [datum['cluster'] for datum in tenImages]
			sc = metrics.silhouette_score(X, labels, metric='euclidean')
			if bestSC is None or bestSC < sc:
				bestK = k
				bestSC = sc
			if (sc - sc3)**2 > (0.00000001)**2:
				print("Not equal")
				input()
		print("For {} classes with {} linkage, the best K is {} with SC {}".\
			format(num_classes, type, bestK, bestSC))
		labels = fcluster(Z, bestK, criterion='maxclust')
		for idx, datum in enumerate(tenImages):
				datum['cluster'] = labels[idx]-1
		nmi = NMI(tenImages)
		print('NMI for {} linkage with {} classes'.format(type, len(classes)), nmi)
		plotLineChart([k[0] for k in scores], [k[1] for k in scores], 'K', 'WCSSD',\
			'WCSSD vs K for {} linkage with {} classes'.format(type, len(classes)))
		plotLineChart([k[0] for k in scores], [k[2] for k in scores], 'K', 'SC',\
			'SC vs K for {} linkage with {} classes'.format(type, len(classes)))

def PCA(data):
	X = np.array([datum['features'] for datum in data])
	Xmeans = np.mean(X, axis=0)
	Xcentered = np.subtract(X, Xmeans)
	cov = np.dot(np.transpose(Xcentered), Xcentered)
	return Xcentered, np.linalg.eig(cov)

def pcaAnalysis(data, name):
	Xcentered, (eig_val, eig_vec) = PCA(data)
	for datum, newF in zip(data, Xcentered):
		datum['features'] = newF	
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i].real) for i in range(len(eig_val))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	eig_pairs = eig_pairs[:10]
	for id1,i in enumerate(eig_pairs):
		mat = np.array(i[1]).reshape(28, 28)
		title = 'Eigenvector {} for {} classes'.format(id1+1, name)
		plotGreyscale(mat, title)
	eig_vectors = [eig_pair[1] for eig_pair in eig_pairs]
	cdata = []
	for datum in data:
		dct = dict()
		dct['class'] = datum['class']
		dct['features'] = [np.dot(datum['features'], eig_vectors[0]), \
		np.dot(datum['features'], eig_vectors[1])]
		cdata.append(dct)
	title = 'Scatter plot for first two principle components for {} classes'.format(name)
	shuffle(cdata)
	plotScatterPlot(cdata[:1000], title)
	del cdata[:]
	for datum in data:
		datum['features'] = np.dot(eig_vectors, datum['features'])		
	del eig_vectors[:]
	bestK, bestScore = kmeansAnalysis(data, name, runs=1, suffix = ' with PCA')
	print("For {} classes with PCA, the best K is {} with SC {}".format(name, bestK, bestScore))
	kmeansWithNMI(data, name, bestK)#name if name != 10 else 8, suffix=' with PCA')

def reportScoresOnScreen(fileName, k):
	global dists
	sel = selector(fileName)
	data, _ = sel.read(100)
	dists = squareform(pdist([d['features'] for d in data]))
	model = Kmeans(k)
	model.cluster(data)
	wcssd = WCSSD(data)
	sc = SC3(data)
	nmi = NMI(data)
	print('WC-SSD', wcssd)
	print('SC', sc)
	print('NMI', nmi)

if __name__ == "__main__":
	if len(sys.argv)==3:
		reportScoresOnScreen(sys.argv[1], int(sys.argv[2]))
	if len(sys.argv)==2 and sys.argv[1]=="4":
		print("Doing PCA Analysis")
		sel = selector('digits-raw.csv')
		for classes in [range(0, 10), [2,4,6,7], [6,7]]:
			data, _ = sel.read(100)
			data = sel.filter(classes, data)
			pcaAnalysis(data, len(classes))
			del data[:]
		print("Completed PCA Analysis")
	elif len(sys.argv)==2 and sys.argv[1]=="2":
		print('Performing exploration...')
		explore()	
		print('Exploration complete!')
	elif len(sys.argv)==2 and sys.argv[1]=="1":
		sel = selector('digits-embedding.csv')
		d1, _ = sel.read(100)
		d2 = sel.filter([2, 4, 6, 7], d1)
		d3 = sel.filter([6, 7], d2)
		print('Performing K-means analysis...')
		for data, num_classes in zip([d1, d2, d3], [10, 4, 2]):
			bestK, bestScore = kmeansAnalysis(data, num_classes)
			print("For {} classes, the best K is {} with SC {}".format(num_classes, bestK, bestScore))
			kmeansWithNMI(data, num_classes, num_classes if num_classes!=10 else 8)
		print('K-means analysis complete!')
	elif len(sys.argv)==2 and sys.argv[1]=="3":
		sel = selector('digits-embedding.csv')
		d1, _ = sel.read(100)
		d2 = sel.filter([2, 4, 6, 7], d1)
		d3 = sel.filter([6, 7], d2)
		print("Performing hierarchical clustering")
		for data, classes in zip([d1, d2, d3], [range(0,10), [2, 4, 6, 7], [6, 7]]):
			comparison(data, classes)
		print("Hierarchical clustering complete")
	'''
	model = Kmeans(32)
	sel = selector('digits-embedding.csv')
	d1, _ = sel.read(10)
	d2 = sel.filter([2, 4, 6, 7], d1)
	d3 = sel.filter([6, 7], d2)
	model.cluster(d1)
	print('NMI', NMI(d1))
	print('SC:', SC(d1))
	print('WCSSD:', WCSSD(d1))
	print('SC2:', SC2(d1))
	from sklearn import metrics
	X = [datum['features'] for datum in d1]
	labels = [datum['cluster'] for datum in d1]
	print('Scipy SC:', metrics.silhouette_score(X, labels, metric='euclidean'))
	cls = [datum['class'] for datum in d1]
	#print(cls, labels)
	print('Scipy NMI', metrics.normalized_mutual_info_score(cls, labels))
	comparison(d1, range(0,10))
	'''