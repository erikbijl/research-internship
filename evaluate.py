import os

def f_measure():
	amt_files = 0
	TP = 0
	FP = 0
	FN = 0
	TN = 0

	cluster_folder = "./clusters/"
	images = []
	for subfolder in os.listdir(cluster_folder):
		for filename in os.listdir(os.path.join(cluster_folder, subfolder)):
			amt_files += 1
			for file_in_same_folder in os.listdir(os.path.join(cluster_folder, subfolder)):
				if filename <= file_in_same_folder:
				#	print("checking: "+ filename+ " " + file_in_same_folder)
					if filename.split("00")[0] == file_in_same_folder.split("00")[0]:
						TP += 1
					else:
						FP += 1
			for other_folder in os.listdir(cluster_folder):
				if other_folder > subfolder:
					for file_in_diff_folder in os.listdir(os.path.join(cluster_folder, other_folder)):
				#		print("checking: "+ filename+ " " + file_in_diff_folder)
						if filename.split("00")[0] == file_in_diff_folder.split("00")[0]:
							FN += 1
						else:
							TN += 1


	pairs = amt_files*(amt_files-1)/2  + amt_files
	sum_pos_neg = TP+FP+FN+TN

	print(str(int(pairs))+"/"+str(sum_pos_neg))
	print("TP: "+str(TP))
	print("FP: "+str(FP))
	print("FN: "+str(FN))
	print("TN: "+str(TN))


	precision = ((TP / (TP + FP)))
	recall = TP / (TP + FN)
	f = 2*(float(precision)*float(recall))/(float(precision) + float(recall))

	print("Precision: "+str(precision))
	print("Recall: "+str(recall))
	print("f_measure: "+str(f))
	print("accuracy: "+str((TP+TN)/sum_pos_neg))
	return f