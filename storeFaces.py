from facematch_functions import getFace, getEmbedding
import cv2
import numpy as np
import codecs, json
import os


#variables to start writing the json file
data = {}
data['faces'] = []
number = 0
images_folder = "/home/erik/Documenten/Msc/Research internship/research-internship/images"

for path in os.listdir(images_folder):
    directory = os.path.join(images_folder, path)
    if os.path.isdir(directory):
        amount_of_files = len(os.listdir(directory))
        for file in os.listdir(directory):
            print(str(number)+"/"+str(amount_of_files)+": "+os.path.basename(file))
            #print("Finding faces in: "+file)
            img = cv2.imread(os.path.join(directory,file))
            faces = getFace(img)
            for face in faces:
                number += 1
                data['faces'].append({  
                    'id': number,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    'file': os.path.join(directory,file),
                    'embedding': face['embedding'].tolist()
                })

cv2.waitKey(0)
cv2.destroyAllWindows()

file_path = "data.json"
json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)