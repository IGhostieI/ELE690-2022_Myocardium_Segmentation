import sys
sys.path.append('/home/stud/casperc/bhome/wmri')
import numpy as np
from organizeimage_TE import *
from CropHeart import crop_heart
import pickle as p 
import orgim_scr as oi
import matplotlib.pyplot as plt 
import numpy as np 
import pydicom as dicom 
import time
from sklearn.model_selection import train_test_split

def nest_flatten(n_list):
    return [element for sublist in n_list for element in sublist]

dataset = 'haglag'
delineation_location = {'haglag':'konsensus_leik_stein', 'vxvy':'erlend'}

filepath = '/home/prosjekt5/EKG/data/wmri/'
filepathDel = f'/home/prosjekt5/EKG/data/wmri/{delineation_location.get(dataset)}/'

prm = oi.userGetOrg(dataset) # Sets the user settings for the organization function within 0i.main()
prm['filepath'] = filepath # filepath must be changed to match the path to data.
prm['filepathDel'] = filepathDel 

#print(prm.keys())
patients_with_delineation = [var for var in prm['Ptsgm'] if var] # Make a list of all the patients that have a deliniation
#print(List)
ids = [] 
imgs = []
Mmyo = []
with open('log.txt', mode='w', encoding='utf-8') as log:
    start = time.perf_counter()
    print('starting')
    for i, patient in enumerate(patients_with_delineation):
        #print(f'#{i}, patient id: {patient}')
        try: 
            inD, b, prm_h, Pt_h = oi.main(dataset, patient) # Extract the info into inD. It contains pictures, delineations etc. 
        except Exception as e:
            log.write(f'#{i}: {patient}, oi.main(), {repr(e)}\n')
            print(f'failed at #{i} oi.main()')
            continue
        #print(inD.keys())
        try:
            cpD = crop_heart(inD, plot=0) # Crops the image of the heart to zoom more onto the myocard
        except Exception as e:
            log.write(f'{i}, {patient}, crop_heart(), {repr(e)}\n')
            print(f'failed at #{i} crop_heart()')
            continue
        cpD['id'] = patient
        ids.append(cpD['id'])
        imgs.append(cpD['X'])
        Mmyo.append(cpD['Mmyo'])
finish = time.perf_counter()
print(f'time: {(finish-start)//60} min, {round(((finish-start)/60-(finish-start)//60)*60, 2)} s')
imgs = np.asarray(imgs)
Mmyo = np.asarray(Mmyo)


train_imgs, test_imgs, train_Mmyo, test_Mmyo, train_id, test_id = train_test_split(imgs, Mmyo, ids, test_size=0.25, train_size=0.75,random_state=1)
train_imgs, val_imgs, train_Mmyo, val_Mmyo, train_id, val_id = train_test_split(train_imgs, train_Mmyo, train_id, test_size=0.2, train_size=0.80, random_state=1)

train_imgs = np.asarray(nest_flatten(train_imgs))

train_Mmyo = np.asarray(nest_flatten(train_Mmyo))

test_imgs = np.asarray(nest_flatten(test_imgs))
test_Mmyo = np.asarray(nest_flatten(test_Mmyo))

val_imgs = np.asarray(nest_flatten(val_imgs))
val_Mmyo = np.asarray(nest_flatten(val_Mmyo))

id_dict = {'full':ids, 'train':train_id, 'test':test_id, 'val':val_id}
data_dict = {'train images':train_imgs,'train Mmyo':train_Mmyo, 'test images':test_imgs, 'test Mmyo':test_Mmyo, 'validation images':val_imgs, 'validation Mmyo':val_Mmyo, 'id':id_dict}

with open(f'{dataset}_imgs_and_Mmyo_0_15_validation.p', 'wb') as data_file:
    p.dump(data_dict, data_file, protocol=p.HIGHEST_PROTOCOL)
