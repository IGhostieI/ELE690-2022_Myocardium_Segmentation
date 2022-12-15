import os
import pickle
import numpy
import organizeimage_TE as orgin

""" fr = "Ipython"
try:
    # Works with ipython
    #ip = get_ipython()
    get_ipython().magic("load_ext autoreload")
    get_ipython().magic("autoreload 2")
except:
    # If cpython interpreter is used
    fr = "cpython" """


def userGetOrg(study='haglag'):
    usersetfile = study + '_set_orgin.p'
    if os.path.isfile(usersetfile):
        with open(usersetfile, "rb") as fp:
            usersettings = pickle.load(fp)
    return usersettings

def print_prm(prm):
    keys = [*prm.keys()]
    for k in range(0,len(keys)):
        print('')
        #print(keys[k])
        s = 'prm[' + '\''  + keys[k] + '\'' + ']'
        if type(prm[keys[k]]) is int:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is bool:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is dict:
            s = s + ' = ' + str(prm[keys[k]])
        elif type(prm[keys[k]]) is str:
            s = s + ' = ' + prm[keys[k]]
        elif type(prm[keys[k]]) is list:
            s = s + ' = ' + str(prm[keys[k]][0:10]) + '(' + str(len(prm[keys[k]])) + ' elements)'
        elif type(prm[keys[k]]) is numpy.ndarray:
            s = s + ' = ' + str(prm[keys[k]][0:10]) + '(' + str(len(prm[keys[k]])) + ' elements)'
        else:
            s = s + ' = ' + str(type(prm[keys[k]]))
        print(s)

def main(study='haglag', PtID='AEA063'):
    prm = userGetOrg(study)
    prm['filepath'] = '/home/prosjekt5/EKG/data/wmri/'
    if study =='haglag':
        prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/konsensus_leik_stein/'
    elif study == 'vxvy':
        prm['filepathDel'] = '/home/prosjekt5/EKG/data/wmri/erlend/'
    inD, b = orgin.organizeimage_TE(prm['filepath'],
                                    prm['filepathDel'],
                                    PtID,
                                    prm)
    Pt = [nm for nm in prm['Ptsgm'] if nm!='']
    return inD, b, prm, Pt


if __name__ == '__main__':
    inD, b, prm, Pt = main()    
