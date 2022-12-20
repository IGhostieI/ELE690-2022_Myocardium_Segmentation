import numpy as np


def crop_heart(inD, crop_method=1, cco=128, plot=1):
    """
    ----------------
    Translated to Python by Casper Cappelen 
    November 2022
---------------- written by: Kjerst Engan

last changes:  March 2016 , incase no deliniation, check
                           for field Mmyo in inD. 

----------------------------------------------------
plot =1    %  if plot ==1, some results are plotted for illustration
 if no wish to plot, set to 0.
  
---  We wish removing some areas around the edge of the image to
  focus on the heart.  this is calledcropping   The parameter is set here.

    """
    cpD = {}
    nosl=len(inD['X'])
    [r,c]=inD['X'][0].shape
    emptyslice=np.zeros([nosl,1])
    if 'Mmyo' in inD:
        
        minx=np.zeros([nosl,1])
        maxx=np.zeros([nosl,1])
        miny=np.zeros([nosl,1])
        maxy=np.zeros([nosl,1])
        
        for i in range(nosl):
            [x,y]=np.where(inD['Mmyo'][i])
            if not x.any():
                emptyslice[i]=1
                minx[i]=r/2
                miny[i]=r/2
                maxx[i]=c/2
                maxy[i]=c/2
            else:
                minx[i]=min(x)
                miny[i]=min(y)
                maxx[i]=max(x)
                maxy[i]=max(y)
           
        
        minpt=[min(minx), min(miny)]
        #maxpt=[max(maxx), max(maxy)]
        
        
        if crop_method==1:
            if cco>min(minpt):
                print('Crop Error!')
            elif cco > (r-max(maxx)):
                print('Crop Error!')
            elif cco > (c-max(maxy)):
                print('Crop Error!')
            

    if crop_method==1:
        nr=r-2*cco
        nc=c-2*cco
        cpD['X']=np.zeros([nosl,nr,nc])
        if 'Mmyo' in inD:
            cpD['Mmyo']=np.zeros([nosl,nr,nc])
            cpD['Minf']=np.zeros([nosl,nr,nc])
            cpD['cent']=np.zeros([nosl,nr,nc])
        
        for i in range(nosl):
            #print(f'i: {i}, cpD: {cpD["X"].shape}, inD: {inD["X"][i].shape}')
            #print(f'i: {i}, cpD: {cpD["X"][i][:,:].shape}, inD: {inD["X"][i][cco:r-cco,cco:c-cco].shape}')
            cpD['X'][i][:,:]=inD['X'][i][cco:r-cco,cco:c-cco]
            if 'Mmyo' in inD:
                #print(f'cpD keys crop: {cpD.keys()}')
                cpD['Mmyo'][i][:,:]=inD['Mmyo'][i][cco:r-cco,cco:c-cco]
                cpD['Minf'][i,:,:]=inD['Minf'][i][cco:r-cco,cco:c-cco]
                cpD['cent'][i][0]=inD['cent'][i][0]-cco
                cpD['cent'][i][1]=inD['cent'][i][1]-cco
                #print(cpD['cent'][i].shape)
    
    return cpD

        


