import matplotlib as mp
import numpy as np
import matplotlib.pyplot as plt

#parameters
titleSize = 30
labelSize = 24
ticksize = 20
markersize=20
dpi = 100
spaceh = 0.25
markersize=20
figureSize=(20,18)
#-----------------------------------------------------------------------------
def getTime(data):
    return(np.arange(len(data))*50.0/1000.0)
#------------------------------------------------------------------------------
def setSubplotDimension(a):
    x = 0
    y = 0
    if ( (a % 1) == 0.0):
        x = a
        y =a
    elif((a % 1) < 0.5):
        x = round(a) +1
        y = round(a)
    else :
        x = round(a)
        y = round(a)
    return (x,y)   
#-----------------------------------------------------------------------------
def plotWeightsImage(weightMatrix,wMax=35.0):

    colums = np.shape(weightMatrix)[0] # = numbers of L2-Neurons
    rows = np.shape(weightMatrix)[1]   # = numbers of input-Neurons
    maxW = np.max(weightMatrix)
    x,y = setSubplotDimension(np.sqrt(colums))
    wMin = 0.0
    fig = plt.figure(figsize=figureSize,dpi=dpi)    
    for i in range(colums):
        imageArray = weightMatrix[i]
        image = np.reshape(imageArray, (-1,np.sqrt(rows)))
        plt.subplot(x,y,i+1)
        im = plt.imshow(image,cmap=mp.cm.Greys_r,aspect='auto',interpolation="nearest",vmin=wMin,vmax=wMax)

    #fig.subplots_adjust(right = 0.7)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    #print(maxW)
    
    return(im)
#-----------------------------------------------------------------------------
def plotWeightsInLatImage(weightMatrix,wMax=7.0):

    colums = np.shape(weightMatrix)[0] # = numbers of post-Neurons
    rows = np.shape(weightMatrix)[1]   # = numbers of pre-Neurons
    wMatrix = np.zeros((colums,colums))
    for i in range(colums):
        line = np.insert(weightMatrix[i][:],i,np.nan)
        wMatrix[i,:] = line
    colums = np.shape(wMatrix)[0]
    rows = np.shape(wMatrix)[1]
    x,y = setSubplotDimension(np.sqrt(colums))
    wMin = 0.0
    fig = plt.figure(figsize=figureSize,dpi=dpi)    
    for i in range(rows):
        imageArray = wMatrix[i]
        image = np.reshape(imageArray, (-1,np.sqrt(rows)))
        plt.subplot(x,y,i+1)
        cmap = mp.cm.Greys_r
        cmap.set_bad('black',1.)
        im = plt.imshow(image,cmap=cmap,aspect='auto',interpolation="nearest",vmin=wMin,vmax=wMax) 
    return(im)
#----------------------------------------------------------------------------
def plotgEx(l2gExc,Layer):
    nbr = np.shape(l2gExc)[1]
    for i in range(int(nbr/nbr)): 
        fig = plt.figure(figsize=figureSize,dpi=dpi)
        fig.add_subplot(4,1,1)
        plt.plot(getTime(l2gExc[:,0+(4*i)]),l2gExc[:,0+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Ex')
        fig.add_subplot(4,1,2)
        plt.plot(getTime(l2gExc[:,1+(4*i)]),l2gExc[:,1+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Ex')
        fig.add_subplot(4,1,3)
        plt.plot(getTime(l2gExc[:,2+(4*i)]),l2gExc[:,2+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Ex')
        fig.add_subplot(4,1,4)
        plt.plot(getTime(l2gExc[:,3+(4*i)]),l2gExc[:,3+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Ex')
        if(Layer=='Exi'):
            plt.savefig('output/V1Layer/gExc_'+str(i)+'.png') 
        if(Layer=='Inhib'):
            plt.savefig('output/InhibitLayer/gExc_'+str(i)+'.png')
    plt.close('all')
#----------------------------------------------------------------------------
def plotgInh(l2gInh,Layer):
    nbr = np.shape(l2gInh)[1]
    for i in range(int(nbr/nbr)): 
        fig = plt.figure(figsize=figureSize,dpi=dpi)
        fig.add_subplot(4,1,1)
        plt.plot(getTime(l2gInh[:,0+(4*i)]),l2gInh[:,0+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Inh')
        fig.add_subplot(4,1,2)
        plt.plot(getTime(l2gInh[:,1+(4*i)]),l2gInh[:,1+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Inh')
        fig.add_subplot(4,1,3)
        plt.plot(getTime(l2gInh[:,2+(4*i)]),l2gInh[:,2+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Inh')
        fig.add_subplot(4,1,4)
        plt.plot(getTime(l2gInh[:,3+(4*i)]),l2gInh[:,3+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Inh')
        if(Layer=='Exi'):
            plt.savefig('output/V1Layer/gInh_'+str(i)+'.png') 
        if(Layer=='Inhib'):
            plt.savefig('output/InhibitLayer/gInh_'+str(i)+'.png')
    plt.close('all')
#----------------------------------------------------------------------------
def plotgEx_gInh(l2gExc,l2gInh,Layer):
    nbr = np.shape(l2gInh)[1]
    for i in range(int(nbr/nbr)): 
        fig = plt.figure(figsize=figureSize,dpi=dpi)
        fig.add_subplot(4,1,1)
        plt.plot(getTime(l2gInh[:,0+(4*i)]),l2gExc[:,0+(4*i)]-l2gInh[:,0+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Exc - g_Inh')
        fig.add_subplot(4,1,2)
        plt.plot(getTime(l2gInh[:,1+(4*i)]),l2gExc[:,1+(4*i)]-l2gInh[:,1+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Exc - g_Inh')
        fig.add_subplot(4,1,3)
        plt.plot(getTime(l2gInh[:,2+(4*i)]),l2gExc[:,2+(4*i)]-l2gInh[:,2+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Exc - g_Inh')
        fig.add_subplot(4,1,4)
        plt.plot(getTime(l2gInh[:,3+(4*i)]),l2gExc[:,3+(4*i)]-l2gInh[:,3+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('g_Exc - g_Inh')
        if(Layer=='Exi'):
            plt.savefig('output/V1Layer/gExc_gInh_'+str(i)+'.png') 
        if(Layer=='Inhib'):
            plt.savefig('output/InhibitLayer/gExc_gInh_'+str(i)+'.png')
    plt.close('all')
    
#----------------------------------------------------------------------------
def plotVM(l2VM,Layer):
    nbr =np.shape(l2VM)[1]
    for i in range(int(nbr/nbr)): 
        fig = plt.figure(figsize=figureSize,dpi=dpi)
        fig.add_subplot(4,1,1)
        plt.plot(getTime(l2VM[:,0+(4*i)]),l2VM[:,0+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('Membran Potential')
        fig.add_subplot(4,1,2)
        plt.plot(getTime(l2VM[:,1+(4*i)]),l2VM[:,1+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('Membran Potential')
        fig.add_subplot(4,1,3)
        plt.plot(getTime(l2VM[:,2+(4*i)]),l2VM[:,2+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('Membran Potential')
        fig.add_subplot(4,1,4)
        plt.plot(getTime(l2VM[:,3+(4*i)]),l2VM[:,3+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('Membran Potential')
        if(Layer=='Exi'):
            plt.savefig('output/V1Layer/membran_'+str(i)+'.png') 
        if(Layer=='Inhib'):
            plt.savefig('output/InhibitLayer/membran_'+str(i)+'.png')

    plt.close('all')
#----------------------------------------------------------------------------
def plotVMean(l2vMean,Layer):
    nbr = np.shape(l2vMean)[1]
    for i in range(int(nbr/nbr)): 
        fig = plt.figure(figsize=figureSize,dpi=dpi)
        fig.add_subplot(4,1,1)
        plt.plot(getTime(l2vMean[:,0+(4*i)]),l2vMean[:,0+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('u_mean')
        fig.add_subplot(4,1,2)
        plt.plot(getTime(l2vMean[:,1+(4*i)]),l2vMean[:,1+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('u_mean')
        fig.add_subplot(4,1,3)
        plt.plot(getTime(l2vMean[:,2+(4*i)]),l2vMean[:,2+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('u_mean')
        fig.add_subplot(4,1,4)
        plt.plot(getTime(l2vMean[:,3+(4*i)]),l2vMean[:,3+(4*i)])
        plt.subplots_adjust(hspace = spaceh)
        plt.xlabel('Time in S')
        plt.ylabel('u_mean')
        if(Layer=='Exi'):
            plt.savefig('output/V1Layer/vmean_'+str(i)+'.png') 
        if(Layer=='Inhib'):
            plt.savefig('output/InhibitLayer/vmean_'+str(i)+'.png')

    plt.close('all')
#-----------------------------------------------------------------------------
def plotInputImages(images):
    
    fig = plt.figure(figsize=figureSize,dpi=dpi)
    for i in range(5):
        for j in range(2):
            imageON = images[:,:0,j+(j*i*2)]
            fig.add_subplot(j+(i*j),5,4)            
            plt.imshow(image,cmap=plt.get_cmap('gray'))
            imageOFF = images[:,:1,j*i]
            fig.add_subplot(j+(i*j*2)+1,5,4)
            plt.imshow(image,cmap=plt.get_cmap('gray'))
            
    plt.savefig('input/images.png')
