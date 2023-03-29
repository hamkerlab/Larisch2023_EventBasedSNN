from ANNarchy import *
setup(dt=1.0, seed=314)#,num_threads=4)#!!!
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from net import *

import sys
from tqdm import tqdm
import tonic
import random
from PIL import Image

'''RANDOM INPUT'''
#-------------------------------------------------------------------------------

def make_cube(events):
    cube = np.zeros((34,34,350), dtype='int32') # hard coded?
    for s in events:
        x = s[0]
        y = s[1]
        t = int(s[2]//1000) # set to ms
        p = s[3]
        if p == 1:
            cube[y,x,t] = 1
        else:
            cube[y,x,t] = -1
    return(cube)

def preprocess(data):
    cube = np.zeros((34,34,350), dtype='int32') # hard coded?
    single_evbs = data
    label = single_evbs[1]
    single_evbs = single_evbs[0]
    for s in single_evbs:
        x = s[0]
        y = s[1]
        t = int(s[2]//1000) # set to ms
        p = s[3]
        if p == 1:
            cube[y,x,t] = 1
        else:
            cube[y,x,t] = -1

    return(cube,label)


def split_cube(cube,n_patches=2):
    ## At the moment only for quadratic inputs and without overlap
    x,y,t = np.shape(cube)
    x_p = x//n_patches
    y_p = y//n_patches

    new_cube = np.zeros((n_patches*2,x_p,y_p,t))
    i_c = 0
    for i_x in range(n_patches):
        for i_y in range(n_patches):
            new_cube[i_c] = cube[0+(x_p*i_x):x_p+(x_p*i_x),0+(y_p*i_y):y_p+(y_p*i_y),:]
            i_c+=1

    return(new_cube)

def makeLilArray(cube):
    c_w, c_h,c_t = np.shape(cube)

    # divide in ON and OFF-Channel
    input_bin = np.zeros((c_w,c_h,c_t,2))
    input_bin[cube>0,0] = 1
    input_bin[cube<0,1] = 1
    
    # convert the input into arrays of spiking time points
    inpt_array =np.rollaxis(input_bin,2,4)

    inpt_array = np.reshape(inpt_array,(c_w*c_h*2,c_t))

    bin_array = []
    for i in range(np.shape(inpt_array)[0]):
        t_s = np.where(inpt_array[i] == 1)[0]
        t_s = t_s.tolist()               
        bin_array.append(t_s)
    return(bin_array)

def make_pic(cube_this,pwd):
    if not os.path.exists(pwd+'/pic'):
         os.makedirs(pwd+'/pic')
    
    w_this,h_this,time_this = np.shape(cube_this)
    lil_array = np.zeros((w_this,h_this,time_this))
    '''lil BILD Array Werte def'''
    lil_x_pos = []
    lil_y_pos = []
    lil_vals = []
    t_s = []
	
    #LIL ARRAY FÜLLEN
    for zeit in range(1,time_this):  
        for x_achse in range(0,w_this):
             for y_achse in range(0,h_this):
                 lil_array[x_achse, y_achse,zeit] = cube_this[x_achse,y_achse,zeit]
                 #BILD ARRAY FÜLLEN
                 if int(lil_array[x_achse, y_achse,zeit]) < 0:
                     lil_vals.append(0)
                     lil_x_pos.append(x_achse)
                     lil_y_pos.append(y_achse)
                     t_s.append(zeit)
                 elif int(lil_array[x_achse, y_achse,zeit]) > 0:
                     lil_x_pos.append(x_achse)
                     lil_y_pos.append(y_achse)
                     t_s.append(zeit)
                     lil_vals.append(1)
    '''CODE KOPIERT -- on the Shoulders of Titans'''
    batch_size = 35# one batch contains events of 100/10/1 ms 
    evt_imgs = np.zeros((1,w_this,h_this,2)) #1 für intitialisierungszwecke --> 4 dimensionale matrize voller 0en
    cnt = 0
    for i in range(len(lil_x_pos)):
        if t_s[i] > (cnt*batch_size): #erstelle neue zeitscheibe wenn intervall erreicht
            img = np.zeros((1,w_this,h_this,2))
            evt_imgs = np.append(evt_imgs,img,axis=0)
            cnt +=1 
        evt_imgs[cnt,int(lil_y_pos[i]-1),int(lil_x_pos[i]-1),int(lil_vals[i])] = 1		

    fig, axs = plt.subplots(1,cnt+1, figsize=(10,1))
    for i in range(cnt+1):
        axs[i].imshow(evt_imgs[i,:,:,0] - evt_imgs[i,:,:,1],cmap='gray',vmin=-1,vmax=1,interpolation='none') 
        #axs[i].colorbar() 
               
    plt.savefig(pwd+'/pic'+'/IMAGE',dpi=300,bbox_inches='tight')

    for i in range(cnt):
        plt.figure()
        plt.imshow(evt_imgs[i,:,:,0].T - evt_imgs[i,:,:,1].T,cmap='RdYlGn',vmin=-1,vmax=1,interpolation='none')
        plt.axis('off')
        plt.savefig(pwd+'/pic'+'/IMAGE_%i'%(i),dpi=300,bbox_inches='tight' )



def add_UniformNoise(cube, n = 1000 ):
    ## add uniformly distributed noise to the event list
    ## make sure, that noise events are not created, where already an event

    ## get the border of the cube    
    max_x, max_y, max_t = np.shape(cube)
    
    ## flatten the cube to an array
    cube_array = np.reshape(cube, (max_x * max_y * max_t))
    ## look where are the zeros -> there is no event
    idx_zeros = np.where(cube_array == 0)[0]
    ## shuffle the indicis
    np.random.shuffle(idx_zeros)

    ## create a list with new ON and OFF events
    event_list = np.random.randint(0,2, n)
    ## set 0 to -1 // OFF events
    idx_event = np.where(event_list == 0)[0]
    event_list[idx_event] = -1
    

    ### set new events in cube
    cube_array[idx_zeros[0:n]] =  event_list

    ## return the reshaped cube
    return(np.reshape(cube_array, (max_x, max_y, max_t) ))

def main():

    if not os.path.exists('output_svm'):
        os.mkdir('output_svm')

    compile()
    loadWeights()
    ## monitor to recored the spike times
    monV1 = Monitor(popV1,['spike'])

    w,h = (24,24)
    n_p = w//patchsize

    
    # load the training set
    data_train = tonic.datasets.NMNIST('../input/', train=True)
    n_samples_train = 60000
        
    rec_spkc_train_V1 = np.zeros((n_samples_train,n_p*n_p,n_exN))
    labels_train = np.zeros(n_samples_train)

    for s in tqdm(range(n_samples_train), ncols=80):
        # first preprocess 
        cube, label = preprocess(data_train[s])
        # save label
        labels_train[s] = label
        # cut out the border
        cube = cube[5:5+w,5:5+h]
        ## split it into four parts
        cube = split_cube(cube,2)
        # create the list of list for the input
        for p in range(4):
            input_bin = makeLilArray(cube[p])
            ## feed the input into the SNN
            reset()
            popInput.reset()
            popInput.spike_times = input_bin                
            simulate(350)
            # get the spike counts via the monitors
            spikesEx = monV1.get('spike')
            for j in range(n_exN):
                rateEx = len(spikesEx[j])
                rec_spkc_train_V1[s,p,j] = rateEx

    np.save('./output_svm/spkC_V1_train',rec_spkc_train_V1)
    np.save('./output_svm/labels_train', labels_train)

    ## do it again for the test set

    data_test = tonic.datasets.NMNIST('../input/', train=False)
    n_samples_test = 10000


    noise_list = np.linspace(0,10000,11)
    for noise in noise_list:
        this_prop= noise
        if not os.path.exists('output_svm/drop_event_prop_%i'%(this_prop)):
            os.mkdir('output_svm/drop_event_prop_%i'%(this_prop))
        pwd = 'output_svm/drop_event_prop_%i'%(this_prop)

        rec_spkc_test_V1 = np.zeros((n_samples_test,n_p*n_p,n_exN))
        labels_test = np.zeros(n_samples_test)
        


        for s in tqdm(range(n_samples_test), ncols=80): #n_samples_test
            # first preprocess 
            events, label = data_test[s]
            
            # save label
            labels_test[s] = label
            '''TRANSFORM EVENTS'''
            cube = make_cube(events)
            cube = add_UniformNoise(cube, n=int(noise))
            # cut out the border
            cube = cube[5:5+w,5:5+h]  #24,24,350        
            if s==1000:
                make_pic(cube,pwd)
            ## split it into four parts
            cube = split_cube(cube,2) #4,12,12,350
            #print(np.shape(cube))
            # create the list of list for the input
            for p in range(4):
                input_bin = makeLilArray(cube[p])
                ## feed the input into the SNN
                reset()
                popInput.reset()
                popInput.spike_times = input_bin                
                simulate(350)
                # get the spike counts via the monitors
                spikesEx = monV1.get('spike')
                for j in range(n_exN):
                    rateEx = len(spikesEx[j])
                    rec_spkc_test_V1[s,p,j] = rateEx
            
        np.save('./output_svm/drop_event_prop_%i/spkC_V1_test'%(this_prop),rec_spkc_test_V1)
        np.save('./output_svm/drop_event_prop_%i/labels_test'%(this_prop), labels_test)

if __name__ == "__main__":
    
    main() 

