#-*- coding:utf-8 -*-
#----------------------imports and environment---------------------------------
import matplotlib as mp
mp.use('Agg')
from ANNarchy import *
setup(dt=1.0)
from time import *
import plots as plt
import matplotlib.pyplot as pt
import numpy as np
from Make_lilarray import Make_lilarray #Spike Source Array importieren
import tonic
from tqdm import tqdm

#------------------------global Variables------------------------------------
nbrOfPatches = 400000 
duration = 350
patchsize = 12 
inputNeurons = patchsize*patchsize*2
n_exN = 144#patchsize*patchsize 144
n_inN = int(n_exN/4)
#---------------------------------neuron definitions-------------------------

## Neuron Model for LGN/Input Layer ##
params = """
EL = -70.4      :population
VTrest = -50.4  :population
taux = 15.0     :population
"""
#if g_exc > -50.4: 1 else:
inpt_eqs ="""
    dg_exc/dt = EL/1000 : min=EL, init=-70.6
    Spike = 0.0
    dresetvar / dt = 1/(1.0) * (-resetvar)
    dxtrace /dt = 1/taux * (- xtrace )
    """

spkNeurLGN = Neuron(parameters=params,
                          equations=inpt_eqs,
                          reset="""g_exc=EL 
                                   Spike = 1.0
                                   resetvar=1.0
                                   xtrace+= 1/taux""", #erhöht sich bei jedem spike wichtig für lernregel--> hilfsvariable um zu wissen wie häufig es in ner gewissen zeit gespiked hat
                          spike="""g_exc > VTrest""")

## Neuron Model for V1-Layer, after Clopath et al.(2008) ##
params = """
gL = 30.0         :population
DeltaT = 2.0      :population
tauw = 144.0      :population
a = 4.0           :population  
b = 0.0805        :population  
EL = -70.6        :population
C = 281.0         :population  
tauz = 40.0       :population  
tauVT= 50.0       :population  
Isp = 400.0       :population  
VTMax = -30.4     :population 
VTrest = -50.4    :population  
taux = 15.0       :population  
tauLTD = 10.0     :population  
tauLTP= 7.0       :population  
taumean = 750.0   :population  
tau_gExc = 1.0    :population  
tau_gInh= 10.0    :population
"""


neuron_eqs = """
noise = Normal(0.0,1.0)
dvm/dt = if state>=2:+3.462 else: if state==1:-(vm+51.75)+1/C*(Isp - (wad+b))+g_Exc-g_Inh else:1/C * ( -gL * (vm - EL) + gL * DeltaT * exp((vm - VT) / DeltaT) - wad + z ) + g_Exc -g_Inh: init = -70.6
dvmean/dt = (pos(vm - EL)**2 - vmean)/taumean    :init = 0.0
dumeanLTD/dt = (vm - umeanLTD)/tauLTD : init=-70.0
dumeanLTP/dt = (vm - umeanLTP)/tauLTP : init =-70.0
dxtrace /dt = (- xtrace )/taux
dwad/dt = if state ==2:0 else:if state==1:+b/tauw else: (a * (vm - EL) - wad)/tauw : init = 0.0
dz/dt = if state==1:-z+Isp-10 else:-z/tauz  : init = 0.0
dVT/dt =if state==1: +(VTMax - VT)-0.4 else:(VTrest - VT)/tauVT  : init=-50.4
dg_Exc/dt = -g_Exc/tau_gExc
dg_Inh/dt = -g_Inh/tau_gInh
state = if state > 0: state-1 else:0
Spike = 0.0
dresetvar / dt = 1/(1.0) * (-resetvar)
           """
spkNeurV1 = Neuron(parameters = params,equations=neuron_eqs,spike="""(vm>VT) and (state==0)""",
                         reset="""vm = 29.0
                                  state = 2.0 
                                  VT = VTMax
                                  Spike = 1.0
                                  resetvar = 1.0
                                  xtrace+= 1/taux""")

#----------------------------------synapse definitions----------------------

#----- Synapse from Poisson to Input-Layer -----#
inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

#--STDP Synapses after Clopath et. al(2010)for Input- to Exitatory- Layer--#
equatSTDP = """
    ltdTerm = if w>wMin : (aLTD*(post.vmean/urefsquare)*pre.Spike * pos(post.umeanLTD - thetaLTD)) else : 0.0
    ltpTerm = if w<wMax : (aLTP * pos(post.vm - thetaLTP) *(pre.xtrace)* pos(post.umeanLTP - thetaLTD)) else : 0.0
      dw/dt = ( -ltdTerm + ltpTerm) :min=0.0, max=wMax"""

parameterFF="""
urefsquare = 9.0        :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 7e-08 *5         :projection
aLTP = 9e-08 *5         :projection
wMin = 0.0              :projection
wMax = 5.0              :projection
"""

ffSyn = Synapse( parameters = parameterFF,
    equations= equatSTDP, 
    pre_spike='''g_target += w''')


#------STDP Synapse like above, with other parameters for input -> inhibitory ----#
parameterInptInhib="""
urefsquare = 9.0        :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 7e-08 *5         :projection
aLTP = 9e-08 *5         :projection
wMin = 0.0              :projection
wMax = 5.0              :projection
"""

ff2Syn = Synapse(parameters = parameterInptInhib,
    equations=equatSTDP,
    pre_spike = '''g_target +=w ''')          
#------STDP Synapse like above, with other parameters for exitatory -> inhibitory ----#
parameterInhib="""
urefsquare = 9.0        :projection
thetaLTD = -70.6        :projection
thetaLTP = -45.3        :projection
aLTD = 7e-08 *5         :projection
aLTP = 7e-08 *5         :projection
wMin = 0.0              :projection
wMax = 1.0              :projection
"""

InhibSyn = Synapse( parameters = parameterInhib,
    equations=equatSTDP , 
    pre_spike='''g_target += w''')

 
##-> delete later, if not necessary
#------------- iSTDP Synapse for Inhibitory- to Exitatory- Layer -----------------------#

equatInhib = ''' w = w :min=hMin, max=hMax
                 dtracePre /dt = - tracePre/ taupre
                 dtracePost/dt = - tracePost/ taupre'''

parameter_iSTDPback="""
taupre = 175                 :projection
aPlus = 7.0*10**(-6)         :projection
Offset = 12.0                :projection
hMin=0.0                     :projection
hMax =1.0                    :projection
"""
inhibBackSynapse = Synapse(parameters = parameter_iSTDPback,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)
                         tracePost += 1''')
                         

#------------------- iSTDP Synapse for Lateral Inhibitory ----------------------------#

parameter_iSTDPlat="""
taupre = 100                :projection
aPlus = 7.0*10**(-6)        :projection
Offset = 10.0               :projection
hMin=0.0                    :projection
hMax =1.0                   :projection
"""
inhibLatSynapse = Synapse(parameters = parameter_iSTDPlat,
                    equations = equatInhib,
                    pre_spike ='''
                         g_target +=w
                         w+= aPlus * (tracePost - Offset)
                         tracePre  += 1 ''',
                    post_spike='''
                         w+=aPlus * (tracePre)
                         tracePost += 1''')

#-----------------------population defintions-----------------------------------#
#popInput = PoissonPopulation(geometry=(patchsize,patchsize,2),rates=50.0) #spike source array population --> poisson löschen
initList = []
for i in range(patchsize*patchsize*2):
    initList.append([])
popInput = SpikeSourceArray(spike_times=initList)
popLGN = Population(geometry=(patchsize,patchsize,2),neuron=spkNeurLGN ) #ann macht 12x12x2 als eindimensionale liste mit on/off abwechselnd
popV1 = Population(geometry=n_exN,neuron=spkNeurV1, name="V1")
popInhibit = Population(geometry=n_inN, neuron = spkNeurV1)
#-----------------------projection definitions----------------------------------
#projPreLayer_PostLayer

#klappt
projInput_LGN = Projection(
    pre = popInput,
    post = popLGN,
    target = 'exc',
    synapse = inputSynapse
).connect_one_to_one(weights = 30.0)

#Klappt

projLGN_V1 = Projection(
    pre=popLGN, 
    post=popV1, 
    target='Exc',
    synapse=ffSyn
).connect_all_to_all(weights = Uniform(0.015,2.0))

projLGN_Inhib = Projection(
    pre = popLGN,
    post= popInhibit,
    target='Exc',
    synapse=ff2Syn
).connect_all_to_all(weights = Uniform(0.0175,1.5))

projV1_Inhib = Projection(
    pre = popV1,
    post = popInhibit,
    target = 'Exc',
    synapse = InhibSyn
).connect_all_to_all(weights = Uniform(0.0175,0.25))
#klappt

projInhib_V1 = Projection(
    pre = popInhibit,
    post= popV1,
    target = 'Inh',
    synapse = inhibBackSynapse
).connect_all_to_all(weights = 0) #recording Uniform(0,1)

projInhib_Lat = Projection(
    pre = popInhibit,
    post = popInhibit,
    target = 'Inh',
    synapse = inhibLatSynapse
).connect_all_to_all(weights = 0)#Uniform(0.0,0.8))

#----------------------------further functions---------------------------------
def createDir():
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/V1toIN'):
        os.mkdir('output/V1toIN')
    if not os.path.exists('output/exitatory'):
        os.mkdir('output/exitatory')
    if not os.path.exists('output/inhibitory'):
        os.mkdir('output/inhibitory')
    if not os.path.exists('output/V1Layer'):
        os.mkdir('output/V1Layer')
    if not os.path.exists('output/InhibitLayer'):
        os.mkdir('output/InhibitLayer')

    if not os.path.exists('input'):
        os.mkdir('input')

#------------------------------------------------------------------------------
def normWeights():
    #print('Norm the Weights!')
    weights= projLGN_V1.w
    for i in range(n_exN):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_V1.w = weights
    weights = projLGN_Inhib.w
    for i in range(n_inN):
        onoff  = np.reshape(weights[i],(patchsize,patchsize,2))
        onNorm = np.sqrt(np.sum(onoff[:,:,0]**2))
        offNorm= np.sqrt(np.sum(onoff[:,:,1]**2))
        onoff[:,:,0]*=offNorm/onNorm
        weights[i] = np.reshape(onoff,patchsize*patchsize*2)
    projLGN_Inhib.w = weights

#-------------------------------------------------------------------------------
def saveWeights(nbr =0):
    np.savetxt('output/exitatory/V1weight_{}.txt'.format(nbr)  ,projLGN_V1.w)
    np.savetxt('output/exitatory/InhibW_%i.txt'%(nbr) ,projLGN_Inhib.w)
    np.savetxt('output/V1toIN/V1toIN_%i.txt'%(nbr)   ,projV1_Inhib.w)
    np.savetxt('output/inhibitory/INtoV1_%i.txt'%(nbr) ,projInhib_V1.w)
    np.savetxt('output/inhibitory/INLat_%i.txt'%(nbr)  ,projInhib_Lat.w)

#-------------------------------------------------------------------------------
def preprocess(data):
    cube = np.zeros((34,34,duration), dtype='int32') # hard coded?
    single_evbs = data
    label = single_evbs[1]
    single_evbs = single_evbs[0]
    start_t = np.random.randint(0,351-duration)
    end_t = start_t + duration

    for s in single_evbs:
        x = s[0]
        y = s[1]
        t = int(s[2]//1000) # set to ms
        p = s[3]
        if ((t >=start_t ) and (t < end_t)):
            if p == 1:
                cube[y,x,t-start_t] = 1
            else:
                cube[y,x,t-start_t] = -1

    return(cube)
#-------------------------------------------------------------------------------
def getInputBin(event_data, duration, patchsize):
    event_data = preprocess(event_data)# create a cube
    inp_bin, start_x, start_y = Make_lilarray(event_data, patchsize, duration)
    return(inp_bin, start_x, start_y)
#-------------------------------------------------------------------------------
def make_spike_pic(population, spikes,text_file_counter):
    #--lil BILD Array Werte def--
    lil_x_pos = []
    lil_y_pos = []
    lil_vals = []
    time_s = []
    
    #LIL ARRAY FÜLLEN 
    for i in range(288):
        if spikes[i] != [1]:
            for j in range(len(spikes[i])):
                time_s.append(spikes[i][j])
                x,y,pol = population.coordinates_from_rank(i)
                lil_x_pos.append(x)
                lil_y_pos.append(y)
                if pol == 0:
                    lil_vals.append(1)
                else: 
                    lil_vals.append(0)
                    
    '''EVENTS VON FLOAT TO INT'''
    lil_x_pos = np.asarray(lil_x_pos)
    lil_y_pos = np.asarray(lil_y_pos)
    lil_vals = np.asarray(lil_vals)
    time_s = np.asarray(time_s)
    
    lil_x_pos = lil_x_pos.astype(int)
    lil_y_pos = lil_y_pos.astype(int)
    lil_vals = lil_vals.astype(int)
    time_s = time_s.astype(int)

    
    batch_size = 50# one batch contains events of 100/10/1 ms 
    evt_imgs = np.zeros((1,12,12,2)) #1 für intitialisierungszwecke --> 4 dimensionale matrize voller 0en
    
    cnt = 0
    for s in range(len(lil_x_pos)):
        if time_s[s] != 1:
            #print(time_s[s])

            if time_s[s] > (cnt*batch_size): #erstelle neue zeitscheibe wenn intervall erreicht
                img = np.zeros((1,12,12,2))
                evt_imgs = np.append(evt_imgs,img,axis=0)
                cnt +=1 
            evt_imgs[cnt,int(lil_y_pos[s]-1),int(lil_x_pos[s]-1),int(lil_vals[s])] = 1
    
    #print(np.shape(evt_imgs))
    
    
    for i in range(cnt+1):
        pt.figure()
        pt.imshow(evt_imgs[i,:,:,0] - evt_imgs[i,:,:,1],cmap='gray',vmin=-1,vmax=1,interpolation='none')
        pt.colorbar() 
        pt.savefig('./output/test_img%i'%(i))
        pt.close()

#------------------------------main function------------------------------------
def run():
    createDir()
    compile() # compile the complete network


    # activate the synaptic plasticity in all projections
    projLGN_Inhib.plasticity = True
    projV1_Inhib.plasticity = True
    projInhib_V1.plasticity = True
    projInhib_Lat.plasticity = True 
    #------- SPIKE MONITOR --------#
    spike_popLGN = Monitor(popLGN, ['spike'])
    #------- neuron Monitors --------#
    V1MonP = Monitor(popV1,['vm','vmean','umeanLTD','umeanLTP'],period=int(duration/2))
    InhibMonP = Monitor(popInhibit,['vm','vmean'],period=int(duration/2))
    V1Mon = Monitor(popV1,['g_Exc','g_Inh','spike'])
    InhibMon=Monitor(popInhibit,['g_Exc','g_Inh','spike'])
    #--------synapse Monitors--------#
    dendriteFF = projLGN_V1.dendrite(0)
    ffWMon = Monitor(dendriteFF,['w','ltdTerm','ltpTerm'],period=int(duration/2))
    dendriteFFI = projLGN_Inhib.dendrite(0)
    ffIMon= Monitor(dendriteFFI,'w',period=int(duration/2))
    dendriteExIn = projV1_Inhib.dendrite(0)
    exInMon = Monitor(dendriteExIn,'w',period=int(duration/2))
    dendriteInEx = projInhib_V1.dendrite(0)
    inExMon = Monitor(dendriteInEx,'w',period=int(duration/2))
    dendriteInLat = projInhib_Lat.dendrite(0)
    inLatMon = Monitor(dendriteInLat,'w',period=int(duration/2))

    #------Objects to save different variables--------#
    rec_frEx = np.zeros((nbrOfPatches,n_exN))
    rec_V1_gExc=np.zeros((nbrOfPatches,n_exN))
    rec_V1_gInh=np.zeros((nbrOfPatches,n_exN))
    rec_frInh= np.zeros((nbrOfPatches,n_inN))
    rec_Inhib_gExc=np.zeros((nbrOfPatches,n_inN))
    rec_Inhib_gInh=np.zeros((nbrOfPatches,n_inN))
    t1 = time() # start time counter to measure the computational time
    
    #------Objects to save Spike Monitor--------#
    rec_m_spikes_popLGN = np.zeros((nbrOfPatches, inputNeurons))


    # load the training set
    data_train = tonic.datasets.NMNIST('./input/', train=True)

    print('Start simulation')

    for i in tqdm(range(nbrOfPatches),ncols=80):
        # reset the input population
        popInput.reset() 
        # choose a random sample
        train_sample = np.random.randint(0,60000)
        train_sample = data_train[train_sample]

        # create the input spike arrays
        inp_bin, start_x, start_y = getInputBin(duration = duration, patchsize = patchsize, event_data = train_sample)
        
        # load the input spike arrays into the input population
        popInput.spike_times = inp_bin

        # simulate 
        simulate(duration)
         
        # norm weights every 20s as mentioned in Clopath et al. (2010)
        if ((i*duration)%20000) == 0: 
            normWeights() 
        
        # get the population data from the monitors and save them
        spikesEx = V1Mon.get('spike')
        gExcEx = V1Mon.get('g_Exc')
        gInhEx = V1Mon.get('g_Inh')
        spikesInh = InhibMon.get('spike')
        gExcInh = InhibMon.get('g_Exc')
        gInhInh = InhibMon.get('g_Inh')

        ## iterate over all neurons to save the input currents and firing rate in Hz
        for j in range(n_exN):
            rec_V1_gExc[i,j] = np.mean(gExcEx[:,j])
            rec_V1_gInh[i,j] = np.mean(gInhEx[:,j])
            rateEx = len(spikesEx[j])*1000/duration
            rec_frEx[i,j] = rateEx
            if (j < (n_inN)):
                rec_Inhib_gExc[i,j]=np.mean(gExcInh[:,j])
                rec_Inhib_gInh[i,j]=np.mean(gInhInh[:,j])
                rateInh = len(spikesInh[j])*1000/duration
                rec_frInh[i,j] = rateInh     
        
        # save the weights from time to time    
        if((i%(nbrOfPatches/10)) == 0):
            saveWeights(i)  

                
        reset(monitors=False)
    t2 = time()

    ## save the final weights
    saveWeights(nbrOfPatches)

    #------get recorded data---------#
    
    ffW = ffWMon.get('w')
    ffLTD = ffWMon.get('ltdTerm')
    ffLTP = ffWMon.get('ltpTerm')
    ffI = ffIMon.get('w')
    exInW = exInMon.get('w')
    inExW = inExMon.get('w')
    inLatW= inLatMon.get('w')

    l2VM = V1MonP.get('vm')
    l2vMean =  V1MonP.get('vmean')
    l2umeanLTD = V1MonP.get('umeanLTD')
    l2umeanLTP = V1MonP.get('umeanLTP')
    iLvMean =InhibMonP.get('vmean')
    iLVM = InhibMonP.get('vm')


    #--------print Time difference-----------#
    print("time of simulation= "+str((duration*nbrOfPatches)/1000)+" s")
    print("time of calculation= "+str(t2-t1)+" s")
    print("factor= "+str((t2-t1)/(duration*nbrOfPatches/1000)))
    
  #----------------plot output---------------#

    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(l2umeanLTD[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(l2umeanLTD[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(l2umeanLTD[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(l2umeanLTD[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/l2umeanLTD_'+str(i)+'.png')

    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(l2umeanLTP[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(l2umeanLTP[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(l2umeanLTP[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(l2umeanLTP[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/l2umeanLTP_'+str(i)+'.png')


    for i in range(1):
        fig = mp.pyplot.figure()
        fig.add_subplot(4,1,1) 
        mp.pyplot.plot(rec_frEx[:,0+(4*i)])
        fig.add_subplot(4,1,2) 
        mp.pyplot.plot(rec_frEx[:,1+(4*i)])
        fig.add_subplot(4,1,3) 
        mp.pyplot.plot(rec_frEx[:,2+(4*i)])
        fig.add_subplot(4,1,4) 
        mp.pyplot.plot(rec_frEx[:,3+(4*i)])
        mp.pyplot.savefig('output/V1Layer/frEx_'+str(i)+'.png')


    plt.plotgEx(rec_V1_gExc,'Exi')
    plt.plotgEx(rec_Inhib_gExc,'Inhib')
    plt.plotgInh(rec_V1_gInh,'Exi')
    plt.plotgInh(rec_Inhib_gInh,'Inhib')
    mp.pyplot.close('all')

    plt.plotVM(l2VM,'Exi')
    plt.plotVM(iLVM,'Inhib')
    plt.plotVMean(l2vMean,'Exi')
    plt.plotVMean(iLvMean,'Inhib')
    mp.pyplot.close('all')

    mp.pyplot.figure()
    mp.pyplot.plot(np.mean(rec_V1_gExc,axis=1),label='gExc')
    mp.pyplot.plot(-1*np.mean(rec_V1_gInh,axis=1),label='-gInh')
    mp.pyplot.plot(np.mean(rec_V1_gExc,axis=1)-np.mean(rec_V1_gInh,axis=1),label= 'gExc-gInh')
    mp.pyplot.legend()
    mp.pyplot.savefig('output/gExc_gInh.png')

    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(ffW,axis=1))
    mp.pyplot.savefig('output/ffWMean.png')
    
    #OUTPUT BILDER ERSTELLEN
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(ffLTD,axis=1))
    mp.pyplot.savefig('output/ffLTD.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(ffLTP,axis=1))
    mp.pyplot.savefig('output/ffLTP.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(ffI,axis=1))
    mp.pyplot.savefig('output/ffI.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(exInW,axis=1))
    mp.pyplot.savefig('output/exInW.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(inExW,axis=1))
    mp.pyplot.savefig('output/inExW.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(inLatW,axis=1))
    mp.pyplot.savefig('output/inLatW.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(l2VM,axis=1))
    mp.pyplot.savefig('output/l2VM.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(l2vMean,axis=1))
    mp.pyplot.savefig('output/l2vMean.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(l2umeanLTD,axis=1))
    mp.pyplot.savefig('output/l2umeanLTD.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(l2umeanLTP,axis=1))
    mp.pyplot.savefig('output/l2umeanLTP.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(iLvMean,axis=1))
    mp.pyplot.savefig('output/iLvMean.png')
    
    mp.pyplot.figure()   
    mp.pyplot.plot(np.mean(iLVM,axis=1))
    mp.pyplot.savefig('output/iLVM.png')
    
    print("finish")
    
#------------------------------------------------------------------------------------


if __name__ == "__main__":
    run()

