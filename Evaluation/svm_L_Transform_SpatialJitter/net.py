from ANNarchy import *

patchsize = 12 
inputNeurons = patchsize*patchsize*2
n_exN = 144#patchsize*patchsize 144
n_inN = int(n_exN/4)


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
#VTrest = -50.4
#-82.363
#if state==1:-82.363 else:
#dg_Exc/dt = 1/tau_gExc * (-g_Exc)
#dg_Inh/dt = 1/tau_gInh*(-g_Inh)
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

inputSynapse =  Synapse(
    parameters = "",
    equations = "",
    pre_spike = """
        g_target += w
                """
)

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
    synapse=inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.0))

projLGN_Inhib = Projection(
    pre = popLGN,
    post= popInhibit,
    target='Exc',
    synapse=inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1))

projV1_Inhib = Projection(
    pre = popV1,
    post = popInhibit,
    target = 'Exc',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.))
#klappt

projInhib_V1 = Projection(
    pre = popInhibit,
    post= popV1,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.)) #recording Uniform(0,1)

projInhib_Lat = Projection(
    pre = popInhibit,
    post = popInhibit,
    target = 'Inh',
    synapse = inputSynapse
).connect_all_to_all(weights = Uniform(0.0,1.))

def loadWeights():
    projLGN_V1.w = np.loadtxt('network_data/V1weight.txt')
    projLGN_Inhib.w =np.loadtxt('network_data/InhibW.txt')
    projV1_Inhib.w = np.loadtxt('network_data/V1toIN.txt')
    projInhib_V1.w = np.loadtxt('network_data/INtoV1.txt')
    projInhib_Lat.w =np.loadtxt('network_data/INLat.txt')
