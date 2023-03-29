#-*- coding:utf-8 -*-

#IMPORT 
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

#define function
def Make_lilarray(datei,patchsize,duration):
	
	#BIG ARRAY EINLESEN
	big_array = datei
	#big_array = big_array.f.arr_0
	
	# LIL ARRAY INIT
	w,h,time = np.shape(big_array)
	lil_array_on = np.zeros((patchsize, patchsize, duration))
	lil_array_off = np.zeros((patchsize, patchsize, duration))

	#random Block auswählen --> Rand weg
	x_start = np.random.randint(5,w-patchsize-5)
	y_start = np.random.randint(5,w-patchsize-5)
	lil_zeit = 0
	#LIL ARRAY FÜLLEN
	for zeit in range(1,duration):  
		lil_zeit += 1
		for x_achse in range(x_start,x_start+patchsize):
			for y_achse in range(y_start,y_start+patchsize):
			#Trennen in ONN/OFF ARRAY
				if big_array[x_achse,y_achse,zeit] < 0.0:
					lil_array_off[x_achse-x_start, y_achse-y_start,lil_zeit] = big_array[x_achse,y_achse,zeit]
				elif big_array[x_achse,y_achse,zeit] > 0.0:
					lil_array_on[x_achse-x_start, y_achse-y_start,lil_zeit] = big_array[x_achse,y_achse,zeit]
					
	#Lil array zufällig flippen aber raus weil Werte überprüfen
	
	if np.random.rand() < 0.5:
		lil_array_on = np.fliplr(lil_array_on)
		lil_array_off = np.fliplr(lil_array_off)
	if np.random.rand() < 0.5:
		lil_array_on = np.flipud(lil_array_on)
		lil_array_off = np.flipud(lil_array_off)
    # and randomly transpose	
	if np.random.rand() < 0.5:
		lil_array_on = np.transpose(lil_array_on,axes = (1,0,2))
		lil_array_off = np.transpose(lil_array_off,axes = (1,0,2))	
	#SPIKE SOURCE ARRAY BAUEN	
	lil_array_on = np.reshape(lil_array_on,(patchsize*patchsize,duration))
	lil_array_off = np.reshape(lil_array_off,(patchsize*patchsize,duration))


	spike_array = []	
	for i in range(np.shape(lil_array_on)[0]):
		t_s = np.where(lil_array_on[i] == 1)[0]
		t_s2 = np.where(lil_array_off[i] == -1)[0]
		t_s = t_s.tolist()
		t_s2 = t_s2.tolist()
		spike_array.append(t_s)
		spike_array.append(t_s2)
		
	return(spike_array, x_start, y_start)
		
