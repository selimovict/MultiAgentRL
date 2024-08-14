import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import copy
import math
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.callbacks import CSVLogger

def GetRewardNum(Fix,IsUndefined,MaxCoef):
    if IsUndefined:
        if Fix==1:
            return 100
        elif Fix==2:
            return 95
        elif Fix==3:
            return 90
        elif Fix==0:
            return -500
        else:
            return -10
    else:
        if MaxCoef<=0.6 *3:
            if Fix==1:
                return 200*1  #200
            elif Fix==2:
                return  160*1 #160
            elif Fix==3:
                return  120*1 #120
            elif Fix == 0:
                return -500
            else:
                return -20

        elif  0.6*3 <MaxCoef and MaxCoef<=3:

            if Fix==1:
                return 200*0.8 #160
            elif Fix==2:
                return  160*0.8 #128
            elif Fix==3:
                return  120*0.8 #96
            else:
                return -30

        elif 3 < MaxCoef and MaxCoef <= 2 * 3:
            if Fix == 0:
                return 600
            elif Fix == 1:
                return 200 * 0.6 #120
            elif Fix == 2:
                return 160 * 0.6 #96
            elif Fix == 3:
                return 120 * 0.6 #72
            else:
                return -40

        elif MaxCoef>2 * 3:
            if Fix==0:
                return 900
            else:
                return -60






def SetRewards(Bt_mat,IsUndefined,MaxCoef):

    Bt_mat_Rooms = [p for p in copy.deepcopy(Bt_mat) if p[0] > 0]
    Fix = 1
    #MaxCoefRooms = Bt_mat_Rooms[0][1] / Bt_mat_Rooms[1][1]
    #print('Max Coef ',MaxCoef)

    for index, Room in enumerate(Bt_mat_Rooms):
        if (index > 0):
            index0 = index - 1
            raz = abs(Bt_mat_Rooms[index][1] - Bt_mat_Rooms[index0][1])
            if raz < 0.0001:
                Room.append(Fix)

            else:
                Fix = Fix + 1
                Room.append(Fix)

        else:
            Room.append(Fix)


    OnesLen = len([X for X in Bt_mat_Rooms if X[2] == 1])

    StateReward=False
    if OnesLen>0.7*len(Bt_mat_Rooms):
        StateReward=True

    #Bt_mat_Rooms
    #TABLE  [INDEX,Bt,RowNum,Reword]

    if StateReward==False or StateReward==True:
        for x in Bt_mat_Rooms:
            R = GetRewardNum(x[2], IsUndefined, MaxCoef)
            x.append(R)




    Bt_mat_Undefined = [p for p in copy.deepcopy(Bt_mat) if p[0] == 0]
    Bt_mat_Undefined[0].append(0)
    R = GetRewardNum(0, IsUndefined, MaxCoef)
    Bt_mat_Undefined[0].append(R)
    Bt_mat_Rooms.append(Bt_mat_Undefined[0])
    #print('Bt_mat_Undefined',Bt_mat_Undefined)
    return Bt_mat_Rooms




def GetBeliefMatrix(Belief_Val):
    Bt = Belief_Val#list(Belief_Val.histogram.values())
    Bt_Orig=copy.deepcopy(Bt)
    Bt_copy = np.array(copy.deepcopy(Bt))
    Bt_copy= -np.sort(-Bt_copy)
    Bt_mat=[]

    for index,elem in enumerate(Bt_copy):
        orig_index=Bt_Orig.index(elem)
        Bt_Orig[orig_index]= - 1000
        Bt_mat.append([orig_index, elem])


    IsUndefined=False
    if Bt_mat[0][0] == 0:
        IsUndefined=True

    MaxCoeff=Bt_mat[0][1]/Bt_mat[1][1]

    print('IsUndefined',IsUndefined)
    print('MaxCoeff',MaxCoeff)


    Bt_mat_Rewards=SetRewards(Bt_mat,IsUndefined,MaxCoeff)

    return  Bt_mat_Rewards,IsUndefined


if __name__ == '__main__':


    InitBeliefVector=[0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    SecondBeliefVector=[0.3214382033486859, 0.06623583761683914, 0.038984036204822874, 0.05969566661826297, 0.06623583761683914, 0.038984036204822874, 0.059695666618262956, 0.059695666618262956, 0.059695666618262956, 0.046615324621110656, 0.18272405791382765]

    Bt1=[0.01704555383162602, 0.16826249701922774, 0.0032155297091683785, 0.04249206821000775, 1.99488753967732764, 0.059441004268443676, 0.08100437285648063, 0.08344101980491309, 0.016879976927144998, 0.06518114824162995, 0.01814928945403011]

    RwardMat,IsUndefined=GetBeliefMatrix(SecondBeliefVector )