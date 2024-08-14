from pomdpx_parser import pomdpx_parser as parser
import xml.etree.ElementTree as ET
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy
import yaml
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
import pickle
import re

def GetActionNum(ActionName):
    if ActionName=='end':
        return 0
    else:
        niz=re.findall(r'\d+', ActionName)
        return int(niz[0])

def GetLocStateNum(LocStateName):
    if LocStateName=='h':
        return -1
    else:
        niz=re.findall(r'\d+', LocStateName)
        return int(niz[0])

def GetFireStateNum(FireStateName):
    if FireStateName=='u':
        return -1
    else:
        niz=re.findall(r'\d+', FireStateName)
        return int(niz[0])

class GeneralCustomEnvironment():
    def __init__(self, General_True_State, TranSitionModel, RewardModel):

        self.NumberOfEns=len(General_True_State)
        self.Envs=[]

        for i in range(self.NumberOfEns):
            self.Envs.append(
                CustomEnvironment(General_True_State[i],
                                  TranSitionModel,
                                  RewardModel)
                            )
        self.state=[x.state for x in self.Envs]

    def Reset(self):
        for i in range(self.NumberOfEns):
            self.Envs[i].apply_transition(AgentState('u'))


        StList = self.Envs[0].transition_model.get_all_states()
        StList.remove(AgentState('u'))
        next_state = random.choice(StList)

        SelectedSubEnv=random.randrange(self.NumberOfEns)
        self.Envs[SelectedSubEnv].apply_transition(next_state)
        self.state=[x.state for x in self.Envs]



class CustomEnvironment(pomdp_py.Environment):

    def __init__(self,init_true_state,TransitionModel, RewardModel):
        super().__init__(init_true_state,TransitionModel, RewardModel)
        print('Environment Defined With TrueState:'+init_true_state.name)

    def Reset(self):
        StList=self.transition_model.get_all_states()
        StList.remove(AgentState('u'))
        next_state = random.choice(StList)
        self.apply_transition(next_state)



    def TakeAction(self,Agent_Actions):
        return self.reward_model.join_reward_func(self.state, Agent_Actions)

class CustomAgent(pomdp_py.Agent):

    def __init__(self, init_belief, PolicyModel,TransitionModel,ObservationModel,RewardModel,AgentName,Model='',Epsilon=1.0):

        super().__init__(init_belief, PolicyModel,TransitionModel,ObservationModel,RewardModel)
        self.ACTIONS = PolicyModel.get_all_actions()
        root_model = ET.parse('POMDP_Models/'+AgentName+'_10r.pomdpx').getroot()
        self.RM = parser.get_reward_model(root_model)
        self.AGENT_STATES = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'h'];
        self.AGENT_ACTIONS = ['scout1', 'scout2', 'scout3', 'scout4', 'scout5', 'scout6', 'scout7', 'scout8', 'scout9', 'scout10', 'end'];
        self.F_STATES = ['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

        self.NumOfStates=len(self.AGENT_STATES)
        self.NumOfBeliefs= len(self.F_STATES)
        self.NumOfActions= len(self.AGENT_ACTIONS )
        self.MyScouting = np.zeros(self.NumOfActions-1)
        self.cur_agent_state='h'
        self.cur_agent_state_index=self.AGENT_STATES.index(self.cur_agent_state)
        self.cur_agent_state_vector=np.zeros(len(self.AGENT_STATES))
        self.cur_agent_state_vector[self.cur_agent_state_index]=1
        self.Reseted = True
        self.AgentName=AgentName
        self.my_init_belief=init_belief
        self.gamma = 0.001; #0.3
        self.epsilon = Epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001
        self.alpha_decay = 0.0001
        self.batch_size = 64
        self.K=3 #NUMBER OF MAX Bt VALUE HAVE TO BE HEIGHER OF SECOND MAX
        self.InMission=False
        self.RW=0
        self.r_lr=0.98
        self.LocalHistory=[]
        self.BeliefHistory=[]
        self.ActionHistory=[]
        self.OldBelief=init_belief
        self.OldOldBelief = init_belief

        self.Action=None
        self.Old_Action=None
        self.Consensus=None

        self.Reseted=True

        self.StatesHistory=[]
        self.BeliefHistory=[]
        self.ConensusHistory=[]
        print('Agent is defined with name:'+self.AgentName)
        self.AgentsDistributionMap=None
        self.AgentsStateMap = None
        self.InitialAgentsDistributionMap=None
        if Model!='':
            self.LoadPolicy(Model)
        else:
            self.model = Sequential()
            self.model.add(Dense(32, input_dim=self.NumOfStates+self.NumOfBeliefs, activation='relu',name="data_input"))
            self.model.add(Dense(64, activation='sigmoid'))
            self.model.add(Dense(128, activation='relu'))
            #self.model.add(layers.Embedding(input_dim=5, output_dim=8))
            #self.model.add(layers.LSTM(16))

            #self.model.add(layers.Embedding(input_dim=5, output_dim=16))
            #self.model.add(layers.LSTM(32))

            #self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.2))

            self.model.add(Dense(64, activation='sigmoid'))
            self.model.add(Dense(128, activation='sigmoid'))
            self.model.add(Dense(self.NumOfActions, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))

    def SetInitialAgetnsDistributionMap(self,InitialAgentsDistributionMap):
        self.InitialAgentsDistributionMap=InitialAgentsDistributionMap

    def SetAgentsDistributionMap(self,AgentsDistributionMap):
        self.AgentsDistributionMap=AgentsDistributionMap

    def SetAgentsStateMap(self, AgentsStateMap):
        self.AgentsStateMap = AgentsStateMap

    def SetAgentsStateMapCoef(self, AgentsStateMapCoef):
        self.AgentsStateMapCoef = AgentsStateMapCoef

    def SaveLocalHistory(self,FilePath):
        file = open(FilePath, 'wb')
        pickle.dump(self.LocalHistory, file)

    def LoadLocalHistory(self,FilePath):
        file = open(FilePath, 'rb')
        self.LocalHistory=pickle.load(file)

    def GetBelief(self,Bt):
        init_belief = pomdp_py.Histogram({
            AgentState("u"): Bt[0],
            AgentState("f1"): Bt[1],
            AgentState("f2"): Bt[2],
            AgentState("f3"): Bt[3],
            AgentState("f4"): Bt[4],
            AgentState("f5"): Bt[5],
            AgentState("f6"): Bt[6],
            AgentState("f7"): Bt[7],
            AgentState("f8"): Bt[8],
            AgentState("f9"): Bt[9],
            AgentState("f10"): Bt[10]
        })
        return init_belief

    def StateInfo(self):
        print(self.AgentName + ' ----- STATE INFO ----- ')
        Bt = copy.deepcopy(list(self.cur_belief.histogram.values()))

        RewardMat, IsUndefined, MaxCoef = self.GetBeliefMatrix(Bt)
        print('                 IS UNDEFINED:',IsUndefined)
        print('                 MAX COEF:', MaxCoef)
        print('                 REWARD MAT')
        for X in RewardMat:
            print('                 ',X)
        #print((RewardMat))

        #Bt.sort()
        #coef = Bt[-1]/Bt[-2]
        #coef2= Bt[-2]/Bt[-3]
        #coef3 =Bt[-3]/Bt[-4]


        #StateIndx=list(self.cur_belief.histogram.values()).index(Bt[-1])
        #State=self.F_STATES[StateIndx]

        #StateIndx2=list(self.cur_belief.histogram.values()).index(Bt[-2])
        #State2=self.F_STATES[StateIndx2]

        #StateIndx3=list(self.cur_belief.histogram.values()).index(Bt[-3])
        #State3=self.F_STATES[StateIndx3]

        #print('                 Coef1:'+str(coef)+'  State1:'+str(State)+' Bt1:'+str(Bt[-1]))
        #print('                 Coef2:' + str(coef2) + '  State2:' + str(State2) + ' Bt2:' + str(Bt[-2]))
        #print('                 Coef3:'+str(coef3)+'  State3:'+str(State3)+' Bt3:'+str(Bt[-3]))

        print('                 cur_belief:' + str(self.cur_belief.histogram.values()) )




    def Reset(self,InitialDistributionMap=None):
        self.Reseted=True
        self.InMission=True
        self.set_belief(self.my_init_belief)
        self.OldBelief=self.my_init_belief
        self.OldOldBelief = self.my_init_belief
        self.MyScouting = np.zeros(self.NumOfActions - 1)
        self.cur_agent_state='h'
        self.cur_agent_state_index=self.AGENT_STATES.index(self.cur_agent_state)
        self.cur_agent_state_vector=np.zeros(len(self.AGENT_STATES))
        self.cur_agent_state_vector[self.cur_agent_state_index]=1
        self.InitialAgentsDistributionMap=InitialDistributionMap
        self.AgentsDistributionMap=InitialDistributionMap
        self.AgentsStateMap=None
        self.AgentsStateMapCoef=None
    def ForgetHistory(self):
        self.LocalHistory=[]

    def choose_action(self,AgentState,Cur_Belief):
        if self.Reseted :
            self.Reseted = False
        Processed_State = self.preprocess_state(AgentState,list(Cur_Belief.histogram.values()))
        RandomAction = random.choice(self.AGENT_ACTIONS)
        ModelAction = self.AGENT_ACTIONS[np.argmax(self.model.predict(Processed_State))]

        # agentstateIndex=self.AGENT_STATES.index(AgentState)
        # goodNextAction=agentstateIndex+1
        # if self.AgentName in ('agent1','agent3'):
        #     if agentstateIndex==10 or agentstateIndex==9:
        #         goodNextAction=0
        #     else:
        #         goodNextAction=agentstateIndex+1
        # elif self.AgentName in ('agent2'):
        #     if agentstateIndex == 10 or agentstateIndex == 0:
        #         goodNextAction = 9
        #     else:
        #         goodNextAction=agentstateIndex-1
        # if goodNextAction>10:
        #     print('d')

        #ProbArray=np.ones(11)
        #ProbArray=ProbArray*0.09
        #ProbArray[goodNextAction]=0.1

        #RandomAction=str(np.random.choice(self.AGENT_ACTIONS, 1, p=ProbArray)[0])

        if (np.random.random() <= self.epsilon):
            #R=self.GetAgentTotalReward(AgentState,self.cur_belief,AgentAction(RandomAction))
            #R=self.AgentReward(AgentState,AgentAction(RandomAction))
            #Rb=self.BeliefReward(self.cur_belief,AgentAction(RandomAction))
            #if Rb != 0 and R > 0:
            #    R = 0
            #elif R < 0:
            #    Rb = 0
            #elif R == 0 and Rb == 0:
            #    R = -19
            print('                 -----   RANDOM ACTION')
            return  AgentAction(RandomAction)
        else:
            #R = self.GetAgentTotalReward(AgentState, self.cur_belief, AgentAction(RandomAction))
            #R = self.AgentReward(AgentState, AgentAction(ModelAction))
            #Rb = self.BeliefReward(self.cur_belief, AgentAction(ModelAction))
            #if Rb != 0 and R > 0:
            #    R = 0
            #elif R < 0:
            #    Rb = 0
            #elif R == 0 and Rb == 0:
            #    R = -19
            print('                 -----   MODEL ACTION')
            return  AgentAction(ModelAction)

    def preprocess_state(self,AgentState,AgemtBeliefs):
        cur_agent_state_index=self.AGENT_STATES.index(AgentState)
        cur_agent_state_vector=np.zeros(self.NumOfStates)
        cur_agent_state_vector[cur_agent_state_index]=1

        return np.reshape(cur_agent_state_vector.tolist()+AgemtBeliefs, [1, self.NumOfStates+self.NumOfBeliefs])

    def GetRewardNum(self,Fix, IsUndefined, MaxCoef):
        if IsUndefined:
            if Fix == 1:
                return 20
            elif Fix == 2:
                return 15
            elif Fix == 3:
                return 10
            elif Fix == 0:
                return -500
            else:
                return -10
        else:
            if MaxCoef < 0.667 * self.K:
                if Fix == 1:
                    return 200 * 1  # 200
                elif Fix == 2:
                    return 160 * 1  # 160
                elif Fix == 3:
                    return 120 * 1  # 120
                #elif Fix == 0:
                #    return -500
                else:
                    return -100

            elif 0.667 * self.K <= MaxCoef and MaxCoef < self.K:

                if Fix == 1:
                    return 200   # 160
                elif Fix == 2:
                    return 160   # 128
                elif Fix == 3:
                    return 120   # 96
                elif Fix == 0:
                    return 100
                else:
                    return -100

            elif self.K <= MaxCoef :  # 2.5 Forsirati End - PROF
                if Fix == 0:
                    return 200
                elif Fix == 1:
                    return 120   # 120
                elif Fix == 2:
                    return 100   # 96
                else:
                    return -100


    def SetRewards(self,Bt_mat, IsUndefined, MaxCoef):

        Bt_mat_Rooms = [p for p in copy.deepcopy(Bt_mat) if p[0] > 0]
        Fix = 1
        # MaxCoefRooms = Bt_mat_Rooms[0][1] / Bt_mat_Rooms[1][1]
        # print('Max Coef ',MaxCoef)

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

        #OnesLen = len([X for X in Bt_mat_Rooms if X[2] == 1])



        # Bt_mat_Rooms
        # TABLE  [INDEX,Bt,RowNum,Reword]


        for x in Bt_mat_Rooms:
            R = self.GetRewardNum(x[2], IsUndefined, MaxCoef)
            x.append(R)

        Bt_mat_Undefined = [p for p in copy.deepcopy(Bt_mat) if p[0] == 0]
        Bt_mat_Undefined[0].append(0)
        R = self.GetRewardNum(0, IsUndefined, MaxCoef)
        Bt_mat_Undefined[0].append(R)
        Bt_mat_Rooms.append(Bt_mat_Undefined[0])
        # print('Bt_mat_Undefined',Bt_mat_Undefined)
        return Bt_mat_Rooms

    def LocationToAction(self,LocationID):
        ActionID=LocationID-1
        if LocationID==0:
            ActionID=10
        return  ActionID

    def GetBeliefMatrix(self,Belief_Val):
        Bt = Belief_Val#list(Belief_Val.histogram.values())
        Bt_Orig = copy.deepcopy(Bt)
        Bt_copy = np.array(copy.deepcopy(Bt))
        Bt_copy = -np.sort(-Bt_copy)
        Bt_mat = []

        for index, elem in enumerate(Bt_copy):
            orig_index = Bt_Orig.index(elem)
            Bt_Orig[orig_index] = - 1000
            Bt_mat.append([orig_index, elem])

        IsUndefined = False
        if Bt_mat[0][0] == 0:
            IsUndefined = True

        MaxCoeff = Bt_mat[0][1] / Bt_mat[1][1]

        #print('IsUndefined', IsUndefined)
        #print('MaxCoeff', MaxCoeff)

        Bt_mat_Rewards = self.SetRewards(Bt_mat, IsUndefined, MaxCoeff)

        for row in Bt_mat_Rewards:
            row.append(self.LocationToAction(row[0]))

        return Bt_mat_Rewards, IsUndefined,MaxCoeff

    def StateReward(self,agent_state,AgentAction):
        ## agent_state is location state ['r1','r2','r3', ... ]
        if AgentAction.name!='end' and agent_state[-1]==AgentAction.name[-1]:
            R=-500
        else:
            R=0
        return R

    def SequentialScountingReward(self,agent_state,AgentAction):
        agent_state_index=self.AGENT_STATES.index(agent_state)
        agen_action_index= self.AGENT_ACTIONS.index(AgentAction.name)
        R=copy.deepcopy(self.RM[0][agen_action_index][0][agent_state_index])
        R=R*10
        if(R==0):
            R=-10
        return R

    def AgentReward(self,agent_state,AgentAction):

        agent_state_index=self.AGENT_STATES.index(agent_state)
        agen_action_index= self.AGENT_ACTIONS.index(AgentAction.name)
        R=copy.deepcopy(self.RM[0][agen_action_index][0][agent_state_index])
        #if(R>=0 and R<6):
        #    R=0
        #elif R==6:
        R=R*10
        if AgentAction.name!='end' and agent_state[-1]==AgentAction.name[-1]:
            R=-500
        return R

    def BeliefReward2(self,Belief_Val_List,AgentAction):
        RewardMat,IsUndefined,MaxCoef = self.GetBeliefMatrix(Belief_Val_List)
        ActionID= self.AGENT_ACTIONS.index(AgentAction.name)

        # RewardMat   [Location,Bt,RowNum,RW,Scout]
        Belief_RW=[t for t in RewardMat if t[4]==ActionID][0][3]
        Ones=len([t for t in RewardMat if t[2]==1])
        IsConstant=False
        if Ones>(len(RewardMat)/2):
            IsConstant=True
        return Belief_RW,IsUndefined,IsConstant,MaxCoef

    def GetBestActionAccordintToReward(self,agent_state,agent_belief):

        Actions=[AgentAction(Ac) for Ac in  self.AGENT_ACTIONS]

        Reward=0
        BestAction=Actions[0]
        for Action in Actions:
            NewRew= self.GetAgentTotalReward(agent_state,agent_belief,Action)
            #print('Action'+Action.name+'   RW:'+str(NewRew))
            if NewRew>=Reward:
                Reward=NewRew
                BestAction=Action
        return BestAction

    def GetAgentTotalReward(self,agent_state,agent_belief,AgentAction):
        Belief_Val_List=list(agent_belief.histogram.values())
        #Reward get according to belief
        Belief_Reward,IsUndefined,IsConstant,MaxCoef =self.BeliefReward2(Belief_Val_List,AgentAction)
        #Reward get accourding to sequential scouting
        Scounting_Reward=self.SequentialScountingReward(agent_state,AgentAction)
        #Reward get according to scouting other room
        State_Reward=self.StateReward(agent_state,AgentAction)

        print('                 STATE:' + str(agent_state) + ' -ACTION:' + str(AgentAction) + '  -RW_secv:' + str(Scounting_Reward) + '  -RW_b:' + str(Belief_Reward)+ '  -RW_stat:' + str(State_Reward)  )


        if IsUndefined:
            #if IsConstant:
            #    print('                 RW_secv + RW_stat  ')
            #    return Scounting_Reward + State_Reward
            #else:
            print('                 RW_secv + RW_stat + RW_b ')
            return Belief_Reward+Scounting_Reward+State_Reward
        else:
            print('                 RW_b + RW_stat ')
            return Belief_Reward+State_Reward


    def BeliefReward(self,Belief_Val,AgentAction):

        Bt=list(Belief_Val.histogram.values())

        Bt_copy = copy.deepcopy(list(self.cur_belief.histogram.values()))
        Bt_copy.sort()
        coef = Bt_copy[-1] / Bt_copy[-2]
        RW=0
        MaxIndex = Bt.index(Bt_copy[-1])
        MaxIndex2 = Bt.index(Bt_copy[-2])
        MaxIndex3 = Bt.index(Bt_copy[-3])
        MaxIndex4 = Bt.index(Bt_copy[-4])

        coef2 = Bt_copy[-2] / Bt_copy[-3]

        ActionID =0
        if AgentAction.name!='end':
            ActionID=int(AgentAction.name[-1])

        if MaxIndex>0:
            if coef >= 2*self.K:
                if AgentAction.name=='end':
                    RW =  700
                else:
                    RW = -100
            if 2*self.K>coef and coef >=  self.K :  #Sigurno znam gdje je pojava
                if AgentAction.name=='end':
                    RW =  700
                elif ActionID==MaxIndex:
                    RW = 200 *0.4 # 80
                elif ActionID==MaxIndex2:
                    RW = 100 *0.4 #40
                elif ActionID==MaxIndex3:
                    RW = 50  *0.4 #20
                elif ActionID==MaxIndex4:
                    RW = 20  *0.4 #8
                else:
                    RW = -100
            elif  self.K > coef and coef >= 0.8 * self.K :
                if AgentAction.name=='end':
                    RW = 50
                elif ActionID==MaxIndex:
                    RW = 200 * 0.6 #120
                elif ActionID==MaxIndex2:
                    RW = 100 * 0.6 #60
                elif ActionID==MaxIndex3:
                    RW = 50 * 0.6  #30
                elif ActionID==MaxIndex4:
                    RW = 20 * 0.6  #12
                else:
                    RW = -80
            elif 0.8*self.K > coef and coef >= 0.6 * self.K :
                if AgentAction.name=='end':
                    RW = -80
                elif ActionID==MaxIndex:
                    RW = 200 * 0.8  #160
                elif ActionID==MaxIndex2:
                    RW = 160 * 0.8  #128
                elif ActionID==MaxIndex3:
                    RW = 120 * 0.8  #96
                elif ActionID==MaxIndex4:
                    RW = 90 * 0.8   #72
                else:
                    RW=-50

            elif  0.6 * self.K >coef:  #and coef >= 0.4* self.K :
                if AgentAction.name=='end':
                    RW = -200
                elif ActionID == MaxIndex:
                    RW = 200*1
                elif ActionID == MaxIndex2:
                    RW = 160*1
                elif ActionID == MaxIndex3:
                    RW = 120*1
                elif ActionID == MaxIndex4:
                    RW = 90*1
                else:
                    RW=-30

        elif MaxIndex==0  and coef2 > 1:   #Trenutno je undefiend max coef al idemo na drugi sljedeci
            if AgentAction.name=='end':
                RW =-500
            elif ActionID == MaxIndex2:
                RW = 200 * 1
            elif ActionID == MaxIndex3:
                RW = 160 * 1
            elif ActionID == MaxIndex4:
                RW = 120 * 1
            else:
                RW = -30


        elif MaxIndex==0 :
            if AgentAction.name=='end':
                RW =-500
            else:
                RW=0

        return  RW

#FF=ShareDate2(Agents[0].StatesHistory[-5],Agents[0].BeliefHistory[-5])
    #CONSENSUS 2

    def SetConsensusBelief(self,ResultBelief):
        #print(self.AgentName + ' ----- SET CON BELIEF ----- '+str(list(ResultBelief)))
        BeliefVector = self.GetBelief(ResultBelief)
        self.set_belief(BeliefVector)

        self.ConensusHistory.append(ResultBelief)


    def ShareData2(self,AllAgentsStateVectors,AllAgentBeliefVector):
        print('CONSENSUS -----  ----- ')

        print('       ----- STATES '+str(len(AllAgentsStateVectors)))
        print('       ----- BELIEF ' + str(len(AllAgentBeliefVector)))

        for p in range(len(AllAgentsStateVectors)):
            print('       ----- STATE ' + str(list(AllAgentsStateVectors[p])))
            print('       ----- BELIE ' + str(list(AllAgentBeliefVector[p])))

        AllStates=np.zeros(len(AllAgentsStateVectors[0]))
        for States in AllAgentsStateVectors:
            AllStates=AllStates+States

        NumOfAgents=len(AllAgentsStateVectors)
        ResultBelief=np.zeros(len(AllAgentsStateVectors[0]))
        #print('Result Belief')
        #print(ResultBelief)
        TakeInAccountCoef=0.8
        NotTakeInAccountCoef=0.2

        if NumOfAgents>1:

            for i in range(len(AllStates)):# i in (0,1,2,3,4....9,10)
                #print('i=',i)
                if AllStates[i]>0:
                    #ResultBelief[i]=
                    for j in range(NumOfAgents): # j in (0,1,2)
                        #print('j=',j)
                        if AllAgentsStateVectors[j][i]==1:
                            coefA=TakeInAccountCoef/AllStates[i]
                            ResultBelief[i]=ResultBelief[i]+coefA*AllAgentBeliefVector[j][i]
                        elif AllAgentsStateVectors[j][i]==0:
                            coefB = NotTakeInAccountCoef / (NumOfAgents-AllStates[i])
                            ResultBelief[i] = ResultBelief[i] + coefB * AllAgentBeliefVector[j][i]
                elif AllStates[i]==0:
                    for j in range(NumOfAgents):
                        coefC=1/NumOfAgents
                        ResultBelief[i]=ResultBelief[i]+coefC*AllAgentBeliefVector[j][i]

        if NumOfAgents==1:
            ResultBelief=AllAgentBeliefVector[0]

        ResultBelief=ResultBelief/sum(ResultBelief)
        BeliefVector = self.GetBelief(ResultBelief)
        self.set_belief(BeliefVector)

        self.ConensusHistory.append(ResultBelief)
        self.StatesHistory.append(AllAgentsStateVectors)
        self.BeliefHistory.append(AllAgentBeliefVector)

        print('       ----- RESULT ' + str(list(ResultBelief)))
        return ResultBelief


    def ShareData(self,AllAgentsStateVectors,AllAgentBeliefVector):
        Zeros=np.zeros(len(AllAgentBeliefVector[0]))

        for i,Bt in enumerate(AllAgentBeliefVector):
            Zeros=Zeros+Bt

        Zeros=Zeros/len(AllAgentBeliefVector)


        BeliefVector=self.GetBelief(Zeros)
        self.set_belief(BeliefVector)

        self.RemeberConsensus(Zeros)

        self.ConensusHistory.append(Zeros)
        self.StatesHistory.append(AllAgentsStateVectors)
        self.BeliefHistory.append(AllAgentBeliefVector)


        return Zeros



    def Remeber(self,OldAgentState,Action,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus,Obs,TrueEnvState,ActionType):
        self.LocalHistory.append([OldAgentState,Action,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus,None,Obs,TrueEnvState,ActionType])


    def RemeberConsensus(self,Consensus):

        #BeliefCopy = copy.deepcopy(Consensus)
        #MaxBelief = BeliefCopy[-1]
        #SecondMax = BeliefCopy[-2]

        # Bt = copy.deepcopy(list(self.cur_belief.histogram.values()))
        # Bt.sort()
        # coef = Bt[-1] / Bt[-2]
        # self.RW = 0
        # if coef > self.K :
        #     print('--- '+str(coef)+'---')
        #     self.RW = 100
        #     self.InMission=False

        #self.LocalHistory[-1][10] = Consensus
        self.Consensus=Consensus


    def Replay(self,batch_size,epochs=512):

        x_batch, y_batch = [], []
        #my_minibatch,his_minibatch=zip(*random.sample(list(zip(self.MyHistory,self.HisHistory)),min(len(self.MyHistory), batch_size)))
        my_minibatch = random.sample(self.LocalHistory, min(len(self.LocalHistory), batch_size))
        #his_minibatch = random.sample(self.HisHistory, min(len(self.HisHistory), batch_size))

        for OldAgentState,ActionName,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus,NewConsensus,Obs,TrueEnvState,ActionType in my_minibatch:

            #print('OldAgentState',OldAgentState)

            process_obs=self.preprocess_state(OldAgentState,OldBelief)
            #print('process_obs:',np.array(process_obs))
            #print('Action:', ActionName)
            Q_target = self.model.predict(process_obs)
            #print('Q_target:',Q_target)
            ActionID=self.AGENT_ACTIONS.index(ActionName)
            #print('ActionID:', ActionID)
            #print('cur RW:', RW)
            #RewardMat, IsUndefined, MaxCoef = self.GetBeliefMatrix(OldBelief)
            #print('---REW MATRIX ---')
            #for ro in RewardMat:
            #    print(ro)
            #print('-------------')
            if Done:
                Q_target[0][ActionID] = RW


            else:
                process_obs_next = self.preprocess_state(NewAgentState,NewBelief)
                #print('process_obs_next:', np.array(process_obs_next))
                Q_target_new=self.model.predict(process_obs_next)
                #print('Q_target_new:', Q_target_new)
                #print('Q target valuee:',(1-self.r_lr)*(Q_target[0][ActionID])+ self.r_lr*(RW + self.gamma * np.max(Q_target_new[0])))
                Q_target[0][ActionID] = (1-self.r_lr)*(Q_target[0][ActionID])+ self.r_lr*(RW + self.gamma * np.max(Q_target_new[0]))

            x_batch.append(process_obs[0])
            y_batch.append(Q_target[0])
        print('------ R E P L A Y ------ ' + (self.AgentName))
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),epochs = epochs , verbose=2)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def ChangeAgentState(self,Action):
        if(Action.name=='end'):
            self.cur_agent_state = 'h'
        else:
            self.cur_agent_state=[x for x in self.AGENT_STATES if x[-1] == Action.name[-1]][0]
        self.cur_agent_state_index=self.AGENT_STATES.index(self.cur_agent_state)
        self.cur_agent_state_vector=np.zeros(len(self.AGENT_STATES))
        self.cur_agent_state_vector[self.cur_agent_state_index]=1

    def Cur_State_Coef(self):
        return self.GetCoef(list(self.cur_belief.histogram.values()))

    def GetCoef(self,Bt_vect):
        Bt_New=copy.deepcopy(Bt_vect)
        Bt_New.sort()
        return  Bt_New[-1]/Bt_New[-2]

    def LoadPolicy(self,ModelLink):
        self.model=keras.models.load_model(ModelLink)

    def GetAction(self):

        AgentShoudEnd=False
        NumOfSubEnvs = len(self.InitialAgentsDistributionMap)
        for t in range(NumOfSubEnvs):
            InitNum = len(self.InitialAgentsDistributionMap[t])
            CurNum = len(self.AgentsDistributionMap[t])
            #if CurNum==0:
            if(CurNum<(InitNum/2)):
                AgentShoudEnd=True
        Type='Model'
        if AgentShoudEnd:
            ActionID = self.AGENT_ACTIONS.index('end')
            Type = 'AgentsAreLeaving'
        else:
            #AgAction = self.GetBestActionAccordintToReward(self.cur_agent_state, self.cur_belief)
            #ActionID =self.AGENT_ACTIONS.index(AgAction.name)
            ActionID = np.argmax(self.model.predict(self.preprocess_state(self.cur_agent_state,
                                                                          list(
                                                                              self.cur_belief.histogram.values()))))
        return  (ActionID,Type)


    def TakeAction(self,Environment,Action):
        if Action.name!='end':
            Indx=self.AGENT_ACTIONS.index(Action.name)
            self.MyScouting[Indx]=self.MyScouting[Indx]+1

        TOTAL_RW=self.GetAgentTotalReward(self.cur_agent_state,self.cur_belief,Action)
        #RW=self.AgentReward(self.cur_agent_state,Action)
        #RW2=self.BeliefReward(self.cur_belief,Action)

        #if RW2!=0 and RW>0:
        #    RW=0
        #elif RW<0:
        #    RW2=0
        #elif RW2==0 and RW==0:
        #    RW=-19
        #TOTAL_RW=RW2+RW
        #if RW2>0:
        #    TOTAL_RW=RW2
        #else:
        #    TOTAL_RW=RW+RW2

        OldConsensus=self.Consensus
        OldState=self.cur_agent_state
        self.ChangeAgentState(Action)
        NewState=self.cur_agent_state
        self.BeliefHistory.append(self.cur_belief)
        self.ActionHistory.append(self.Action)
        self.OldOldBelief=self.OldBelief
        self.OldBelief=self.cur_belief

        self.Old_Action=self.Action
        self.Action=Action

        if Action.name != 'end':
            real_observation_agent = self.observation_model.sample(Environment.state, Action)

            agent_new_belief = pomdp_py.update_histogram_belief(self.cur_belief,
                                                                  Action, real_observation_agent,
                                                                  self.observation_model,
                                                                  self.transition_model)

            self.update_history(Action, real_observation_agent)
            self.set_belief(agent_new_belief)
            Done=False
        elif Action.name == 'end':
            real_observation_agent=None
            agent_new_belief=self.cur_belief

        #print(self.AgentName + ' TakedAction:' + Action.name)
        #print(self.AgentName + ' Observation:' + real_observation_agent.name)
        #print(self.AgentName + ' Cur_Belief:' + str(self.cur_belief.histogram))


        #scoutingCopy=copy.deepcopy(self.MyScouting)
        #scoutingCopy.sort()
        if Action.name=='end':
            Done=True
            self.InMission=False
        #elif scoutingCopy.max()-scoutingCopy.min()>2:
        #    Done=True
        #    self.InMission = False


        #RW=self.AgentReward(OldState,Action)
        #RW2=self.BeliefReward(agent_new_belief,Action)

        #if RW2>0:
        #    TOTAL_RW=RW2
        #else:
        #    TOTAL_RW=RW+RW2
        #TOTAL_RW=RW+RW2

        oldCoef=self.GetCoef(list(self.OldBelief.histogram.values()))
        newCoef=self.GetCoef(list(self.cur_belief.histogram.values()))

        return OldState,NewState,Action,Done,TOTAL_RW,oldCoef,newCoef,list(self.OldBelief.histogram.values()),list(self.cur_belief.histogram.values()),OldConsensus,real_observation_agent



def SaveRewardProgres(FilePath,RewardProgressList):
    with open(FilePath, 'wb') as file:
        #file = open(FilePath, 'wb')
        pickle.dump(RewardProgressList, file)

def LoadRewardProgres(FilePath):
    with open(FilePath, 'rb') as file:
        return pickle.load(file)


class AgentState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, AgentState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "AgentState(%s)" % self.name


class AgentAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, AgentAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "AgentAction(%s)" % self.name

class AgentObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, AgentObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "AgentObservation(%s)" % self.name

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "end":
            return 0.25

        if action.name[-1]==next_state.name[-1]:
            if observation.name in {'video'}:
                return 0.45
                #return 0.5-1e-9
            if observation.name in {'thermal'}:
                return 0.45
                #return 0.5-1e-9
            elif observation.name in {'none'}:
                return 0.09
                #return 1e-9
            else:
                return 0.01
                #return 1e-9
        else:
            if observation.name in {'video'}:
                return 0.045
                #return 1e-9
            if observation.name in {'thermal'}:
                return 0.045
                #return 1e-9
            elif observation.name in {'none'}:
                return 0.9
                #return 1-1e-9-1e-9-1e-9
            else:
                return 0.01
                #return 1e-9


#Environment_State (Real)   , Action
    def sample(self, next_state, action):

        #sampleProbs=[0.25,0.25,0.25,0.25]

        if action.name == "end":
            sampleProbs = [0.25,0.25,0.25,0.25]
        elif action.name[-1]==next_state.name[-1]:
            sampleProbs = [0.45 , 0.45 , 0.09 , 0.01 ]
            #sampleProbs = [0.5-1e-9, 0.5-1e-9, 1e-9, 1e-9]
        else:
            sampleProbs = [0.045, 0.045, 0.9  , 0.01 ]
            #sampleProbs = [1e-9, 1e-9, 1.0 - 1e-9 - 1e-9 - 1e-9, 1e-9]

        ObservationList=self.get_all_observations()

        return random.choices(ObservationList, sampleProbs)[0];

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [AgentObservation(s)
                for s in ["video", "thermal","none","unknown"]]


# Observation model
class ObservationModelOld(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "end":
            return 0.25

        if action.name[-1]==next_state.name[-1]:
            if observation.name in {'video'}:
                return 0.4
            if observation.name in {'thermal'}:
                return 0.4
            elif observation.name in {'none'}:
                return 0.19
            else:
                return 0.01
        else:
            if observation.name in {'video'}:
                return 0.04
            if observation.name in {'thermal'}:
                return 0.04
            elif observation.name in {'none'}:
                return 0.8
            else:
                return 0.12


#Environment_State (Real)   , Action
    def sample(self, next_state, action):

        #sampleProbs=[0.25,0.25,0.25,0.25]

        if action.name == "end":
            sampleProbs = [0.25,0.25,0.25,0.25]
        elif action.name[-1]==next_state.name[-1]:
            sampleProbs = [0.4 , 0.4 , 0.18 , 0.02 ]
        else:
            sampleProbs = [0.04, 0.04, 0.8  , 0.12 ]

        ObservationList=self.get_all_observations()

        return random.choices(ObservationList, sampleProbs)[0];

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [AgentObservation(s)
                for s in ["video", "thermal","none","unknown"]]

class TransitionModel(pomdp_py.TransitionModel):

    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        #if action.name.startswith("end"):
        #    return 0.5
        #else:
        if state.name=='u' and next_state.name=='u':
            return 0.7
        elif state.name=='u':
            return 0.03
        elif next_state.name == state.name:
            return 1.0 - 1e-9
        else:
            return 1e-9


    def sample(self, state, action):
        if action.name.startswith("end"):
            return random.choice(self.get_all_states())
        else:
            return AgentState(state.name)


    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [AgentState(s) for s in {'u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10'}]

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "end":
            return 10
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

    def join_reward_func(self,state,Actions):
        return 100

class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = {AgentAction(s)
              for s in ["scout1", "scout2", "scout3", "scout4", "scout5", "scout6", "scout7", "scout8", "scout9", "scout10","end"]}

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]


    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)


    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    print(f'Hi Zavrsio')  # Press ⌘F8 to toggle the breakpoint.


def AverageConsensusFunction(AllAgentsStateVectors,AllAgentBeliefVector):
    print('----AverageConsensusFunction-----')
    print(AllAgentBeliefVector)
    ResultBelief = np.zeros(len(AllAgentsStateVectors[0]))
    for Bt in AllAgentBeliefVector:
        ResultBelief = ResultBelief + Bt

    ResultBelief = ResultBelief / len(AllAgentBeliefVector)
    print(ResultBelief)
    print('----AverageConsensusFunction-----')
    return ResultBelief

def ConsensusFunction(AllAgentsStateVectors,AllAgentBeliefVector):
    print('CONSENSUS -----'+'STATES:'+str(len(AllAgentsStateVectors))+' BELIEF:' + str(len(AllAgentBeliefVector)))


    for p in range(len(AllAgentsStateVectors)):
        print('       ----- STATE ' + str(list(AllAgentsStateVectors[p])))
        print('       ----- BELIE ' + str(list(AllAgentBeliefVector[p])))

    AllStates=np.zeros(len(AllAgentsStateVectors[0]))
    for States in AllAgentsStateVectors:
        AllStates=AllStates+States

    NumOfAgents=len(AllAgentsStateVectors)
    ResultBelief=np.zeros(len(AllAgentsStateVectors[0]))
    #print('Result Belief')
    #print(ResultBelief)
    TakeInAccountCoef=0.8
    NotTakeInAccountCoef=0.2

    if NumOfAgents>1:

        for i in range(len(AllStates)):# i in (0,1,2,3,4....9,10)
            #print('i=',i)
            if AllStates[i]>0:
                #ResultBelief[i]=
                for j in range(NumOfAgents): # j in (0,1,2)
                    #print('j=',j)
                    if AllAgentsStateVectors[j][i]==1:
                        coefA=TakeInAccountCoef/AllStates[i]
                        ResultBelief[i]=ResultBelief[i]+coefA*AllAgentBeliefVector[j][i]
                    elif AllAgentsStateVectors[j][i]==0:
                        coefB = NotTakeInAccountCoef / (NumOfAgents-AllStates[i])
                        ResultBelief[i] = ResultBelief[i] + coefB * AllAgentBeliefVector[j][i]
            elif AllStates[i]==0:
                for j in range(NumOfAgents):
                    coefC=1/NumOfAgents
                    ResultBelief[i]=ResultBelief[i]+coefC*AllAgentBeliefVector[j][i]

    if NumOfAgents==1:
        ResultBelief=AllAgentBeliefVector[0]

    ResultBelief=ResultBelief/sum(ResultBelief)


    print('       ----- RESULT ' + str(list(ResultBelief)))
    return ResultBelief



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ModelFolderName ='POMDP20X6/FireModel_NewRW_NewCons_ver10'
    ExpirenceFileName='newCon_ver10_ep4'
    AgentsExpirienceFolder='POMDP20x6/Experience'
    print_hi('Tarik Selimovic')
    root_model = ET.parse('POMDP_Models/agent1_10r.pomdpx').getroot()
    print(root_model)
    description, discount, states, actions, observations, state_names, observability = parser.get_general_info(root_model)
    #belief = parser.get_initial_belief(root_model)
    #transition_probs = parser.get_matrix('StateTransitionFunction', root_model, observability)
    #observation_probs = parser.get_matrix('ObsFunction', root_model, observability)
    #Exmplae
    #transition_probs[0][actions[0][5]]

    obs_noise=0.15
    belief=0.05
    init_belief = pomdp_py.Histogram({
        AgentState("u"): 0.5,
        AgentState("f1"): belief,
        AgentState("f2"): belief,
        AgentState("f3"): belief,
        AgentState("f4"): belief,
        AgentState("f5"): belief,
        AgentState("f6"): belief,
        AgentState("f7"): belief,
        AgentState("f8"): belief,
        AgentState("f9"): belief,
        AgentState("f10"): belief,
        })
    #EnvironmentState=AgentState("f9")
    #observation_model=ObservationModel(obs_noise)
    #transition_model=TransitionModel()
    #Action=AgentAction("scout9")
    #real_observation_agent = observation_model.sample(EnvironmentState, Action)


    init_true_state = random.choice([AgentState(s) for s in {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10'}])


    env = CustomEnvironment(init_true_state,
                               TransitionModel(),
                               RewardModel())

    #Epsilon=0.5
    Epsilon = 1
    Models=['','','']
    #Models=['FireModel_NewGroupWorking_NewRW_ver2/agent1_NN_model_verFirstTrain_Epizode_6_1500',
    #         'FireModel_NewGroupWorking_NewRW_ver2/agent2_NN_model_verFirstTrain_Epizode_6_1500',
    #         'FireModel_NewGroupWorking_NewRW_ver2/agent3_NN_model_verFirstTrain_Epizode_6_1500']

    #Models = ['FireModel_NewRW_NewCons_ver10/agent1_NN_model_verFirstTrain_1_Final',
    #          'FireModel_NewRW_NewCons_ver10/agent2_NN_model_verFirstTrain_1_Final',
    #          'FireModel_NewRW_NewCons_ver10/agent3_NN_model_verFirstTrain_1_Final']

    agent_1 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent1',Models[0],Epsilon)

    agent_2 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent2',Models[1],Epsilon)

    agent_3 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent3',Models[2],Epsilon)

    MaxIteration=150
    Agents=[agent_1, agent_2, agent_3]
    #[agent.LoadLocalHistory('AgentsExpirience/LH_' + agent.AgentName+'_'+'newCon_ver10_ep1') for agent in Agents]
    #Agents=[agent_1]
    NumberOfEpisodes=5
    NumberOfEpisodesEnd=15000
    running_reward=[0,0,0]
    Rewards=[]
    for Epizode in range(NumberOfEpisodes,NumberOfEpisodesEnd):
        env.Reset()
        [agent.Reset() for agent in Agents]

        DDD = False
        print("==== Episode %d ==== " % (Epizode + 1))
        i = 0
        print("True state:", env.state)
        episode_reward=[0,0,0]

        while (not DDD):
            print("==== Ep / Iteration "+str((Epizode + 1))+"/ %d ==== " % (i + 1))
            IterationBeliefVectors = []
            IterationStateVectors = []
            for agent in Agents:
                if agent.InMission:
                    agent.StateInfo()
                    action_sample = agent.choose_action(agent.cur_agent_state,agent.cur_belief)
                    OldState, NewState, Action, Done, RW,oldCoef,newCoef, prevBelief, curBelief, OldConensus,real_observation_agent = agent.TakeAction(env, action_sample)
                    episode_reward[int(agent.AgentName[-1])-1]+=RW
                    if real_observation_agent is None:
                        obs='NULL'
                    else:
                        obs=real_observation_agent.name
                    print('                 ------AFTER ACTION -----')
                    print('                 TOTAL RW:' + str(RW))
                    print('                 Observation:' + obs)
                    print('                 True state:', env.state)
                    print('                 cur_belief:' + str(curBelief) )
                    print('                 epsilon:' + str(agent.epsilon) )
                    if obs!='NULL':
                        IterationBeliefVectors.append(list(agent.cur_belief.histogram.values()))
                        IterationStateVectors.append(agent.cur_agent_state_vector)

                    agent.Remeber(OldState, Action.name, NewState, RW, Done,oldCoef,newCoef, prevBelief, curBelief, OldConensus,obs,env.state.name,'Model')

            if len(IterationStateVectors)>0:
                ConBelief=ConsensusFunction(np.array(IterationStateVectors),np.array(IterationBeliefVectors))

            [agent.SetConsensusBelief(ConBelief) for agent in Agents if agent.InMission]
            #[agent.ShareData2(np.array(IterationStateVectors),np.array(IterationBeliefVectors)) for agent in Agents if agent.InMission]
            i=i+1

            if (Agents[0].InMission==False and Agents[1].InMission==False and Agents[2].InMission==False) or MaxIteration==i:#Agents[0].InMission==False or MaxIteration==i: #
                DDD=True

        print(' --- EPIZODE REWARDS {0}  --- '.format(Epizode + 1))
        for inx,REW in enumerate(episode_reward):
            print("Agent{0}  RW:{1}".format(inx + 1, REW))
            running_reward[inx]=0.05*REW+(1-0.05)*running_reward[inx]


        if Epizode%10==0:
            print(' --- RUNING REWARDS --- ')
            for inx,ll in enumerate(running_reward):
                print("Agent{0}  RW:{1}".format(inx+1,ll))

            Rewards.append(copy.deepcopy(running_reward))
            SaveRewardProgres(AgentsExpirienceFolder+'/Rewards_10_' + ExpirenceFileName,Rewards)

            print(' --- TRAINING --- ENV_STATE:'+env.state.name)
            [agent.Replay(batch_size=256,epochs=512) for agent in Agents]

        if Epizode%300==0:
            SaveRewardProgres(AgentsExpirienceFolder+'/Rewards_300_' + ExpirenceFileName, Rewards)
            [agent.SaveLocalHistory(AgentsExpirienceFolder+'/LH_'+agent.AgentName+'_'+ExpirenceFileName) for agent in Agents]
            agent_1.model.save(ModelFolderName+'/agent1_NN_model_verFirstTrain_Epizode_4_'+str(Epizode))
            agent_2.model.save(ModelFolderName+'/agent2_NN_model_verFirstTrain_Epizode_4_'+str(Epizode))
            agent_3.model.save(ModelFolderName+'/agent3_NN_model_verFirstTrain_Epizode_4_'+str(Epizode))





    agent_1.model.save(ModelFolderName+'/agent1_NN_model_verFirstTrain_4_Final')
    agent_2.model.save(ModelFolderName+'/agent2_NN_model_verFirstTrain_4_Final')
    agent_3.model.save(ModelFolderName+'/agent3_NN_model_verFirstTrain_4_Final')
