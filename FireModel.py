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
        self.F_STATES=['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

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
        self.gamma = 0.3
        self.epsilon = Epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001
        self.alpha_decay = 0.0001
        self.batch_size = 64
        self.K=3 #NUMBER OF MAX Bt VALUE HAVE TO BE HEIGHER OF SECOND MAX
        self.InMission=False
        self.RW=0
        self.r_lr=0.8
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

        if Model!='':
            self.LoadPolicy(Model)
        else:
            self.model = Sequential()
            self.model.add(Dense(8, input_dim=self.NumOfStates+self.NumOfBeliefs, activation='relu',name="data_input"))
            self.model.add(Dense(16, activation='sigmoid'))
            self.model.add(Dense(32, activation='relu'))
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
        Bt = copy.deepcopy(list(self.cur_belief.histogram.values()))
        Bt.sort()
        coef=Bt[-1]/Bt[-2]
        StateIndx=list(self.cur_belief.histogram.values()).index(Bt[-1])
        State=self.F_STATES[StateIndx]
        print('                 Coef:'+str(coef)+'  State:'+str(State)+' Bt:'+str(Bt[-1]))

    def Reset(self):
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
            R=self.AgentReward(AgentState,AgentAction(RandomAction))
            Rb=self.BeliefReward(self.cur_belief,AgentAction(RandomAction))
            print(self.AgentName+' ----- RANDOM ACTION ----- STATE:'+AgentState+' -ACTION:'+RandomAction+'  -RW:'+str(R)+'  -RW_b:'+str(Rb))
            self.StateInfo()
            return  AgentAction(RandomAction)
        else:
            R = self.AgentReward(AgentState, AgentAction(ModelAction))
            Rb = self.BeliefReward(self.cur_belief, AgentAction(ModelAction))
            print(self.AgentName+' ----- MODEL  ACTION ----- STATE:'+AgentState+' -ACTION:'+ModelAction+'  -RW:'+str(R)+'  -RW_b:'+str(Rb))
            self.StateInfo()
            return  AgentAction(ModelAction)

    def preprocess_state(self,AgentState,AgemtBeliefs):
        cur_agent_state_index=self.AGENT_STATES.index(AgentState)
        cur_agent_state_vector=np.zeros(self.NumOfStates)
        cur_agent_state_vector[cur_agent_state_index]=1

        return np.reshape(cur_agent_state_vector.tolist()+AgemtBeliefs, [1, self.NumOfStates+self.NumOfBeliefs])

    def AgentReward(self,agent_state,AgentAction):

        agent_state_index=self.AGENT_STATES.index(agent_state)
        agen_action_index= self.AGENT_ACTIONS.index(AgentAction.name)
        R=copy.deepcopy(self.RM[0][agen_action_index][0][agent_state_index])
        #if(R>=0 and R<6):
        #    R=0
        #elif R==6:

        R=R*10
        if R==0:
            R=-5
        if AgentAction.name!='end' and agent_state[-1]==AgentAction.name[-1]:
            R=-1000
        return R

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

        ActionID =0
        if AgentAction.name!='end':
            ActionID=int(AgentAction.name[-1])

        if MaxIndex>0:
            if coef >  self.K :  #Sigurno znam gdje je pojava
                if AgentAction.name=='end':
                    RW =  700
                elif ActionID==MaxIndex:
                    RW = 200 *0.4
                elif ActionID==MaxIndex2:
                    RW = 100 *0.4
                elif ActionID==MaxIndex3:
                    RW = 50  *0.4
                elif ActionID==MaxIndex4:
                    RW = 20  *0.4

            elif coef > 0.8 * self.K :
                if AgentAction.name=='end':
                    RW = -50
                elif ActionID==MaxIndex:
                    RW = 200 *0.6
                elif ActionID==MaxIndex2:
                    RW = 100 *0.6
                elif ActionID==MaxIndex3:
                    RW = 50 *0.6
                elif ActionID==MaxIndex4:
                    RW = 20 *0.6

            elif coef > 0.6 * self.K :
                if AgentAction.name=='end':
                    RW = -100
                elif ActionID==MaxIndex:
                    RW = 200 *0.8
                elif ActionID==MaxIndex2:
                    RW = 160 *0.8
                elif ActionID==MaxIndex3:
                    RW = 120 *0.8
                elif ActionID==MaxIndex4:
                    RW = 90 *0.8

            elif coef > 0.4* self.K :
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
                #elif ActionID == MaxIndex2 and MaxIndex2 > 0:
                #    RW = 60


        if MaxIndex==0:
            if AgentAction.name=='end':
                RW =-1000

        return  RW


    def ShareData(self,AllAgentsStateVectors,AllAgentBeliefVector):
        Zeros=np.zeros(len(AllAgentBeliefVector[0]))

        for Bt in AllAgentBeliefVector:
            Zeros=Zeros+Bt

        Zeros=Zeros/len(AllAgentBeliefVector)

        BeliefVector=self.GetBelief(Zeros)
        self.set_belief(BeliefVector)

        self.RemeberConsensus(Zeros)

        self.ConensusHistory.append(Zeros)
        self.StatesHistory.append(AllAgentsStateVectors)
        self.BeliefHistory.append(AllAgentBeliefVector)


        return Zeros



    def Remeber(self,OldAgentState,Action,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus):
        self.LocalHistory.append([OldAgentState,Action,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus,None])


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

        for OldAgentState,Action,NewAgentState,RW,Done,oldCoef,newCoef,OldBelief,NewBelief,OldConsensus,NewConsensus in my_minibatch:


            process_obs=self.preprocess_state(OldAgentState,OldBelief)
            Q_target = self.model.predict(process_obs)
            ActionID=self.AGENT_ACTIONS.index(Action)
            if Done:
                Q_target[0][ActionID] = RW
            else:
                process_obs_next = self.preprocess_state(NewAgentState,NewBelief)
                Q_target_new=self.model.predict(process_obs_next)
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

    def TakeAction(self,Environment,Action):
        if Action.name!='end':
            Indx=self.AGENT_ACTIONS.index(Action.name)
            self.MyScouting[Indx]=self.MyScouting[Indx]+1

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

        real_observation_agent = self.observation_model.sample(Environment.state, Action)

        agent_new_belief = pomdp_py.update_histogram_belief(self.cur_belief,
                                                              Action, real_observation_agent,
                                                              self.observation_model,
                                                              self.transition_model)

        self.update_history(Action, real_observation_agent)
        self.set_belief(agent_new_belief)
        Done=False
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


        RW=self.AgentReward(OldState,Action)
        RW2=self.BeliefReward(agent_new_belief,Action)

        TOTAL_RW=RW+RW2

        oldCoef=self.GetCoef(list(self.OldBelief.histogram.values()))
        newCoef=self.GetCoef(list(self.cur_belief.histogram.values()))
        return OldState,NewState,Action,Done,TOTAL_RW,oldCoef,newCoef,list(self.OldBelief.histogram.values()),list(self.cur_belief.histogram.values()),OldConsensus

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
            if observation.name in {'thermal'}:
                return 0.45
            elif observation.name in {'none'}:
                return 0.09
            else:
                return 0.01
        else:
            if observation.name in {'video'}:
                return 0.045
            if observation.name in {'thermal'}:
                return 0.045
            elif observation.name in {'none'}:
                return 0.9
            else:
                return 0.01


#Environment_State (Real)   , Action
    def sample(self, next_state, action):

        #sampleProbs=[0.25,0.25,0.25,0.25]

        if action.name == "end":
            sampleProbs = [0.25,0.25,0.25,0.25]
        elif action.name[-1]==next_state.name[-1]:
            sampleProbs = [0.45 , 0.45 , 0.09 , 0.01 ]
        else:
            sampleProbs = [0.045, 0.045, 0.9  , 0.01 ]

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    EnvironmentState=AgentState("f9")
    observation_model=ObservationModel(obs_noise)
    transition_model=TransitionModel()
    Action=AgentAction("scout9")
    real_observation_agent = observation_model.sample(EnvironmentState, Action)


    init_true_state = random.choice([AgentState(s) for s in {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10'}])


    env = CustomEnvironment(init_true_state,
                               TransitionModel(),
                               RewardModel())

    #Epsilon=0.5
    Epsilon = 1
    Models=['','','']
    #Models=['FireModel_NewGroupWorking_NewRW/agent1_NN_model_verFirstTrain_Epizode_6_1500',
    #         'FireModel_NewGroupWorking_NewRW/agent2_NN_model_verFirstTrain_Epizode_6_1500',
    #         'FireModel_NewGroupWorking_NewRW/agent3_NN_model_verFirstTrain_Epizode_6_1500']

    #Models = ['FireModel_NewGroupWorking_Ver3/agent1_NN_model_verFirstTrain_Epizode_1_600',
    #          'FireModel_NewGroupWorking_Ver3/agent2_NN_model_verFirstTrain_Epizode_1_600',
    #          'FireModel_NewGroupWorking_Ver3/agent3_NN_model_verFirstTrain_Epizode_1_600']

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

    MaxIteration=250
    Agents=[agent_1, agent_2, agent_3]
    #[agent.LoadLocalHistory('AgentsExpirience/LH_' + agent.AgentName+'_ver5') for agent in Agents]
    #Agents=[agent_1]
    NumberOfEpisodes=0
    NumberOfEpisodesEnd=10000
    for Epizode in range(NumberOfEpisodes,NumberOfEpisodesEnd):
        env.Reset()
        [agent.Reset() for agent in Agents]

        DDD = False
        print("==== Episode %d ==== " % (Epizode + 1))
        i = 0
        print("True state:", env.state)


        while (not DDD):
            print("==== Ep / Iteration "+str((Epizode + 1))+"/ %d ==== " % (i + 1))
            IterationBeliefVectors = []
            IterationStateVectors = []
            for agent in Agents:
                if agent.InMission:
                    action_sample = agent.choose_action(agent.cur_agent_state,agent.cur_belief)
                    OldState, NewState, Action, Done, RW,oldCoef,newCoef, prevBelief, curBelief, OldConensus = agent.TakeAction(env, action_sample)
                    IterationBeliefVectors.append(list(agent.cur_belief.histogram.values()))
                    IterationStateVectors.append(agent.cur_agent_state_vector)
                    agent.Remeber(OldState, Action.name, NewState, RW, Done,oldCoef,newCoef, prevBelief, curBelief, OldConensus)

            [agent.ShareData(np.array(IterationStateVectors),np.array(IterationBeliefVectors)) for agent in Agents]
            i=i+1

            if (Agents[0].InMission==False and Agents[1].InMission==False and Agents[2].InMission==False) or MaxIteration==i:#Agents[0].InMission==False or MaxIteration==i: #
                DDD=True

        if Epizode%10==0:
            print(' --- TRAINING --- ENV_STATE:'+env.state.name)
            [agent.Replay(batch_size=512,epochs=256) for agent in Agents]

        if Epizode%300==0:
            [agent.SaveLocalHistory('AgentsExpirience_LongPOMDP/LH_'+agent.AgentName+'_ver5') for agent in Agents]
            agent_1.model.save('FireModel_NewRW_LongPOMDP/agent1_NN_model_verFirstTrain_Epizode_1_'+str(Epizode))
            agent_2.model.save('FireModel_NewRW_LongPOMDP/agent2_NN_model_verFirstTrain_Epizode_1_'+str(Epizode))
            agent_3.model.save('FireModel_NewRW_LongPOMDP/agent3_NN_model_verFirstTrain_Epizode_1_'+str(Epizode))





    agent_1.model.save('FireModel_NewRW_LongPOMDP/agent1_NN_model_verFirstTrain_1_Final')
    agent_2.model.save('FireModel_NewRW_LongPOMDP/agent2_NN_model_verFirstTrain_1_Final')
    agent_3.model.save('FireModel_NewRW_LongPOMDP/agent3_NN_model_verFirstTrain_1_Final')
