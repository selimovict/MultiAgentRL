from pomdpx_parser import pomdpx_parser as parser
import xml.etree.ElementTree as ET
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.callbacks import CSVLogger

from FireModel_ver2 import AgentAction
from FireModel_ver2 import AgentState
from FireModel_ver2 import CustomAgent
from FireModel_ver2 import CustomEnvironment
from FireModel_ver2 import GeneralCustomEnvironment
from FireModel_ver2 import PolicyModel
from FireModel_ver2 import TransitionModel
from FireModel_ver2 import ObservationModel
from FireModel_ver2 import RewardModel

from FireModel_ver2 import ConsensusFunction
from FireModel_ver2 import AverageConsensusFunction
import pickle
import matplotlib.pyplot as plt

F_STATES = ['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']


def LocalHistorySummary(LocalHistory):
    F_STATES = ['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

    # pd.DataFrame(LocalHistorySummary(SimulationLog[0]['agent3']),columns=['location_state.data','coef.data','fire_state.data','belief.data','action.data']).to_string()

    return pd.DataFrame([[X[0], X[5], F_STATES[np.argmax(X[7])], X[1],X[13],X[3],X[11], X[7]] for X in LocalHistory],
                        columns=['location_state.data', 'coef.data', 'fire_state.data','action.data','action.type','RW.data','obs.data', 'belief.data'])

ExpirenceFileName='newCon_ver10_ep1'
FilePath='AgentsExpirience/Rewards_10_' + ExpirenceFileName
def LoadRewardProgres(FilePath):
    with open(FilePath, 'rb') as file:
        return pickle.load(file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Tarik Selimovic')
    obs_noise = 0.15
    belief = 0.05
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
    init_true_state = random.choice(
        [AgentState(s) for s in {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'}])
    env = CustomEnvironment(init_true_state,
                            TransitionModel(),
                            RewardModel())





    # Models=['FireModel_NewGroupWorking/agent1_NN_model_verFirstTrain_Epizode_5_8700','FireModel_NewGroupWorking/agent2_NN_model_verFirstTrain_Epizode_5_8700','FireModel_NewGroupWorking/agent3_NN_model_verFirstTrain_Epizode_5_8700']
    # Models=['FireModel_NewGroupWorking/agent1_NN_model_verFirstTrain_4_Final','FireModel_NewGroupWorking/agent2_NN_model_verFirstTrain_4_Final','FireModel_NewGroupWorking/agent3_NN_model_verFirstTrain_4_Final']
    # Models=['FireModel_NewGroupWorking/agent1_NN_model_verFirstTrain_3_Final','FireModel_NewGroupWorking/agent2_NN_model_verFirstTrain_3_Final','FireModel_NewGroupWorking/agent3_NN_model_verFirstTrain_3_Final']

    # Models=['FireModel_NewGroupWorking/agent1_NN_model_verFirstTrain_5_Final','FireModel_NewGroupWorking/agent2_NN_model_verFirstTrain_5_Final','FireModel_NewGroupWorking/agent3_NN_model_verFirstTrain_3_Final']

    # Models = ['FireModel_NewGroupWorking_Ver2/agent1_NN_model_verFirstTrain_Epizode_1_5700',
    #          'FireModel_NewGroupWorking_Ver2/agent2_NN_model_verFirstTrain_Epizode_1_5700',
    #          'FireModel_NewGroupWorking_Ver2/agent3_NN_model_verFirstTrain_Epizode_1_5700']

    Models = ['FireModel_NewRW/agent1_NN_model_verFirstTrain_Epizode_1_9000',
              'FireModel_NewRW/agent2_NN_model_verFirstTrain_Epizode_1_9000',
              'FireModel_NewRW/agent3_NN_model_verFirstTrain_Epizode_1_9000']

    Models_vr2 = ['FireModel_NewRW_ver2/agent1_NN_model_verFirstTrain_Epizode_1_3900',
                  'FireModel_NewRW_ver2/agent2_NN_model_verFirstTrain_Epizode_1_3900',
                  'FireModel_NewRW_ver2/agent3_NN_model_verFirstTrain_Epizode_1_3900']

    Models_newCon = ['FireModel_NewRW_NewCons/agent1_NN_model_verFirstTrain_Epizode_1_7200',
                     'FireModel_NewRW_NewCons/agent2_NN_model_verFirstTrain_Epizode_1_7200',
                     'FireModel_NewRW_NewCons/agent3_NN_model_verFirstTrain_Epizode_1_7200']

    Models_newCon_ver2 = ['FireModel_NewRW_NewCons_ver2/agent1_NN_model_verFirstTrain_Epizode_1_5400',
                          'FireModel_NewRW_NewCons_ver2/agent2_NN_model_verFirstTrain_Epizode_1_5400',
                          'FireModel_NewRW_NewCons_ver2/agent3_NN_model_verFirstTrain_Epizode_1_5400']

    Models_newCon_ver4 = ['FireModel_NewRW_NewCons_ver6/agent1_NN_model_verFirstTrain_Epizode_1_5700',
                          'FireModel_NewRW_NewCons_ver6/agent2_NN_model_verFirstTrain_Epizode_1_5700',
                          'FireModel_NewRW_NewCons_ver6/agent3_NN_model_verFirstTrain_Epizode_1_5700']

    Models_newCon_ver6 = ['FireModel_NewRW_NewCons_ver10/agent1_NN_model_verFirstTrain_Epizode_4_11700',
                          'FireModel_NewRW_NewCons_ver10/agent2_NN_model_verFirstTrain_Epizode_4_11700',
                          'FireModel_NewRW_NewCons_ver10/agent3_NN_model_verFirstTrain_Epizode_4_11700']

    Models_newCon_ver10 = ['FireModel_NewRW_NewCons_ver10/agent1_NN_model_verFirstTrain_3_Final',
                           'FireModel_NewRW_NewCons_ver10/agent2_NN_model_verFirstTrain_3_Final',
                           'FireModel_NewRW_NewCons_ver10/agent3_NN_model_verFirstTrain_3_Final']

# 3_5100 Dobar model
    agent_1 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent1', Models_newCon_ver6[0])

    agent_2 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent2', Models_newCon_ver6[1])

    agent_3 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent3', Models_newCon_ver6[2])

    agent_4 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent4', Models_newCon_ver6[0])

    agent_5 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent5', Models_newCon_ver6[1])

    agent_6 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent6', Models_newCon_ver6[2])

    agent_7 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent7', Models_newCon_ver6[0])

    agent_8 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent8', Models_newCon_ver6[1])

    agent_9 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent9', Models_newCon_ver6[2])

    agent_10 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent10', Models_newCon_ver6[0])

    agent_11 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent11', Models_newCon_ver6[1])

    agent_12 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent12', Models_newCon_ver6[2])



    #Agents = [agent_1, agent_2, agent_3]
    NumberOfSimulation = 20
    #AgentMaps = [[agent_1, agent_2, agent_3]]
    AgentMaps= [[agent_1, agent_2, agent_3],[agent_4, agent_5, agent_6],[agent_7, agent_8, agent_9],[agent_10, agent_11, agent_12]]
    #AgentMaps = [[agent_1, agent_2, agent_3], [agent_4, agent_5, agent_6],[agent_7, agent_8, agent_9]]

    NumOfSubEnvs=len(AgentMaps)
    States=[AgentState('u')]*NumOfSubEnvs
    my_state = random.choice(
    [AgentState(s) for s in {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'}]
    )
    States[random.randrange(NumOfSubEnvs)]=my_state

    general_env = GeneralCustomEnvironment(States,
                                           TransitionModel(),
                                           RewardModel())

    def GetAgentIndex(AgentName):
        for i in range(len(AgentMaps)):
            for j in range(len(AgentMaps[i])):
                if (AgentMaps[i][j].AgentName==AgentName):
                    return (i,j)


    MaxIteration = 30

    NumberOfIteration = [{} for i in range(NumberOfSimulation)]  # np.zeros(NumberOfSimulation)
    SimulationLog = []
    for simulation in range(NumberOfSimulation):
        Iteration = 0
        AgentStateMap = [[AgentState(F_STATES[np.argmax(list(agent.cur_belief.histogram.values()))]) for agent in agents] for agents in AgentMaps]
        AgentStateMapCoef = [[agent.Cur_State_Coef for agent in agents] for agents in AgentMaps]
        AgentDistributionMap = [[agent.AgentName for agent in agents] for agents in AgentMaps]
        general_env.Reset()
        print('Sim EnvState:' + str(general_env.state))
        [[agent.Reset(AgentDistributionMap) for agent in agents] for agents in AgentMaps]
        [[agent.ForgetHistory() for agent in agents] for agents in AgentMaps]

        AgentInMission = sum([ len(Agents) for Agents in AgentMaps])# len(Agents)
        DDD = False

        ItersPerAgent = {}
        for in_Agents in AgentMaps:
            for ag in in_Agents:
                ItersPerAgent[ag.AgentName] = 0
        AgemtSimLog = {}
        AgemtSimLog['EnvState'] = str(general_env.state)
        while DDD != True:
            print('--------------ITERATION:' + str(Iteration) + '---------------')
            #for Agents in AgentMaps:
            for EnvID,Agents in enumerate(AgentMaps):

                IterationBeliefVectors = []
                IterationStateVectors = []
                for Agent in Agents:

                    if Agent.InMission:
                        print('Agent:' + Agent.AgentName)
                        print('State:' + Agent.cur_agent_state)
                        print('Belief:' + str(Agent.cur_belief.histogram))
                        Agent.StateInfo()
                        (ActionID,ActionType)=Agent.GetAction()
                        print('ActionID:' + str(ActionID))
                        #ActionID = np.argmax(Agent.model.predict(Agent.preprocess_state(Agent.cur_agent_state,
                        #                                                                list(
                        #                                                                    Agent.cur_belief.histogram.values()))))

                        NextAction = Agent.AGENT_ACTIONS[ActionID]
                        print('Agent:' + Agent.AgentName + ';It:'+str(Iteration)+'  NextModelAction:' + NextAction)
                        OldState, NewState, Action, Done, RW, oldCoef, newCoef, prevBelief, curBelief, OldConensus,real_observation_agent = Agent.TakeAction( general_env.Envs[EnvID], AgentAction(NextAction))

                        if real_observation_agent is None:
                            obs='NULL'
                        else:
                            obs=real_observation_agent.name
                        print('Agent:' + Agent.AgentName + '  Observation:' + obs+ ' RW:'+str(RW))
                        print('Agent OldCoef:' + str(oldCoef))
                        print('Agent NewCoef:' + str(newCoef))
                        if obs!='NULL':
                            IterationBeliefVectors.append(list(Agent.cur_belief.histogram.values()))
                            IterationStateVectors.append(Agent.cur_agent_state_vector)
                        Agent.Remeber(OldState, Action.name, NewState, RW, Done, oldCoef, newCoef, prevBelief, curBelief,
                                      OldConensus,obs,general_env.Envs[EnvID].state.name,ActionType)
                        ItersPerAgent[Agent.AgentName] = ItersPerAgent[Agent.AgentName] + 1
                        if Done or ItersPerAgent[Agent.AgentName] == MaxIteration:

                            AgentInMission = AgentInMission - 1

                            AgemtSimLog[Agent.AgentName] = LocalHistorySummary(Agent.LocalHistory)
                            print(AgemtSimLog[Agent.AgentName].to_string())
                            AgemtSimLog[Agent.AgentName+"_LocalHistory"] = Agent.LocalHistory

                            AgemtSimLog[Agent.AgentName+"_State"] = F_STATES[np.argmax(list(Agent.cur_belief.histogram.values()))]
                            AgState=AgentState(F_STATES[np.argmax(list(Agent.cur_belief.histogram.values()))])

                            AgentDistributionMap = [[agent for agent in agents if agent!=Agent.AgentName] for agents in AgentDistributionMap]
                            A_Indx=GetAgentIndex(Agent.AgentName)
                            AgentStateMap[A_Indx[0]][A_Indx[1]]=F_STATES[np.argmax(list(Agent.cur_belief.histogram.values()))]
                            AgentStateMapCoef[A_Indx[0]][A_Indx[1]]=Agent.GetCoef(list(Agent.cur_belief.histogram.values()))

                            [[agent.SetAgentsDistributionMap(AgentDistributionMap) for agent in agents] for agents in
                             AgentMaps]

                            [[agent.SetAgentsStateMap(AgentStateMap) for agent in agents] for agents in
                             AgentMaps]
                            [[agent.SetAgentsStateMapCoef(AgentStateMapCoef) for agent in agents] for agents in
                             AgentMaps]



                            #AgentMaps_New = [[agent for agent in Agents if agent.AgentName != Agent.AgentName] for Agents in
                            #                 AgentMaps]

                if len(IterationStateVectors)>0:
                    ConBelief=ConsensusFunction(np.array(IterationStateVectors),np.array(IterationBeliefVectors))
                    #ConBelief = AverageConsensusFunction(np.array(IterationStateVectors), np.array(IterationBeliefVectors))
                    [agent.SetConsensusBelief(ConBelief) for agent in Agents if agent.InMission]

                #[agent.ShareData2(np.array(IterationStateVectors), np.array(IterationBeliefVectors)) for agent in Agents if agent.InMission]
            Iteration = Iteration + 1
            print('AgentInMission:' + str(AgentInMission))
            if AgentInMission == 0:  # or Iteration==MaxIteration:#Agents[0].InMission==False or MaxIteration==i: #
                DDD = True
                SimulationLog.append(AgemtSimLog)

        print('EnvState:' + str(general_env.state))
        [Agent.StateInfo() for Agent in Agents]
        NumberOfIteration[simulation]['Iterations'] = Iteration
        NumberOfIteration[simulation]['SimulationID'] = simulation
        NumberOfIteration[simulation]['TrueState'] = str(general_env.state)
        States = [F_STATES[np.argmax(list(Ag.cur_belief.histogram.values()))] for Ag in Agents]
        NumberOfIteration[simulation]['AgentStateMap']=AgentStateMap
        NumberOfIteration[simulation]['AgentStateMapCoef'] = AgentStateMapCoef
        for m,Env_St in enumerate(AgentStateMap):
            for n,_St in enumerate(Env_St):
                if(_St=='u'):
                    AgentStateMapCoef[m][n]=AgentStateMapCoef[m][n]/10

        NumberOfIteration[simulation]['AgentStateMapCoef_Scaled'] = AgentStateMapCoef

        SubEnvs=[max(coef) for coef in AgentStateMapCoef]
        env_index = SubEnvs.index(max(SubEnvs))
        state_index=AgentStateMapCoef[env_index].index(max(AgentStateMapCoef[env_index]))

        _St=AgentStateMap[env_index][state_index]

        DetectedState=[AgentState('u')]*NumOfSubEnvs
        DetectedState[env_index]=AgentState(_St)

        NumberOfIteration[simulation]['Success'] = (str(general_env.state)==str(DetectedState))
## BITNO ZAKOMENTARISANO ##
print('NumberOfIterations:' + str(NumberOfIteration))
Successed = [x for x in NumberOfIteration if x['Success'] == True]
Unsuccessed = [x for x in NumberOfIteration if x['Success'] == False]
IterPerSim = [x['Iterations'] for x in NumberOfIteration]
Mean = np.mean(IterPerSim)
Std = np.std(IterPerSim)
SuccesProcentage = len(Successed) / len(NumberOfIteration) * 100
UnsuccessedProcentage = len(Unsuccessed) / len(NumberOfIteration) * 100

print("Iterations MEAN: %.2f" % Mean)
print("Iterations STD: %.2f" % Std)

print("Simulation Success : %.2f %%" % SuccesProcentage)
print("Simulation Unsuccess :%.2f %%" % UnsuccessedProcentage)
#### KRAJ

def GetSimulationInfo(SimulationID):
    print('Env State-----------------')
    print(SimulationLog[SimulationID]['EnvState'])

    for i,Agents in enumerate(AgentMaps):
        print('EnvIndex:'+str(i))
        for Agent in Agents:
            print("Agent:" + Agent.AgentName)
            print(SimulationLog[SimulationID][Agent.AgentName].to_string())


    # print('Agent1--------------------')
    # print(SimulationLog[SimulationID]['agent1'].to_string())
    # print('Agent2--------------------')
    # print(SimulationLog[SimulationID]['agent2'].to_string())
    # print('Agent3--------------------')
    # print(SimulationLog[SimulationID]['agent3'].to_string())
    # print('Agent4--------------------')
    # print(SimulationLog[SimulationID]['agent4'].to_string())
    # print('Agent5--------------------')
    # print(SimulationLog[SimulationID]['agent5'].to_string())
    # print('Agent6--------------------')
    # print(SimulationLog[SimulationID]['agent6'].to_string())

def GetAgentIterationReward(SimulationID,AgentName, IterationID):
    Agents_Name=['agent1','agent2','agent3']
    AgentID=Agents_Name.index(AgentName)
    LH=SimulationLog[SimulationID][AgentName+'_LocalHistory']
    print('State', LH[IterationID][0])
    print('CurBt', LH[IterationID][7])
    print('Action', LH[IterationID][1])
    print('Reward', LH[IterationID][3])
    BeliefRewardMat,IsUndefined,MaxCoef=Agents[AgentID].GetBeliefMatrix(LH[IterationID][7])
    print('IsUndefined',IsUndefined)
    print('MaxCoef', MaxCoef)
    for row in BeliefRewardMat:
        print(row)

# plt.figure('Simulation')
# plt.title('Iteration per Simulation')
# plt.bar(['Simulation %d'%i for i,x in enumerate(IterPerSim)],IterPerSim)
# plt.xlabel('Simulation')
# plt.ylabel('Number of iteration')
# plt.xticks(rotation = 45)
# plt.show()




#env.Reset()
#Agents[0].Reset()
#Agents[1].Reset()

#action_sample0 = Agents[0].choose_action(Agents[0].cur_agent_state,Agents[0].cur_belief)
#OldState, NewState, Action, Done, RW,oldCoef,newCoef, prevBelief, curBelief, OldConensus,real_observation_agent = Agents[0].TakeAction(env, action_sample0)
#if real_observation_agent is None:
#    obs = 'NULL'
#else:
#    obs = real_observation_agent.name

#if obs != 'NULL':
#    IterationBeliefVectors.append(list(Agents[0].cur_belief.histogram.values()))
#    IterationStateVectors.append(Agents[0].cur_agent_state_vector)


#action_sample1 = Agents[1].choose_action(Agents[1].cur_agent_state,Agents[1].cur_belief)
#OldState, NewState, Action, Done, RW,oldCoef,newCoef, prevBelief, curBelief, OldConensus,real_observation_agent = Agents[1].TakeAction(env, action_sample1)
#if real_observation_agent is None:
#    obs = 'NULL'
#else:
#    obs = real_observation_agent.name

#if obs != 'NULL':
#    IterationBeliefVectors.append(list(Agents[1].cur_belief.histogram.values()))
#    IterationStateVectors.append(Agents[1].cur_agent_state_vector)
#
#Agents[0].ShareData2(np.array(IterationStateVectors),np.array(IterationBeliefVectors))
#Agents[1].ShareData2(np.array(IterationStateVectors),np.array(IterationBeliefVectors))

#Bt_copy0 = copy.deepcopy(list(Agents[0].cur_belief.histogram.values()))
#Bt_copy1 = copy.deepcopy(list(Agents[1].cur_belief.histogram.values()))

def GetFigureReward(Rewards):
    plt.figure(figsize=(9, 3))

    plt.subplot(311)
    plt.plot([R[0] for R in Rewards])
    plt.title('agent1 reward')
    plt.subplot(312)
    plt.plot([R[1] for R in Rewards])
    plt.title('agent2 reward')
    plt.subplot(313)
    plt.plot([R[2] for R in Rewards])
    plt.title('agent3 reward')
    plt.suptitle('Categorical Plotting')
    plt.show()


#Agents[0].LoadLocalHistory('AgentsExpirience/LH_'+Agents[0].AgentName+'_'+'newCon_ver8_ep1')
#Agents[0].LocalHistory=random.sample(Agents[0].LocalHistory, 500)
#Agents[0].Replay(500,1024)