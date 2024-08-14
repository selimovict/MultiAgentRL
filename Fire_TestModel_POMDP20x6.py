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

from FireModel_ver2_POMDP20x6 import AgentAction
from FireModel_ver2_POMDP20x6 import AgentState
from FireModel_ver2_POMDP20x6 import CustomAgent
from FireModel_ver2_POMDP20x6 import CustomEnvironment
from FireModel_ver2_POMDP20x6 import PolicyModel
from FireModel_ver2_POMDP20x6 import TransitionModel
from FireModel_ver2_POMDP20x6 import ObservationModel
from FireModel_ver2_POMDP20x6 import RewardModel

from FireModel_ver2_POMDP20x6 import ConsensusFunction
from FireModel_ver2_POMDP20x6 import AverageConsensusFunction
import pickle
import matplotlib.pyplot as plt

F_STATES = ['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10','f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19','f20']


def LocalHistorySummary(LocalHistory):
    F_STATES = ['u', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10','f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19','f20']

    # pd.DataFrame(LocalHistorySummary(SimulationLog[0]['agent3']),columns=['location_state.data','coef.data','fire_state.data','belief.data','action.data']).to_string()

    return pd.DataFrame([[X[0], X[5], F_STATES[np.argmax(X[7])], X[1],X[3],X[11], X[7]] for X in LocalHistory],
                        columns=['location_state.data', 'coef.data', 'fire_state.data','action.data','RW.data','obs.data', 'belief.data'])

ExpirenceFileName='newCon_ver10_ep1'
FilePath='AgentsExpirience/Rewards_10_' + ExpirenceFileName
def LoadRewardProgres(FilePath):
    with open(FilePath, 'rb') as file:
        return pickle.load(file)
#LoadRewardProgres(FilePath='POMDP20x6/Experience/Rewards_300_newCon_ver10_ep4')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Tarik Selimovic')
    obs_noise = 0.15
    belief = 0.025
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
        AgentState("f11"): belief,
        AgentState("f12"): belief,
        AgentState("f13"): belief,
        AgentState("f14"): belief,
        AgentState("f15"): belief,
        AgentState("f16"): belief,
        AgentState("f17"): belief,
        AgentState("f18"): belief,
        AgentState("f19"): belief,
        AgentState("f20"): belief,
    })
    init_true_state = random.choice(
        [AgentState(s) for s in {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9','f10','f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19','f20'}])
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

    EpizodeID='15000'
    Models = [
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent1_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent2_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent3_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent4_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent5_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              'POMDP20x6_AMD/FireModel_NewRW_NewCons_ver10/agent6_NN_model_verFirstTrain_Epizode_4_'+EpizodeID,
              ]



    Models_newCon_ver10 = ['FireModel_NewRW_NewCons_ver10/agent1_NN_model_verFirstTrain_3_Final',
                           'FireModel_NewRW_NewCons_ver10/agent2_NN_model_verFirstTrain_3_Final',
                           'FireModel_NewRW_NewCons_ver10/agent3_NN_model_verFirstTrain_3_Final']

# 3_5100 Dobar model
    agent_1 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent1', Models[0],SequentiallyScouting={'Indx':[1,2,3,4],'Dir':'F'})

    agent_2 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent2', Models[1],SequentiallyScouting={'Indx':[10,9,8,7],'Dir':'B'})

    agent_3 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent3', Models[2],SequentiallyScouting={'Indx':[4,5,6,7],'Dir':'F'})

    agent_4 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent4', Models[3],SequentiallyScouting={'Indx':[11,12,13,14],'Dir':'F'})

    agent_5 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent5', Models[4],SequentiallyScouting={'Indx':[20,19,18,17],'Dir':'B'})

    agent_6 = CustomAgent(init_belief,
                          PolicyModel(),
                          TransitionModel(),
                          ObservationModel(obs_noise),
                          RewardModel(), 'agent6', Models[5],SequentiallyScouting={'Indx':[14,15,16,17],'Dir':'F'})


    Agents = [agent_1, agent_2, agent_3,agent_4, agent_5, agent_6]

    NumberOfSimulation = 100

    # Agents=[agent_2]
    MaxIteration = 30

    agent_indx = 0

    NumberOfIteration = [{} for i in range(NumberOfSimulation)]  # np.zeros(NumberOfSimulation)
    SimulationLog = []
    for simulation in range(NumberOfSimulation):
        Iteration = 0
        env.Reset()
        print('Sim EnvState:' + env.state.name)
        [agent.Reset() for agent in Agents]
        [agent.ForgetHistory() for agent in Agents]
        AgentInMission = len(Agents)
        DDD = False
        ItersPerAgent = np.zeros(len(Agents))
        ItersPerAgent = {}
        for ag in Agents:
            ItersPerAgent[ag.AgentName] = 0
        AgemtSimLog = {}
        AgemtSimLog['EnvState'] = env.state.name
        while DDD != True:
            print('--------------ITERATION:' + str(Iteration) + '---------------')
            IterationBeliefVectors = []
            IterationStateVectors = []
            for Agent in Agents:

                if Agent.InMission:
                    print('Agent:' + Agent.AgentName)
                    print('State:' + Agent.cur_agent_state)
                    print('Belief:' + str(Agent.cur_belief.histogram))
                    #Agent.StateInfo()
                    #ActionID = np.argmax(Agent.model.predict(Agent.preprocess_state(Agent.cur_agent_state,
                    #                                                                list(
                    #                                                                    Agent.cur_belief.histogram.values()))))
                    AgAction=Agent.GetBestActionAccordintToReward(Agent.cur_agent_state,Agent.cur_belief)
                    NextAction=AgAction.name
                    #NextAction = Agent.AGENT_ACTIONS[ActionID]
                    print('Agent:' + Agent.AgentName + '  NextModelAction:' + NextAction)
                    OldState, NewState, Action, Done, RW, oldCoef, newCoef, prevBelief, curBelief, OldConensus,real_observation_agent = Agent.TakeAction( env, AgentAction(NextAction))

                    if real_observation_agent is None:
                        obs='NULL'
                    else:
                        obs=real_observation_agent.name
                    print('Agent:' + Agent.AgentName + '  Observation:' + obs+ ' RW:'+str(RW))

                    if obs!='NULL':
                        IterationBeliefVectors.append(list(Agent.cur_belief.histogram.values()))
                        IterationStateVectors.append(Agent.cur_agent_state_vector)
                    Agent.Remeber(OldState, Action.name, NewState, RW, Done, oldCoef, newCoef, prevBelief, curBelief,
                                  OldConensus,obs,env.state.name,'model')
                    ItersPerAgent[Agent.AgentName] = ItersPerAgent[Agent.AgentName] + 1
                    if Done or ItersPerAgent[Agent.AgentName] == MaxIteration:
                        AgentInMission = AgentInMission - 1

                        AgemtSimLog[Agent.AgentName] = LocalHistorySummary(Agent.LocalHistory)
                        AgemtSimLog[Agent.AgentName+"_LocalHistory"] = Agent.LocalHistory
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

        print('EnvState:' + env.state.name)
        #[Agent.StateInfo() for Agent in Agents]
        NumberOfIteration[simulation]['Iterations'] = Iteration
        NumberOfIteration[simulation]['SimulationID'] = simulation
        States = [F_STATES[np.argmax(list(Ag.cur_belief.histogram.values()))] if np.argmax(list(Ag.cur_belief.histogram.values()))!=0 else env.state.name for Ag in Agents]
        NumberOfIteration[simulation]['Success'] = (States[0] == States[1] == States[2]==States[3] == States[4] == States[5]  == env.state.name)

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


def GetSimulationInfo(SimulationID):
    print('Env State-----------------')
    print(SimulationLog[SimulationID]['EnvState'])
    print('Agent1--------------------')
    print(SimulationLog[SimulationID]['agent1'].to_string())
    print('Agent2--------------------')
    print(SimulationLog[SimulationID]['agent2'].to_string())
    print('Agent3--------------------')
    print(SimulationLog[SimulationID]['agent3'].to_string())
    print('Agent4--------------------')
    print(SimulationLog[SimulationID]['agent4'].to_string())
    print('Agent5--------------------')
    print(SimulationLog[SimulationID]['agent5'].to_string())
    print('Agent6--------------------')
    print(SimulationLog[SimulationID]['agent6'].to_string())

def GetAgentIterationReward(SimulationID,AgentName, IterationID):
    Agents_Name=['agent1','agent2','agent3','agent4','agent5','agent6']
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
    plt.figure(figsize=(12, 4))

    plt.subplot(611)
    plt.plot([R[0] for R in Rewards])
    plt.title('agent1 reward')

    plt.subplot(612)
    plt.plot([R[1] for R in Rewards])
    plt.title('agent2 reward')

    plt.subplot(613)
    plt.plot([R[2] for R in Rewards])
    plt.title('agent3 reward')
    plt.suptitle('Categorical Plotting')

    plt.subplot(614)
    plt.plot([R[3] for R in Rewards])
    plt.title('agent4 reward')
    plt.suptitle('Categorical Plotting')

    plt.subplot(615)
    plt.plot([R[4] for R in Rewards])
    plt.title('agent5 reward')
    plt.suptitle('Categorical Plotting')

    plt.subplot(616)
    plt.plot([R[5] for R in Rewards])
    plt.title('agent6 reward')
    plt.suptitle('Categorical Plotting')

    plt.show()

#Rewards=LoadRewardProgres(FilePath='POMDP20x6_AMD/Experience/Rewards_300_newCon_ver10_ep4')
#GetFigureReward(Rewards)

#Agents[0].LoadLocalHistory('AgentsExpirience/LH_'+Agents[0].AgentName+'_'+'newCon_ver8_ep1')
#Agents[0].LocalHistory=random.sample(Agents[0].LocalHistory, 500)
#Agents[0].Replay(500,1024)