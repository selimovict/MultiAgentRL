import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
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

csv_logger = CSVLogger('log.csv', append=True, separator=';')

class CustomEnvironment(pomdp_py.Environment):

    def __init__(self,init_true_state,TransitionModel, RewardModel):
        super().__init__(init_true_state,TransitionModel, RewardModel)

    def Reset(self):
        next_state = random.choice(self.transition_model.get_all_states())
        self.apply_transition(next_state)

    def TakeAction(self,Agent_1_Action,Agent_2_Action):

        return self.reward_model.join_reward_func(self.state, [Agent_1_Action, Agent_2_Action])

class CustomAgent(pomdp_py.Agent):

    def GetActionByPolicy(self,MyObs,HisObs):
        Q_target=self.model.predict(self.preprocess_state2(MyObs,HisObs))
        return self.ACTIONS[np.argmax(Q_target)]


    def LoadPolicy(self,ModelLink):
        self.model=keras.models.load_model(ModelLink)

    def TakeAction(self,Environment,Action):
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

        if Action.name.startswith("open"):
            Done=True

        return Done
        
        
        

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def choose_action2(self,my_Obs_Old,my_Obs, epsilon):

        #if self.AgentName=='agent_1':
        #    return  self.ACTIONS[0]

        ob_tL_my_Obs_Old = my_Obs_Old.histogram[TigerState('tiger-left')]
        ob_tR_my_Obs_Old  = my_Obs_Old.histogram[TigerState('tiger-right')]

        ob_tL_my_Obs = my_Obs.histogram[TigerState('tiger-left')]
        ob_tR_my_Obs = my_Obs.histogram[TigerState('tiger-right')]

        My_Obs_Final_Old = [ob_tL_my_Obs_Old, ob_tR_my_Obs_Old]
        My_Obs_Final = [ob_tL_my_Obs, ob_tR_my_Obs]

        Processed_State=self.preprocess_state2(My_Obs_Final_Old,My_Obs_Final)


        RandomAction=random.choice(self.ACTIONS)

        ModelAction=self.ACTIONS[np.argmax(self.model.predict(Processed_State))]

        if self.Reseted :
            self.Reseted = False
            return self.ACTIONS[0]

        if (np.random.random() <= epsilon):
            print(' ----- RANDOM ACTION -----')
            return  RandomAction
        else:
            print(' ----- MODEL ACTION -----')
            return  ModelAction

        #return RandomAction if (np.random.random() <= epsilon) else ModelAction

    def choose_action(self,old_other,old_state, state, epsilon):
        ob_tL_Old_Other = old_other.histogram[TigerState('tiger-left')]
        ob_tR_Old_Other = old_other.histogram[TigerState('tiger-right')]
        st_Old_Other = [ob_tL_Old_Other, ob_tR_Old_Other]

        ob_tL_Old = old_state.histogram[TigerState('tiger-left')]
        ob_tR_Old = old_state.histogram[TigerState('tiger-right')]

        st_Old = [ob_tL_Old,ob_tR_Old]

        ob_tL = state.histogram[TigerState('tiger-left')]
        ob_tR = state.histogram[TigerState('tiger-right')]
        st=[ob_tL,ob_tR]
        Obs = self.preprocess_state(st_Old,st)

        ACTIONS = [TigerAction(s) for s in ["listen","open-left", "open-right"]]
        RandomAction=random.choice(ACTIONS)

        ModelAction=ACTIONS[np.argmax(self.model.predict(Obs))]

        if self.Reseted :
            self.Reseted = False
            return ACTIONS[0]

        if (np.random.random() <= epsilon):
            print(' ----- RANDOM ACTION -----')
            return  RandomAction
        else:
            print(' ----- MODEL ACTION -----')
            return  ModelAction

        #return RandomAction if (np.random.random() <= epsilon) else ModelAction

    def preprocess_state(self,old_state, state):
        return np.reshape(old_state+state, [1, self.NumOfStates])

    def preprocess_state2(self,My_Hist,His_Hist):
        return np.reshape(My_Hist+His_Hist, [1, self.NumOfStates])


    def Replay(self,batch_size):

        x_batch, y_batch = [], []
        minibatch = random.sample(self.Memory, min(len(self.Memory), batch_size))
        for obs_old,obs, actionID, obs_new, JointReward, done in minibatch:

            process_obs=self.preprocess_state(obs_old,obs)
            Q_target = self.model.predict(process_obs)
            if done:
                Q_target[0][actionID]=JointReward
            else:
                process_obs_next = self.preprocess_state(obs,obs_new)
                Q_target[0][actionID] = (1-self.r_lr)*(Q_target[0][actionID])+ self.r_lr*( JointReward + self.gamma * np.max(self.model.predict(process_obs_next)[0]))
            x_batch.append(process_obs[0])
            y_batch.append(Q_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),epochs = 24, verbose=2)

        print('------ R E P L A Y ------ ')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def Replay2(self,batch_size):

        x_batch, y_batch = [], []
        my_minibatch,his_minibatch=zip(*random.sample(list(zip(self.MyHistory,self.HisHistory)),min(len(self.MyHistory), batch_size)))
        #my_minibatch = random.sample(self.MyHistory, min(len(self.MyHistory), batch_size))
        #his_minibatch = random.sample(self.HisHistory, min(len(self.HisHistory), batch_size))

        for (my_old_obs,my_obs, my_actionID, my_obs_new, my_JointReward, my_done),(his_obs, his_actionID, his_obs_new, his_JointReward, his_done) in zip(my_minibatch,his_minibatch):
            if my_JointReward!=his_JointReward:
                print("----- E R R O R --------")
            if my_done!=his_done:
                print("----- E R R O R --------")

            process_obs=self.preprocess_state2(my_old_obs,my_obs)
            Q_target = self.model.predict(process_obs)

            if my_done or his_done:
                Q_target[0][my_actionID] = my_JointReward
            else:
                process_obs_next = self.preprocess_state2(my_obs,my_obs_new)
                Q_target_new = self.model.predict(process_obs_next)
                Q_target[0][my_actionID] = (1-self.r_lr)*(Q_target[0][my_actionID])+ self.r_lr*(my_JointReward + self.gamma * np.max(Q_target_new[0]))

            x_batch.append(process_obs[0])
            y_batch.append(Q_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch),epochs = 12, verbose=0)

        print('------ R E P L A Y ------ ')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def Remeber2(self,MyHistory,HisHistory,JointReward,Done):
        My_OldObs=[MyHistory[0].histogram[TigerState('tiger-left')],MyHistory[0].histogram[TigerState('tiger-right')]]
        My_Action= 0 if MyHistory[1].name=='listen' else ( 1 if MyHistory[1].name=='open-left' else 2 )
        My_NewObs=[MyHistory[2].histogram[TigerState('tiger-left')],MyHistory[2].histogram[TigerState('tiger-right')]]
        My_OldOldObs = [MyHistory[3].histogram[TigerState('tiger-left')],MyHistory[3].histogram[TigerState('tiger-right')]]

        self.MyHistory.append([My_OldOldObs,My_OldObs, My_Action, My_NewObs, JointReward,Done])

        His_OldObs=[HisHistory[0].histogram[TigerState('tiger-left')],HisHistory[0].histogram[TigerState('tiger-right')]]
        His_Action= 0 if HisHistory[1].name=='listen' else ( 1 if HisHistory[1].name=='open-left' else 2 )
        His_NewObs=[HisHistory[2].histogram[TigerState('tiger-left')],HisHistory[2].histogram[TigerState('tiger-right')]]

        self.HisHistory.append([His_OldObs, His_Action, His_NewObs, JointReward, Done])


    def Remeber(self,OldObservation,Observation,Action,NewObservations,JointReward,Done):



        ob_tL_Old = OldObservation.histogram[TigerState('tiger-left')]
        ob_tR_Old = OldObservation.histogram[TigerState('tiger-right')]

        ob_tL = Observation.histogram[TigerState('tiger-left')]
        ob_tR = Observation.histogram[TigerState('tiger-right')]


        ob_tL_New = NewObservations.histogram[TigerState('tiger-left')]
        ob_tR_New = NewObservations.histogram[TigerState('tiger-right')]

        Obs = [ob_tL,ob_tR]
        Obs_New = [ob_tL_New, ob_tR_New]
        Obs_Old = [ob_tL_Old, ob_tR_Old]

        self.ObservationsList.append(Obs)

        act = [0, 0, 0]
        ActionID=0
        if(Action.name=='listen'):
            act[0]=JointReward
            ActionID = 0
        elif(Action.name=='open-left'):
            act[1] = JointReward
            ActionID = 1
        elif(Action.name=='open-right'):
            act[2] = JointReward
            ActionID = 2

        self.Epizodes[-1].append([Obs,act,Obs_New,JointReward])
        self.Memory.append([Obs_Old,Obs, ActionID, Obs_New, JointReward, Done])
        self.ValuesList.append(act)

    def Reset(self):
        self.Reseted=True
        self.set_belief(self.my_init_belief)
        self.OldBelief=self.my_init_belief
        self.OldOldBelief = self.my_init_belief

    def __init__(self, init_belief, PolicyModel,TransitionModel,ObservationModel,RewardModel,AgentName):
        super().__init__(init_belief, PolicyModel,TransitionModel,ObservationModel,RewardModel)
        self.ACTIONS = [TigerAction(s) for s in ["listen", "open-left", "open-right"]]
        self.AgentName=AgentName
        self.my_init_belief=init_belief
        self.Epizodes=[]
        self.Memory=[]
        self.MyHistory = []
        self.HisHistory= []
        self.Epizodes.append([])
        self.ObservationsList = []
        self.ValuesList = []
        self.gamma = 1
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001
        self.alpha_decay = 0.0001
        self.batch_size = 64

        self.r_lr=0.9

        self.NumOfStates=4

        self.OldBelief=init_belief
        self.OldOldBelief = init_belief

        self.Action=None
        self.Old_Action=None

        self.Reseted=True

        self.model = Sequential()
        self.model.add(Dense(8, input_dim=self.NumOfStates, activation='relu'))
        self.model.add(Dense(16, activation='sigmoid'))
        #self.model.add(layers.Embedding(input_dim=5, output_dim=8))
        #self.model.add(layers.LSTM(16))

        #self.model.add(layers.Embedding(input_dim=5, output_dim=16))
        #self.model.add(layers.LSTM(32))

        #self.model.add(Dense(32, activation='relu'))
        #self.model.add(Dropout(0.2))

        self.model.add(Dense(64, activation='sigmoid'))
        self.model.add(Dense(3, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))


class TigerState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerState):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerState(%s)" % self.name

    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")

class TigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerAction(%s)" % self.name


class TigerObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, TigerObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "TigerObservation(%s)" % self.name

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            if observation.name == next_state.name: # heard the correct growl
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5


    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)


    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return TigerState(state.name)


    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [TigerState(s) for s in {"tiger-left", "tiger-right"}]


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


    def join_reward_func(self,state,Actions):
        DecTigerProblem=False
        if(len(Actions)==2):
            DecTigerProblem=True
        else:
            return

        actions_name=[a.name.replace('open-','') for a in Actions]

        elem_pos=2 if state.name=='tiger-left' else 3
        JointActions=[('listen','listen',-2,-2),
                      ('listen','left',-101,+9),
                      ('listen','right',+9,-101),
                      ('left','listen',-101,+9),
                      ('left','left',-50,+20),
                      ('left','right',-100,-100),
                      ('right','listen',+9,-101),
                      ('right','left',-100,-100),
                      ('right','right',+20,-50)]

        JointReward=[s[elem_pos] for i,s in enumerate(JointActions) if actions_name[0] in s[0] and actions_name[1] in s[1]]
        return JointReward[0]

class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {TigerAction(s) for s in {"listen","open-left", "open-right"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]


    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS


class TigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())


        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="TigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram({TigerState("tiger-left"): belief,
                                          TigerState("tiger-right"): 1.0 - belief})
        tiger_problem = TigerProblem(obs_noise,  # observation noise
                                     init_true_state, init_belief)
        tiger_problem.agent.set_belief(init_belief, prior=True)
        return tiger_problem


class DecTigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())


        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())

        super().__init__(agent, env, name="TigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram({TigerState("tiger-left"): belief,
                                          TigerState("tiger-right"): 1.0 - belief})
        tiger_problem = TigerProblem(obs_noise,  # observation noise
                                     init_true_state, init_belief)
        tiger_problem.agent.set_belief(init_belief, prior=True)
        return tiger_problem


def test_planner(tiger_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): an instance of the tiger problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(tiger_problem.agent.tree)
            import pdb; pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        print("True state: %s" % tiger_problem.env.state)
        print("Belief: %s" % str(tiger_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(tiger_problem.env.reward_model.sample(tiger_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        real_observation = TigerObservation(tiger_problem.env.state.name)
        print(">> Observation: %s" % real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(tiger_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          tiger_problem.agent.observation_model,
                                                          tiger_problem.agent.transition_model)
            tiger_problem.agent.set_belief(new_belief)

            if action.name.startswith("open"):
                # Make it clearer to see what actions are taken until every time door is opened.
                print("\n")

def main():
    init_true_state = random.choice([TigerState("tiger-left"),
                                     TigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})

    tiger_problem = TigerProblem(0.15,  # observation noise
                                 init_true_state, init_belief)




    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(tiger_problem, vi, nsteps=3)

    # Reset agent belief
    tiger_problem.agent.set_belief(init_belief, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=50,
                           rollout_policy=tiger_problem.agent.policy_model,
                           show_progress=True)
    test_planner(tiger_problem, pouct, nsteps=10)
    TreeDebugger(tiger_problem.agent.tree).pp

    # Reset agent belief
    tiger_problem.agent.set_belief(init_belief, prior=True)
    tiger_problem.agent.tree = None

    print("** Testing POMCP **")
    tiger_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=tiger_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)
    test_planner(tiger_problem, pomcp, nsteps=10)
    TreeDebugger(tiger_problem.agent.tree).pp




if __name__ == '__main__':
    ##main()

    #ACTIONS = [TigerAction(s) for s in {"open-left", "open-right", "listen"}]
    #ACTIONS = [TigerAction(s) for s in { "listen" , "open-left" , "open-right"}]
    #Initial Environment and Belief

    init_true_state = random.choice([TigerState("tiger-left"),
                                    TigerState("tiger-right")])


    init_belief = pomdp_py.Histogram({TigerState("tiger-left"): 0.5,
                                      TigerState("tiger-right"): 0.5})


    agent_1 = CustomAgent(init_belief,
                       PolicyModel(),
                       TransitionModel(),
                       ObservationModel(),
                       RewardModel(),'agent_1')



    agent_2 = CustomAgent(init_belief,
                       PolicyModel(),
                       TransitionModel(),
                       ObservationModel(),
                       RewardModel(),'agent_2')

    env = CustomEnvironment(init_true_state,
                               TransitionModel(),
                               RewardModel())

    Games=0
    j=0
    scores = deque(maxlen=100)


    SuccessEpizode=0
    for Epizode in range(1000):
        env.Reset()
        agent_1.Reset()
        agent_2.Reset()
        Done=False
        print("==== Episode %d ==== " % (Epizode+1))
        i=0
        print("True state:", env.state)
        Joint_Reward=0

        while (not Done):
            print("==== Iteration %d ==== " % (i + 1))

            #Agent_1_Action = agent_1.choose_action(agent_2.OldBelief, agent_1.OldBelief, agent_1.cur_belief,
            #                                       agent_1.epsilon)
            #Agent_2_Action = agent_2.choose_action(agent_1.OldBelief, agent_2.OldBelief, agent_2.cur_belief,
            #                                       agent_2.epsilon)

            Agent_1_Action = agent_1.choose_action2(agent_1.OldBelief, agent_1.cur_belief,agent_1.epsilon)
            Agent_2_Action = agent_2.choose_action2(agent_2.OldBelief, agent_2.cur_belief,agent_2.epsilon)

            print("OLD Agent_1_Belief:", agent_1.OldBelief)
            print("OLD Agent_2_Belief:", agent_2.OldBelief)

            print("Agent_1_Belief:", agent_1.cur_belief)
            print("Agent_2_Belief:", agent_2.cur_belief)

            print("Action_1:", Agent_1_Action)
            print("Action_2:", Agent_2_Action)

            Done1 = agent_1.TakeAction(env, Agent_1_Action)
            Done2 = agent_2.TakeAction(env, Agent_2_Action)

            join_reward = env.TakeAction(Agent_1_Action, Agent_2_Action)


            print("NEW Agent_1_Belief :", agent_1.cur_belief)
            print("NEW Agent_2_Belief :", agent_2.cur_belief)
            print("JOINT REWARD :", join_reward)
            i=i+1
            Done=(Done1 or Done2)

            agent_1_history = [agent_1.OldBelief, Agent_1_Action, agent_1.cur_belief,agent_1.OldOldBelief]
            agent_2_history = [agent_2.OldBelief, Agent_2_Action, agent_2.cur_belief,agent_2.OldOldBelief]

            agent_1.Remeber2(agent_1_history, agent_2_history, join_reward, Done)
            agent_2.Remeber2(agent_2_history, agent_1_history, join_reward, Done)


            Joint_Reward = Joint_Reward + join_reward

        scores.append(i)
        mean_score = np.mean(scores)


        print('[Episode {}] - Mean survival time over last 1 episodes was {} ticks.'.format(Epizode, mean_score))
        print('[Episode {}] - Uspjesnih epizoda je bilo {}.'.format(Epizode, SuccessEpizode))
        if Epizode>5:
            print('[Episode {}] - Procentualno to je broj   {:10.2f}%'.format(Epizode, SuccessEpizode/Epizode*100))
        print("Last Action Agent_1:", agent_1.Action)
        print("Last Action Agent_2:", agent_2.Action)

        if Joint_Reward>0:
            SuccessEpizode=SuccessEpizode+1
            print('- - - S U C C E S S E D - - -')
        else:
            print('- - - F A I L E D - - -')
        agent_1.Replay2(agent_1.batch_size)
        agent_2.Replay2(agent_2.batch_size)

    agent_1.model.save('Tiger_Model_2/Agent_1_Hist_Agent_1NN_Agent_2ActionNN')
    agent_2.model.save('Tiger_Model_2/Agent_2_Hist_Agent_1NN_Agent_2ActionNN')