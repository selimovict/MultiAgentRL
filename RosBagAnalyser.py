import bagpy
from bagpy import bagreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GetList(InputString):
    li = list(InputString.replace('(','').replace(')','').replace(' ','').split(","))
    Result=[float(x) for x in li]
    return Result

b = bagreader('RosBag/UAV_logs_Room4_Sim2.bag')

UAV_Names=['UAV1','UAV2','UAV3']

Topic_List=['/UAV1/UAV_Log','/UAV2/UAV_Log','/UAV3/UAV_Log']
Uav_infos={}
Uav_log_values={}
for Topic in Topic_List:
    UAV_data=b.message_by_topic(Topic)
    UAV_table_data=pd.read_csv(UAV_data)
    UAV_table_data['belief.data'] = UAV_table_data['belief.data'].apply(lambda x: GetList(x))
    Uav_Info=UAV_table_data[['Time','location_state.data','coef.data','fire_state.data','belief.data','action.data']].to_string()
    UAV_Logs=np.array([X for X in UAV_table_data[['Time','fire_state.data','coef.data','belief.data','iteration.data','location_state.data','action.data']].values])
    Uav_log_values[Topic]=UAV_Logs
    Uav_infos[Topic]=Uav_Info

BeliefVectors={}
TimeVectors={}
for Topic in Topic_List:
    BeliefVectors_List=[x[3] for x in Uav_log_values[Topic]]
    Time=[x[0] for x in Uav_log_values[Topic]]
    BeliefVectors[Topic]=BeliefVectors_List
    TimeVectors[Topic]=Time

plt.figure('UAV1,UAV2,UAV3')
TT=[131,132,133]
#plt.subplot(131)
for i,Topic in enumerate(Topic_List):
    Bt_Uav=Bt=BeliefVectors[Topic]
    Time=TimeVectors[Topic]
    plt.subplot(TT[i])
    plt.title('Belief per iteration [%s] '%Topic)
    plt.plot(Time,Bt_Uav,label=['u','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
    plt.xlabel('Iteration')
    plt.ylabel('Value of Belief')
    plt.legend()
plt.show()



for i,Topic in enumerate(Topic_List):
    print('----------------------')
    print(Topic)
    print(Uav_infos[Topic])


#Bt_Uav_1=Bt=BeliefVectors[Topic_List[0]]

#plt.plot(Bt_Uav_1)
#plt.plot(Bt_Uav_1,label=['u','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
#plt.legend()
#U0=[t[0] for t in Bt]

#plt.figure("Belief vector")
#plt.title('Belief per iteration')
#plt.plot(np.arange(len(Bt_Uav_1)),Bt_Uav_1,label=['u','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
#plt.xlabel('Iteration')
#plt.ylabel('Value of Belief')
#plt.legend()
#plt.show()



#UAV1_data=b.message_by_topic(Uav_Log_TopicNames[0])
#UAV1_table_data=pd.read_csv(UAV1_data)
#UAV1_Logs=np.array([X for X in UAV1_table_data[['Time','coef.data']].values])

#for t in b.topics:
#    data = b.message_by_topic(t)
#    print(data)

#df_imu = pd.read_csv(data)
