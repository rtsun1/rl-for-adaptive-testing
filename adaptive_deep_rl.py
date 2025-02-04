# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### If there is a GPU, set all training and testing to be done on the GPU, otherwise on the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(device)

### The function to simulate the examinee and generate the response under 3PLM
def RESPOND(item_para, theta, D=1):
    a = item_para[:,0]
    b = item_para[:,1]
    c = item_para[:,2]
    p = (1-c) / (1 + np.exp(-D*a*(theta-b))) + c
    resp = (np.random.rand(1) <= p).astype(int)
    return resp

### The function to calculate the Fisher item information under 3PLM
def FI(item_para, theta, D=1):
    a = item_para[:,0]
    b = item_para[:,1]
    c = item_para[:,2]
    info = D**2*a**2*(1-c) / (c+np.exp(D*a*(theta-b))) / (1+np.exp(-D*a*(theta-b)))**2
    return info

### The function to deploy maximum likelihood estimation under 3PLM for training phase
def MLE(item_paras, resp, D=1):
    a = item_paras[:,0]  # disc
    b = item_paras[:,1]  # diff
    c = item_paras[:,2]  # guessing 1/n_answers
    def mins_likelihood(x):
        logl = 0
        for i in range(len(resp)):  # WHY LOOP?
            p = (1-c[i]) / (1 + np.exp(-D*a[i]*(x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10) 
            logl -= resp[i] * np.log(p) + (1-resp[i]) * np.log((1-p))
        return logl
    theta = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.array(theta).reshape(1,)

### The function to parallelly deploy maximum likelihood estimation under 3PLM for validation and testing phases
def MLE_TEST(item_paras, resp, D=1):
    def mins_likelihood(x):
        logl = 0
        for i in range(resp_i.shape[0]):
            p = (1-c[i]) / (1 + np.exp(-D*a[i]*(x - b[i]))) + c[i]
            p = np.clip(p, 1e-10, 1 - 1e-10) 
            logl -= resp_i[i] * np.log(p) + (1-resp_i[i]) * np.log((1-p))
        return logl  
    theta = np.zeros(resp.shape[1])
    for i in range(resp.shape[1]):
        resp_i = resp[:,i]
        a = item_paras[:,i,0]
        b = item_paras[:,i,1]
        c = item_paras[:,i,2]
        theta[i] = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.expand_dims(theta, axis=0)


### The function to create a Q-network
class Net(nn.Module):
    def __init__(self,
                 input_size,
                 first_hidden,
                 second_hidden,
                 action_space,
                 dropout_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.out = nn.Linear(second_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self,x):
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.dropout(self.fc2(x))
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


### The function to select items by Q-network with epsilon-greedy for training phase
def Choose_Action(item_id, state, epsilon):
    if np.random.randn() >= epsilon:
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device) 
        item_id = torch.from_numpy(item_id).to(device).long()
        action_value = eval_net(state)
        action_value[:,item_id] = torch.zeros(item_id.shape).to(device)
        action = torch.max(action_value, -1)[1].cpu().numpy()
    else: 
        if any(item_id) : action = np.random.choice(np.delete(np.arange(action_space), item_id)).astype('int64').reshape(1,)
        else : action = np.random.choice(np.arange(action_space)).astype('int64').reshape(1,)
    return action

### The function to select items by Q-network for validation and testing phases
def Choose_Action_Test(item_id, state):
    state = torch.FloatTensor(state.swapaxes(0,1)).to(device)
    action_value = eval_net(state).detach().cpu().numpy()
    if item_id.shape[0] > 0 : action_value[np.tile(np.arange(item_id.shape[1])[np.newaxis, :], (item_id.shape[0],1)), item_id] = np.zeros(item_id.shape)
    action = action_value.argmax(axis=1)  # (n_items, n_actions)
    return action

### The function that limits the network parameters to positive, such that the output of Q-network (the expectation of cumulative information) is positive
def Apply_Positive_Constraint(model, min_value=0.0):
    for param in model.parameters():
        param.data = torch.clamp(param.data, min=min_value)
        
        
### The function to train Q-network including validation
def TRAIN(gamma,
          memory_capacity = 1000,
          epsilon = 0.1,
          batch_size = 128,
          q_network_iteration = 40,
          learning_rate = 1e-3,
          training_size = 5000,
          validation_size = 200,
          validation_interval = 200):
    
    global best_valid
        
    loss_func = nn.MSELoss()
    eval_net.train()
    optimizer = optim.Adam(eval_net.parameters(), lr=learning_rate)    
    
    memory = np.zeros((memory_capacity, input_size * 2 + 2))
    memory_counter = 0
    learn_step_counter = 0
    training_loss = 0
    training_reward = 0
    
    
        
    for j in range(training_size) :
        state = np.concatenate((np.zeros(input_size-1), np.random.rand(1) - 0.5))
        item_id = np.array([]).astype('int64')
        resp = np.array([]).astype('int64')
        
        for i in range(test_length) :
            action = Choose_Action(item_id, state, epsilon)
            item_id = np.concatenate((item_id, action))
            resp = np.concatenate((resp, train_valid_resp[j, action]))
            reward = FI(item_bank[action,], state)
            if len(np.unique(resp)) == 1 :
                if resp[-1] == 1 :
                    next_state = np.array([state[-1] + (item_bank[:,1].max() - state[-1]) / 2])
                else : 
                    next_state = np.array([state[-1] - (state[-1] - item_bank[:,1].min()) / 2])
            else : next_state = MLE(item_bank[item_id,], resp)
            
            if input_size > 1 : next_state = np.concatenate((state[-(input_size-1):], next_state))

            memory[memory_counter % memory_capacity, :] = np.hstack((state, action, reward, next_state))
            memory_counter += 1
            state = next_state

            if memory_counter >= batch_size :     
                batch_memory = memory[np.random.choice(min(memory_counter, memory_capacity), batch_size, replace=False), :]
                batch_state = torch.FloatTensor(batch_memory[:, :input_size]).to(device)
                batch_action = torch.LongTensor(batch_memory[:, input_size:input_size+1].astype(int)).to(device)
                batch_reward = torch.FloatTensor(batch_memory[:, input_size+1:input_size+2]).to(device)
                batch_next_state = torch.FloatTensor(batch_memory[:,-input_size:]).to(device)
                
                q_eval = eval_net(batch_state).gather(1, batch_action)
                q_next = target_net(batch_next_state).detach()
                if i == test_length-1 : q_target = batch_reward
                else :  q_target = batch_reward + gamma * q_next.max(1)[0].view(batch_size, 1)
                loss = loss_func(q_eval, q_target)
                        
                optimizer.zero_grad()
                loss.backward()                                
                Apply_Positive_Constraint(eval_net)            
                optimizer.step()
                        
                training_reward += reward[0]
                training_loss += loss.item()
                learn_step_counter += 1
                        
                if learn_step_counter % q_network_iteration == 0: 
                    target_net.load_state_dict(eval_net.state_dict())
               
       
        ### Validation ###
        if (j+1) % validation_interval == 0 :

            eval_net.eval()
            valid_loss = 0
            valid_reward = 0
            valid_bias = np.zeros((test_length, validation_size))

            validation_id = np.random.choice(validation_size, training_size, replace=False)                   
            valid_theta = train_valid_theta[validation_id]
            valid_resp = train_valid_resp[validation_id,:]   

            state = np.concatenate((np.zeros((input_size-1, validation_size)), np.expand_dims((np.random.rand(validation_size)-0.5), axis=0)))
            item_id = np.array([])

            for i in range(test_length):
                action = Choose_Action_Test(item_id, state)
                reward = np.transpose(FI(item_bank[action,], state))
                
                if i == 0 :
                    item_id = action[np.newaxis, :]
                    resp = valid_resp[np.arange(validation_size), item_id]
                else : 
                    item_id = np.concatenate((item_id, action[np.newaxis, :]))
                    resp = np.concatenate((resp, valid_resp[np.arange(validation_size), item_id[-1]][np.newaxis, :]))
                                                         
                theta_0 = np.zeros(validation_size)
                idx_full = np.sum(resp, axis=0) == resp.shape[0]
                idx_zero = np.sum(resp, axis=0) == 0
                idx_norm = np.bitwise_not(idx_full | idx_zero)
                theta_0[idx_full] = state[-1,idx_full] + (item_bank[:,1].max() - state[-1,idx_full]) / 2
                theta_0[idx_zero] = state[-1,idx_zero] + (item_bank[:,1].min() - state[-1,idx_zero]) / 2
                theta_0[idx_norm] = np.squeeze(MLE_TEST(item_bank[item_id[:,idx_norm]], resp[:,idx_norm]))
                
                q_eval = eval_net(torch.FloatTensor(np.transpose(state)).to(device)).gather(1, torch.LongTensor(action[:,np.newaxis,]).to(device))
                if input_size > 1 : state = np.concatenate((state[-(input_size-1):], theta_0[np.newaxis, :]))
                else: state = theta_0[np.newaxis, :]
                q_next = target_net(torch.FloatTensor(np.transpose(state)).to(device)).detach()
                if i == test_length-1 : 
                    q_target = torch.FloatTensor(reward).to(device)
                else : 
                    q_target = torch.FloatTensor(reward).to(device) + gamma * q_next.max()
                valid_loss += loss_func(q_eval, q_target).item()
                valid_reward += np.mean(reward)
                valid_bias[i] = theta_0 - valid_theta
                
                

            step_valid = np.transpose(np.vstack((np.arange(1,test_length+1), np.mean(valid_bias, axis=1), np.sqrt(np.mean(valid_bias**2, axis=1)), np.mean(abs(valid_bias), axis=1))))            
            print("subject: {}, loss: {:.3f}, reward: {:.3f}\n\n{}\n".format(j+1, valid_loss/validation_size, valid_reward, step_valid))            
            
            
            try :
                best_valid
            except NameError :
                best_valid = step_valid[-1,1:][np.newaxis, :]
                torch.save(eval_net, 'dqn_gamma_' + str(gamma) + '_1.t7')
                TEST()

            if (abs(step_valid[-1,1]) < 0.05) & (step_valid[-1,2] < 0.2) & (step_valid[-1,3] < 0.2) :
                best_valid = np.vstack((best_valid, step_valid[-1,1:]))
                torch.save(eval_net, 'dqn_gamma_' + str(gamma) + '_' + str(best_valid.shape[0]) + '.t7')
                
                TEST()

            eval_net.train()

    


### The function to test Q-network
def TEST(testing_size = 5000):
    
    with torch.no_grad():
        eval_net.eval()

        state = np.concatenate((np.zeros((input_size-1, testing_size)), np.expand_dims((np.random.rand(testing_size)-0.5), axis=0)))
        item_id = np.array([])
        dqn_step = np.zeros((1, 4))
    
        for i in range(test_length):
            action = Choose_Action_Test(item_id, state)
            if i == 0 :
                item_id = action[np.newaxis, :]
                resp = test_resp[np.arange(testing_size), item_id]
            else : 
                item_id = np.concatenate((item_id, action[np.newaxis, :]))
                resp = np.concatenate((resp, test_resp[np.arange(testing_size), item_id[-1]][np.newaxis, :]))
            
            theta_0 = np.zeros([1,testing_size])
            idx_full = np.sum(resp, axis=0) == resp.shape[0]
            idx_zero = np.sum(resp, axis=0) == 0
            idx_norm = np.bitwise_not(idx_full | idx_zero)
            theta_0[:,idx_full] = state[-1,idx_full] + (item_bank[:,1].max() - state[-1,idx_full]) / 2
            theta_0[:,idx_zero] = state[-1,idx_zero] + (item_bank[:,1].min() - state[-1,idx_zero]) / 2
            theta_0[:,idx_norm] = MLE_TEST(item_bank[item_id[:,idx_norm]], resp[:,idx_norm])
            
            if i == 0 :
                theta = theta_0
            else :
                theta = np.concatenate((theta, theta_0))
                
            dqn_step = np.vstack([dqn_step, np.array([i+1, np.mean(theta_0 - theta_test), np.sqrt(np.mean((theta_0 - theta_test)**2)), np.mean(abs(theta_0 - theta_test))])])
            print("step {:g}, bias {:.3f}, rmse {:.3f}, mae {:.3f}".format(dqn_step[-1,0], dqn_step[-1,1], dqn_step[-1,2], dqn_step[-1,3]))
            
            if input_size > 1 : state = np.concatenate((state[-(input_size-1):], theta_0))
            else: state = theta_0
            
        user_id = np.repeat(np.arange(1, testing_size+1), test_length).reshape(-1, 1)
        step = np.tile(np.arange(1, test_length+1), testing_size).reshape(-1, 1)
        item_id = (item_id + 1).transpose().reshape(-1, 1)
        resp = resp.transpose().reshape(-1, 1)
        theta_est = theta.transpose().reshape(-1, 1)  
        bias = (theta-theta_test).transpose().reshape(-1, 1)
        dqn_data = np.hstack([user_id, step, item_id, resp, theta_est, bias])
        dqn_data = pd.DataFrame(dqn_data).rename(columns={0:'userID',1:'step',2:'itemID',3:'resp',4:'theta_est',5:'bias'})        
        dqn_data.to_csv('./records_real_bank_responses_DQN_gamma_' + str(gamma) + '_' + str(best_valid.shape[0]) + '.csv')



### Hyperparameters
test_length = 40

input_size = 1
first_hidden = 50
second_hidden = 30
action_space = 371  # item bank size
dropout_rate = 0

gamma = 0.1 # The optial gamma value is in the t7 file name


### Load the item parameters
item_bank = np.array(pd.read_csv('real item bank.csv')[['a','b','c']])  # (n_items, 1[a], 1[b], 1[c])

### Read the real responses and true theta for training
train_valid_theta = np.array(pd.read_csv('true theta for training.csv'))  # table w/ 1 col: theta scalar. rows: subj_id (n_subj[theta])
train_valid_resp = np.array(pd.read_csv('real responses for training.csv'))  # table w/ shape: (n_subj, n_items, 1[if answered multiple times?], 1[1 for right 0 for wrong])
train_valid_theta = train_valid_theta.squeeze()
print(item_bank.shape, train_valid_theta.shape, train_valid_resp.shape)

test_resp = np.array(pd.read_csv('real responses for testing.csv'))
theta_test = np.array(pd.read_csv('true theta for testing.csv'))
theta_test = theta_test.squeeze()
### Create two Q-networks
eval_net = Net(input_size, first_hidden, second_hidden, action_space, dropout_rate).to(device)
target_net = Net(input_size, first_hidden, second_hidden, action_space, dropout_rate).to(device)

### Initialize parameters in Q-networks
eval_net.initialize()
target_net.initialize()

### Start training phase (including validation) 
TRAIN(gamma, training_size=200)



### Load the pre-trained model
eval_net = Net(input_size, first_hidden, second_hidden, action_space, dropout_rate).to(device)

network_id = 1
eval_net = torch.load('dqn_gamma_' + str(gamma) + '_' + str(network_id) + '.t7')

### Start testing
TEST()