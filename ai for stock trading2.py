import h5py
import talib
import datetime
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader as pdr
#from collections import deque
#import tensorflow as tf
#from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Input,Dense
#from tensorflow.keras import optimizers
start = datetime.datetime(2014, 2, 10)
end = datetime.datetime(2021, 2, 10)
stock = pdr.get_data_yahoo('DAL', start=start, end=end)
#stock['Adj Close'].plot(grid=True)
adx = talib.ADX(stock['High'], stock['Low'], stock['Close'], timeperiod=14)
cci = talib.CCI(stock['High'], stock['Low'], stock['Close'], timeperiod=14)
rsi = talib.RSI(stock['Close'], timeperiod=14)
stock['ADX'] = adx
stock['CCI'] = cci
stock['RSI'] = rsi
data = stock[['Adj Close', 'ADX', 'CCI', 'RSI']].dropna()
class SingleStockTradingEnv(): 
    def __init__(self, 
                 data=None, 
                 initial_capital=100000, 
                 transaction_cost=0):
        
        self.price_history = data.iloc[:,0]
        self.indicator_history1 = data.iloc[:,1]
        self.indicator_history2 = data.iloc[:,2]
        self.indicator_history3 = data.iloc[:,3]
        self.n_step = self.price_history.shape[0] # n stock
        self.n_stock = 1
        
        self.n_buy = None
        self.n_sell = None
        self.action_plot = None
        
        self.begin_portfolio_value = initial_capital
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.cur_step = None
        self.position = None
        self.stock_price = None
        self.indicator1 = None
        self.indicator2 = None
        self.indicator3 = None
        self.cash = None
        
        self.action_list = list(map(list, itertools.product(np.arange(-100,200,100), repeat=self.n_stock))) #Posible action_space
        self.action_space = np.arange(len(self.action_list)**self.n_stock) #Action space <------ [0,1,2]
        self.state_space = 5*self.n_stock+1 #State space <------ [pos, price, indi1, indi2, indi3, cash]
        
        self.reset()
         
    def reset(self):
        self.action_plot = []
        self.n_buy = 0
        self.n_sell = 0
        self.cur_step = 0
        self.position = np.zeros(self.n_stock)
        self.stock_price = self.price_history[self.cur_step]
        self.indicator1 = self.indicator_history1[self.cur_step]
        self.indicator2 = self.indicator_history2[self.cur_step]
        self.indicator3 = self.indicator_history3[self.cur_step]
        self.cash = self.initial_capital
        return self._get_obs()
        
    def step(self, action):
        assert action in self.action_space
        prev_val = self.port_val()
        
        self.cur_step += 1
        self.stock_price = self.price_history[self.cur_step]
        self.indicator1 = self.indicator_history1[self.cur_step]
        self.indicator2 = self.indicator_history2[self.cur_step]
        self.indicator3 = self.indicator_history3[self.cur_step]
        
        self._trade(action)
        
        cur_val = self.port_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        
        
        return self._get_obs(), reward, done
    
    def _get_obs(self): 
        obs = np.empty(self.state_space)
        obs[0] = self.position
        obs[1] = self.stock_price
        obs[2] = self.indicator1
        obs[3] = self.indicator2
        obs[4] = self.indicator3
        obs[5] = self.cash
        return obs
      
    def port_val(self):
        return self.position*self.stock_price + self.cash 
        
    def _trade(self,action):      
        #e.g. [-500] mean:
        #sell 500 unit
        pos_size = self.action_list[action][0]
        
        #buy
        if pos_size > 0 and self.cash >= self.stock_price*pos_size: 
            self.position += pos_size
            self.cash -= self.stock_price*pos_size
            self.n_buy += 1
            self.action_plot.append(2)
         
        #sell
        elif pos_size < 0 and self.position >= abs(pos_size): 
            self.position -= abs(pos_size)
            self.cash += self.stock_price*abs(pos_size)
            self.n_sell += 1
            self.action_plot.append(0)
        #hold
        else:
            self.action_plot.append(1) 
class Agent():
    def __init__(self):
        n_bins = 10
        self.position_bins = pd.cut([0,2000],bins=n_bins,retbins=True)[1][1:-1]
        self.stock_price_bins = pd.cut([0,60],bins=n_bins,retbins=True)[1][1:-1]
        self.indicator1_bins = pd.cut([0,30],bins=n_bins,retbins=True)[1][1:-1]
        self.indicator2_bins = pd.cut([-200,200],bins=n_bins,retbins=True)[1][1:-1]
        self.indicator3_bins = pd.cut([0,70],bins=n_bins,retbins=True)[1][1:-1]
        self.cash_bins = pd.cut([0,150000],bins=n_bins,retbins=True)[1][1:-1]
        self.learning_rate = 0.01
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay_factor =0.999
        self.total_reward2 = []
        self.total_reward_test = []
        self.total_reward_test_2 = []
        self.average_total_reward = []
        self.q_table =np.zeros((n_bins,n_bins,n_bins,n_bins,n_bins,n_bins)+(3,))
    
    def trade(self,env,number_of_episode =10): 
        for i in range(number_of_episode):
            print("episode {} of {}".format(i+1,number_of_episode))
            observation = env.reset()
            position,stock_price,indicator1,indicator2,indicator3,cash = observation
            state  = (self.digitize(position,self.position_bins),self.digitize(stock_price,self.stock_price_bins),self.digitize(indicator1,self.indicator1_bins),self.digitize(indicator2,self.indicator2_bins),self.digitize(indicator3,self.indicator3_bins),self.digitize(cash,self.cash_bins))
            self.epsilon *= self.epsilon_decay_factor
            total_reward =  0
            done = False
            while not done:
                if self.q_table_empty(state) or self.prob(self.epsilon):
                    action= self.action_by_random(env)
                else:
                    action= self.get_action_with_highreward(state)
                
                new_observation,reward,done = env.step(action)
                total_reward += reward
                new_position,new_stock_price,new_indicator1,new_indicator2,new_indicator3,new_cash = new_observation
                new_state = (self.digitize(new_position,self.position_bins),self.digitize(new_stock_price,self.stock_price_bins),self.digitize(new_indicator1,self.indicator1_bins),self.digitize(new_indicator2,self.indicator2_bins),self.digitize(new_indicator3,self.indicator3_bins),self.digitize(new_cash,self.cash_bins))
                #q-tables
                self.q_table[state][action]+=self.learning_rate*(reward+self.discount_factor*self.getreward(new_state)-self.q_table[state][action])
                state = new_state
                
                
            self.total_reward2.extend(total_reward)
    
    def test(self,env,number_of_episode =1):
        print("episode {} of {}".format(1,number_of_episode))
        observation = env.reset()
        position,stock_price,indicator1,indicator2,indicator3,cash = observation
        state  = (self.digitize(position,self.position_bins),self.digitize(stock_price,self.stock_price_bins),self.digitize(indicator1,self.indicator1_bins),self.digitize(indicator2,self.indicator2_bins),self.digitize(indicator3,self.indicator3_bins),self.digitize(cash,self.cash_bins))
        total_reward_1 =  0
        total_reward_2 =  0
        done = False
        while not done:
            action= self.get_action_with_highreward(state)
            new_observation,reward,done = env.step(action)
            total_reward_1 = reward
            total_reward_2 += total_reward_1
            new_position,new_stock_price,new_indicator1,new_indicator2,new_indicator3,new_cash = new_observation
            new_state = (self.digitize(new_position,self.position_bins),self.digitize(new_stock_price,self.stock_price_bins),self.digitize(new_indicator1,self.indicator1_bins),self.digitize(new_indicator2,self.indicator2_bins),self.digitize(new_indicator3,self.indicator3_bins),self.digitize(new_cash,self.cash_bins))
            state = new_state
            self.total_reward_test.append(total_reward_1)
            
        self.total_reward_test_2.extend(total_reward_2)
        average_total_reward = (total_reward_2)/env.n_step
        self.average_total_reward.extend(average_total_reward)
        action = np.array(env.action_plot)
        price = env.price_history
        self.df_buy,self.df_sell = self.scater_plot(price,action)
        
    

        
        

            
    
    def q_table_empty(self,state):
        return np.sum(self.q_table[state])==0
    def action_by_random(self,env):
        return random.randint(0,2)
    def get_action_with_highreward(self,state):
        return np.argmax(self.q_table[state])
    def getreward(self,state):
        return np.max(self.q_table[state])
    def prob(self,prob):
        return np.random.random()< prob
    def digitize(self,value,bins):
        return np.digitize(x=value,bins=bins)
    def scater_plot(self,data,action):
        list_buy =[]
        list_sell =[]
        for i in range(len(action)):
            if action[i] ==0:
                list_sell.append([i,data[i]])
            elif action[i] == 2:
                list_buy.append([i,data[i]])
        df_buy = pd.DataFrame(list_buy)
        df_sell = pd.DataFrame(list_sell)
        return df_buy ,df_sell


env = SingleStockTradingEnv(data[:1000])
env2 =SingleStockTradingEnv(data[1000:1300])
agent = Agent()
agent.trade(env)
y=agent.total_reward2
plt.subplot(2,1,1)
plt.figure(1, figsize=(15,5))
plt.plot(y)
plt.xlabel('Episode')
plt.ylabel('reward')
plt.title('Performance')
y_1=agent.test(env2)
plt.subplot(2,1,2)
plt.plot(agent.total_reward_test)
plt.xlabel('Date')
plt.ylabel('reward')
plt.title('Performance')

plt.figure(2, figsize=(10,5))
plt.plot(env2.price_history.reset_index(drop=True))
plt.scatter(agent.df_buy[0],agent.df_buy[1],color='red', lw=0.01)
plt.scatter(agent.df_sell[0],agent.df_sell[1],color='green', lw=0.01)

print("Reward for test is {}".format(agent.total_reward_test_2))
print("Average reward is {}".format(agent.average_total_reward))
plt.show()











