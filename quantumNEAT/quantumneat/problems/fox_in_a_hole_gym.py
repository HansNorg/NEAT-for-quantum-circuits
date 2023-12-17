import numpy as np
# import tensorflow as tf
# import tensorflow_quantum as tfq
# import cirq

class FoxInAHole():
    def __init__(self, n_holes=5, env_config={}):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        #super.__init__()

    def reset(self):
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(2*(self.n_holes-2))) #2(n-2) encoding
        # self.state = -1*np.ones(int(self.n_holes)) 
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """

        if action == self.hole_nr:
            reward = 1
            self.state[self.guess_counter]=action
            done = True
        
        elif self.guess_counter == 2*self.n_holes-5:
            reward = -1
            self.state[self.guess_counter]=action
            done = True
        else:
            reward = -1
            self.state[self.guess_counter]=action
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    def unbounded_step(self, action):
        if action == self.hole_nr:
            reward = 1
            done = True
        
        else:
            reward = -1
            done = False
            self.move_fox()
        
        return self.state, reward, done, {}
            
class FoxInAHolev2():
    """
    Class object for the Fox in hole game. The amount of holes, the max game length and the length of the input state are all adjustable.
    """
    def __init__(self, n_holes=5, max_steps= 6, len_state = 4):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        self.len_state = len_state
        self.max_steps = max_steps


    def reset(self):
        """
        Resets the environment.
        """
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(self.len_state)) #n previous picks encoding
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        """
        Moves the fox randomly to a hole either to the right or to the left.
        """
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            state (np.array): numpy array with the last two made guesses (inlcuding the guess made with this functions call).
            reward (int): the reward for the guess.
            done (bool): True if the game is finished, False if not.
            {} (?): Ignore this, its a remnant of the gym environment guidelines. 

        """

        if action == self.hole_nr:
            reward = 1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = True
        
        elif self.guess_counter == self.max_steps-1:
            reward = -1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = False # Changed from True
        else:
            reward = -1
            state = np.roll(self.state, 1)
            state[0]=action
            self.state = state
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    
class FoxInAHoleBounded():
    def __init__(self, n_holes=5, env_config={}, len_state = None):
        self.n_holes = n_holes
        self.hole_nr = None
        self.state = None
        self.guess_counter = 0
        #super.__init__()

    def reset(self):
        # reset the environment to initial random state
        self.hole_nr = np.random.randint(low=0,high=self.n_holes, size= 1)
        self.state = -1*np.ones(int(2*(self.n_holes-2))) #2(n-2) encoding
        # self.state = -1*np.ones(int(self.n_holes)) 
        self.guess_counter = 0
        return self.state
    
    def move_fox(self):
        max_hole_nr = self.n_holes-1
        hole_nr = self.hole_nr
        if hole_nr < max_hole_nr and hole_nr > 0:
            coin = np.random.random(1)

            if coin < 0.5:
                self.hole_nr -= 1
            else:
                self.hole_nr += 1
            
        elif hole_nr == max_hole_nr:
            self.hole_nr -= 1
        
        else:
            self.hole_nr += 1
        

    def step(self, action):
        """"
        Takes the agents guess as to where the fox is. If the guess is wrong, the fox moves to the next hole.

        Input:
            actions (int): the number of the hole to check for the fox

        Returns:
            observation (np.array): numpy array with the observation according to the encoding (0 for no fox, 1 for maybe fox, 2 for fox)
            reward (int): the reward for the guess

        """

        if action == self.hole_nr:
            reward = 1
            self.state[self.guess_counter]=action
            done = True
        
        elif self.guess_counter == 2*self.n_holes-5:
            reward = -1
            self.state[self.guess_counter]=action
            done = True
        else:
            reward = 0
            self.state[self.guess_counter]=action
            done = False

            self.move_fox()
        
        self.guess_counter+=1


        return self.state, reward, done, {}
    
    def unbounded_step(self, action):
        if action == self.hole_nr:
            reward = 1
            done = True
        
        else:
            reward = -1
            done = False
            self.move_fox()
        
        return self.state, reward, done, {}
    
class QFIAHv1():
    """
    Class for a quantum version of FoxInAHole, supporting the superposition of the fox across holes.
    """
    def __init__(self, n_holes, len_state, max_steps=None, prob_1= 0.5, prob_2= 0.5):
        split_1 = prob_1**(0.5)
        split_2 = prob_2**(0.5)
        self.len_state = len_state
        self.n_holes= n_holes
        if max_steps ==None:
            self.max_steps = 2*(n_holes-2)
        else:
            self.max_steps = max_steps
        # if n_holes ==5:
        #     self.T_odd = np.array([[0, split_1, 0, 0, 0],
        #                         [1, 0, split_1, 0, 0],
        #                         [0, split_2, 0, split_1, 0],
        #                         [0, 0, split_2, 0, 1],
        #                         [0, 0, 0, split_2, 0]])
            
        #     self.T_even = np.array([[0, split_1, 0, 0, 0],
        #                         [-1, 0, split_1, 0, 0],
        #                         [0, -1*split_2, 0, split_1, 0],
        #                         [0, 0, -1*split_2, 0, 1],
        #                         [0, 0, 0, -1*split_2, 0]])
        
       
        T_odd = np.zeros(shape= (n_holes, n_holes))
        T_even = np.zeros(shape= (n_holes, n_holes))

        for i in range(n_holes -2):
            T_odd[i,i+1]= split_1
            T_odd[i+2, i+1]= split_2

            T_even[i, i+1]= split_2
            T_even[i+2, i+1]= -1*split_1
        
        T_odd[1,0]=1
        T_odd[-2,-1]= 1 
        T_even[1,0]=-1
        T_even[-2,-1]= 1

        self.T_odd = T_odd
        self.T_even = T_even        

    def reset(self):

        initial_hole = np.random.randint(0,5,1)
        self.fox_state= np.zeros(self.n_holes)
        self.fox_state[initial_hole]=1
        self.game_state = np.ones(self.len_state)*-1
        self.move_counter = 0


        return self.game_state

    def move_fox(self):
        fox_state =self.fox_state
        if self.move_counter%2==0:
            fox_state= np.matmul(self.T_even, fox_state)
        else:
            fox_state= np.matmul(self.T_odd, fox_state)
        
        norm = np.linalg.norm(fox_state)
        self.fox_state = fox_state/norm
        
        self.move_counter+=1
        
        
    def step(self, action):
        game_state = np.roll(self.game_state, 1)
        game_state[0]=action
        self.game_state = game_state
        prob_correct = self.fox_state[action]**2
        coin = np.random.random(1)

        if prob_correct>coin:
            reward = 1
            done = True
        
        else:
            reward = -1
            if self.move_counter== self.max_steps-1:
                done = True
            else:
                done = False
                fox_state = self.fox_state
                fox_state[action]= 0
                norm = np.linalg.norm(fox_state)
                self.fox_state= fox_state/norm
                self.move_fox()

        return self.game_state, reward, done, {}
    

class QFIAHv2():
    """
    Class for a quantum version of FoxInAHole, supporting the superposition of the fox across holes and 'tunneling'.
    """
    def __init__(self, n_holes, len_state, max_steps=None, prob_1= 0.5, prob_2= 0.5, tunneling_prob=0.2):
        split_1 = (prob_1*(1-tunneling_prob))**(0.5)
        split_2 = (prob_2*(1-tunneling_prob))**(0.5)
        tunnel_split_1 = (prob_1*tunneling_prob)**(0.5)
        tunnel_split_2 = (prob_2*tunneling_prob)**(0.5)
        self.len_state = len_state
        self.n_holes= n_holes
        if max_steps ==None:
            self.max_steps = 2*(n_holes-2)
        else:
            self.max_steps = max_steps

        T_odd = np.zeros(shape= (n_holes, n_holes))
        T_even = np.zeros(shape= (n_holes, n_holes))

        for i in range(n_holes -2):
            T_odd[i,i+1]= split_1
            T_odd[i+2, i+1]= split_2
            T_even[i, i+1]= split_2
            T_even[i+2, i+1]= -1*split_1

            try:
                T_odd[i,i+2]= tunnel_split_1
            except:
                pass
            try:
                T_odd[i+2, i]= tunnel_split_2
            except:
                pass

            try:
                T_even[i, i+2]= tunnel_split_2
            except:
                pass

            try:
                T_even[i+2, i]= -1*tunnel_split_1
            
            except:
                pass
        
        T_odd[-1,-2] = prob_2**(0.5)
        T_odd[0,1] = prob_1**(0.5)

        T_even[-1,-2] = -prob_1**(0.5)
        T_even[0,1] = prob_2**(0.5)
      
        # Hier nog tunneling aanpassen!

        T_odd[1,0]=(1-tunneling_prob)
        T_odd[2,0]= ((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_odd[-2,-1]= (1-tunneling_prob)
        T_odd[-3,-1]= ((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_even[1,0]=-(1-tunneling_prob)
        T_even[2,0]= -((1-tunneling_prob)*tunneling_prob)**(0.5)
        T_even[-2,-1]= (1-tunneling_prob)
        T_even[-3,-1]= ((1-tunneling_prob)*tunneling_prob)**(0.5)

        T_odd[-1,0]= tunneling_prob**(0.5)
        T_odd[0,-1]= tunneling_prob**(0.5)

        T_even[-1,0]= tunneling_prob**(0.5)
        T_even[0,-1]= -tunneling_prob**(0.5)

        self.T_odd = T_odd
        self.T_even = T_even        
        

    def reset(self):

        initial_hole = np.random.randint(0,5,1)
        self.fox_state= np.zeros(self.n_holes)
        self.fox_state[initial_hole]=1
        self.game_state = np.ones(self.len_state)*-1
        self.move_counter = 0


        return self.game_state

    def move_fox(self):
        fox_state =self.fox_state
        if self.move_counter%2==0:
            fox_state= np.matmul(self.T_even, fox_state)
        else:
            fox_state= np.matmul(self.T_odd, fox_state)
        
        norm = np.linalg.norm(fox_state)
        self.fox_state = fox_state/norm
        
        self.move_counter+=1
        
        
    def step(self, action):
        game_state = np.roll(self.game_state, 1)
        game_state[0]=action
        self.game_state = game_state
        prob_correct = self.fox_state[action]**2
        coin = np.random.random(1)

        if prob_correct>coin:
            reward = 1
            done = True
        
        else:
            reward = -1
            if self.move_counter== self.max_steps-1:
                done = True
            else:
                done = False
                fox_state = self.fox_state
                fox_state[action]= 0
                norm = np.linalg.norm(fox_state)
                self.fox_state= fox_state/norm
                self.move_fox()

        return self.game_state, reward, done, {}
