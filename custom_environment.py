import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque

class CustomEnvironment(object):
    """
    Small wrapper for gym Custom environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state
    is constructed.
    """
    def __init__(self, gym_env, input_size, agent_history_length, extra_args=None):
        self.env = gym_env
        print('TODO initialization with extra_args:{}'.format(extra_args))
        if extra_args.get('init_with_args') is not None:
            self.env.init_with_args(**extra_args)
            print('DONE add extra_args:{} to init env'.format(extra_args))
        else:
            print('No need to add extra_args')

        self.input_size = input_size
        _work = [i for i in range(2, input_size) if input_size % i == 0]
        self.resized_width = _work[-2:][0]
        print('self.resized_width: {}'.format(self.resized_width))
        self.resized_height = int(input_size / self.resized_width)
        print('self.resized_height: {}'.format(self.resized_height))
        self.agent_history_length = agent_history_length

        self.gym_actions = range(gym_env.action_space.n)
        # if (gym_env.spec.id == "isc_poo-v001"):
        #     print("Doing workaround for pong or breakout")
        #     # Gym returns 6 possible actions for breakout and pong.
        #     # Only three are used, the rest are no-ops. This just lets us
        #     # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
        #     self.gym_actions = [1,2,3]

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the Custom game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)
        
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        # return resize(rgb2gray(observation), (self.resized_width, self.resized_height))
        return observation

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        # s_t1 = np.empty((self.agent_history_length, self.resized_height, self.resized_width))
        s_t1 = np.empty((self.agent_history_length, self.input_size))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info
