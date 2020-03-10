from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np


class PointEnv(Env):
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._state = self._state + action
        x, y = self._state
        reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        print("reward:", reward)
        return Step(observation=next_observation, reward=reward, done=done)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    def render(self):
        print('current state:', self._state)

