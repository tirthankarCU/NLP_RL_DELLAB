import gym
from gym import spaces
import pygame
import numpy as np
from enum import Enum 
import vision_pyGame as vga
'''
ACTIONS
'''
class ACTION(Enum):
    PICK_BIG=1
    PICK_MED=2
    PICK_SMALL=3
    PUT_BIG=4
    PUT_MED=5
    PUT_SMALL=6

'''
BOX TYPE
'''
class BOXTYPE(Enum):
    NONE=0
    BIG=1
    MEDIUM=2
    SMALL=3

class RlNlpWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
############################################
    def __init__(self,mx_timeSteps=50,render_mode=None):
        self.no=np.random.randint(0,1000)[0]
        self.carry=False
        self.boxType=BOXTYPE.NONE
        self.mode=0
        if render_mode=='rgb_array':
            self.mode=1
        self._visual=vga.draw_main(self.metadata['render_modes'][self.mode],self.metadata['render_fps'],self.no)
        self._text='TBD'
        self._question='TBD'
        self.mx_timeSteps,self.curr_time=mx_timeSteps,0
############################################
    def _get_obs(self):
        return {"text": self._text,"question":self._quesiton,"visual": self._visual}

    def _get_info(self):
        return {
            "progress": "currently feature not required."
        }
############################################
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
############################################
    def step(self, action):

        def pick(boxArr,b_type):
            if self.carry==False:
                for box in boxArr:
                    if not box.isEmpty:
                        self.carry=True 
                        self.boxType=b_type
                        box.isEmpty=False
                        return 0 
            else:
                return -1
            
        def put(b_type):
            if self.carry==False or self.boxType!=b_type:
                return -1
            self.boxType=BOXTYPE.NONE 
            self.carry=False 
            for box in vga.constructArrElement[b_type.value-1]:
                if box.isEmpty:
                    box.isEmpty=False 
                    return 0 
                
        def checkSolution():
            result,power=0,100
            for c in vga.constructArrElement:
                cnt_box=0
                for box in c:
                    if not box.isEmpty:
                        cnt_box+=1
                result+=cnt_box*power 
                power/=10
            return True if self.no==result else False 
        
        if action==ACTION.PICK_BIG:
            reward=pick(vga.big_block,BOXTYPE.BIG)
        elif action==ACTION.PICK_MED:
            reward=pick(vga.medium_block,BOXTYPE.MEDIUM)
        elif action==ACTION.PICK_SMALL:
            reward=pick(vga.small_block,BOXTYPE.SMALL)
        elif action==ACTION.PUT_BIG:
            reward=put(BOXTYPE.BIG)
        elif action==ACTION.PUT_MED:
            reward=put(BOXTYPE.MEDIUM)
        elif action==ACTION.PUT_SMALL:
            reward=put(BOXTYPE.SMALL)

        self._visual=vga.drawAgain()
        self.curr_time+=1
        terminated = True if self.curr_time>self.mx_timeSteps else checkSolution()
        if terminated:
            sign=1 if checkSolution() else -1
        reward = sign*10 if terminated else reward
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
############################################
    def render(self):
        pass 
############################################
    def _render_frame(self):
        pass
############################################
    def close(self):
        pass
############################################