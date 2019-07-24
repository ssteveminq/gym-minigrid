from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add
import numpy as np

class ActivePerceptionEnv(MiniGridEnv):
    """
    Single-room square grid environment with moving obstacles
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            n_obstacles=4
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # self.initial_sets=[]
        # self.initial_sets.append([])


        # Reduce obstacles if there are too many
        if n_obstacles <= size/2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size/2)
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 2)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        self.obs_initpos= []
        
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            init_pos =self.place_obj(self.obstacles[i_obst], max_tries=100)
            self.obs_initpos.append(init_pos)

        self.mission = "get all the objects in field of view"

    def getdist_coefficient(self, dist_input):
        min_depth =0.5
        min_slope =10
        max_slope =0.8
        max_depth =6.0

        Dc= 1/(1+np.exp(-min_slope*(dist_input-min_depth)))*(1/(1+np.exp(max_slope*(dist_input-max_depth))))
        return Dc

    def ac_reward(self):
        obs_dist_sum=0.0
        r_coeff = 1.0

        for i_obst in range(len(self.obstacles)):
            obs_pos = self.obstacles[i_obst].cur_pos
            obs_dist= np.linalg.norm(self.agent_pos-obs_pos)
            obs_coeff = self.getdist_coefficient(obs_dist)
            r_coeff *=obs_coeff

            # obs_dist_sum+=obs_dist

        # maxdist= self.max_distance()
        # print("r_coeff"+str(r_coeff))
        # reward = coefficient times maxreward(2)
        ac_reward = r_coeff*2.0
        return ac_reward 
        
        # return 2 - 0.9 * (self.step_count / self.max_steps)

    def max_distance(self):

        print(self)
        # half_height= self.grid_size/2
        # half_weight = self.grid_size/2
        half_height= self.height/2
        half_width= self.width/2
        print(half_height)
        print(half_width)
        maxdist = np.sqrt(half_height**2+half_width**2)
        print("max_distance: "+ str(maxdist))



    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'goal'

        obs, reward, done, info = MiniGridEnv.step(self, action)
        # reward=

        # If the agent tries to walk over an obstacle
        if action == self.actions.forward and not_clear:
            reward = -1
            done = True
            # print("collision with objects")
            return obs, reward, done, info

        # If the agent can see both objects in FOV of agentto walk over an obstacle

        # Update obstacle positions
        ObsinFOV=True
        num_obs = len(self.obstacles)


        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            # print("object i:",i_obst)
            # obs_pos=self.place_obj_trajectory(self.obstacles[i_obst], self.obstacles[i_obst].cur_idx,'Circle', top=self.obs_initpos[i_obst], size=(4,4), max_tries=100)
            # self.grid.set(*old_pos, None)

            try:
                # print("trajectory")
                # obs_pos=self.place_obj_trajectory(self.obstacles[i_obst], self.obstacles[i_obst].cur_idx,'Circle', top=top, size=(2,2), max_tries=100)
                # if num_obs<4:
                    # obs_pos=self.place_obj_trajectory(self.obstacles[i_obst], self.obstacles[i_obst].cur_idx,'Circle', top=top, size=(2,2), max_tries=100)
                # else:
                    # obs_pos=self.place_obj_trajectory(self.obstacles[i_obst], self.obstacles[i_obst].cur_idx,'Circle', top=top, size=(4,4), max_tries=100)
                # obs_pos=self.place_obj(self.obstacles[i_obst], top=self.obs_initpos[i_obst], size=(4,4), max_tries=100)
                obs_pos=self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
                self.grid.set(*old_pos, None)
                bview = self.in_view(obs_pos[0], obs_pos[1])

                if bview == False:
                    ObsinFOV = False

            except:
                pass
            
            # bview = self.in_view(obs_pos[0], obs_pos[1])
            # if bview == False:
               # ObsinFOV = False

        if ObsinFOV ==True:
            done = True
            reward =2 
            # reward =self.ac_reward() 
        # acreward=self.ac_reward()

        return obs, reward, done, info

class ActivePerceptionEnv5x5(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=5, n_obstacles=1)

class ActivePerceptionRandomEnv5x5(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None, n_obstacles=1)

class ActivePerceptionEnv6x6(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=3)

class ActivePerceptionRandomEnv6x6(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_obstacles=1)

class ActivePerceptionEnv8x8(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None, n_obstacles=3)

class ActivePerceptionEnv12x12(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=12, agent_start_pos=None, n_obstacles=2)

class ActivePerceptionEnv12x12x4(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=12, agent_start_pos=None, n_obstacles=4)


class ActivePerceptionEnv16x16(ActivePerceptionEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=3)

register(
    id='MiniGrid-Active-Perception-5x5-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv5x5'
)

register(
    id='MiniGrid-Active-Perception-Random-5x5-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionRandomEnv5x5'
)

register(
    id='MiniGrid-Active-Perception-6x6-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv6x6'
)

register(
    id='MiniGrid-Active-Perception-Random-6x6-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionRandomEnv6x6'
)

register(
    id='MiniGrid-Active-Perception-8x8-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv8x8'
)

register(
    id='MiniGrid-Active-Perception-12x12-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv12x12'
)

register(
    id='MiniGrid-Active-Perception-12x12x4-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv12x12x4'
)


register(
    id='MiniGrid-Active-Perception-16x16-v0',
    entry_point='gym_minigrid.envs:ActivePerceptionEnv16x16'
)
