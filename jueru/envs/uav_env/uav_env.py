import sys
import time

import cv2
import numpy as np
import gym
import pygame
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from pygame import event
from pygame.constants import K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7
from PIL import Image

from jueru.envs.uav_env.Entities import Obstacle, Target, Agent


class Uav_env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    def __init__(self, world_size, step_size, obstacle_num, obs_size, max_step_num, display=False, fixed=False):
        super(Uav_env, self).__init__()

        # Size of the 1D-grid
        self.world_size = world_size
        self.step_size = step_size  # time, in unit of second
        self.max_step_num = max_step_num
        self._max_episode_steps = max_step_num
        self.display = display
        self.fixed = fixed

        # for obstacles
        self.obstacle = Obstacle(world_size=self.world_size, number_obstacle=obstacle_num)
        self.obstacle_num = obstacle_num
        self.obst_shape = self.obstacle.shape_list
        self.obst_pos = self.obstacle.pos_list
        self.obst_size = self.obstacle.size_list  # ([a,b], [a,b],[a,b],[a,b],[a,b],)
        self.obstacle_num_init = self.obstacle_num

        self.obs_size = obs_size
        # for target
        self.target = Target(world_size=self.world_size, number_target=1)
        self.target_num = 1
        self.target_size = self.target.size_list
        self.target_pos = self.target.pos_list
        self.target_num_init = 1

        # for agent
        self.agent = Agent()
        self.agent.reset()
        self.agent_pos = np.round(self.agent.pose)
        self.agent_size = self.agent.size

        self.count = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_list = [[1, 1], [-1, -1], [0, 0], [-1, 1], [1, -1]]
        self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=255,
                                                                  shape=(1, 22, 22), dtype=np.uint8),
                                              'target': spaces.Box(low=0, high=1,
                                                                   shape=(4,), dtype=np.float32)
                                              })

        self.pygame_init()

    def reset_obstacle(self):
        # word_size change to self.world_size - 2* obs_size
        self.obstacle = Obstacle(world_size=self.world_size, number_obstacle=self.obstacle_num)
        self.obst_shape = self.obstacle.shape_list
        # print('shape', self.obst_shape)
        self.obst_pos = self.obstacle.pos_list
        # print('pos',self.obst_pos)
        self.obst_size = self.obstacle.size_list  # ([a,b], [a,b],[a,b],[a,b],[a,b],)
        # print('size', self.obst_size)

    def reset_target(self):
        # the same as pre
        self.target = Target(world_size=self.world_size, number_target=1)
        self.target_num = self.target_num_init
        self.target_size = self.target.size_list
        self.target_pos = self.target.pos_list
        # print('pos',self.trash_pos)

    def reset_agent(self):
        self.agent.reset()
        self.agent_pos = self.agent.pose

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize variable
        self.count = 0

        self.map_done = []

        self.reset_obstacle()
        self.reset_target()
        self.reset_agent()

        # Initialize target and agent location
        self.init_state()
        obs = self._get_obs()
        self.render_rest()
        # if self.render_mode:
        #     self.pygame_init()
        return obs

    def step(self, action):
        info = {}
        self.count = self.count + 1
        # print(self.agent_pos)

        #print(action)
        reward = self.excute_action(action)

        # reward = self.get_reward(action)
        # if reward > 0:
        #     info = 1
        # else:
        #     info = -1

        done = self.get_done()

        obs = self._get_obs()
        #print(reward)
        return obs, reward, done, info

    def render(self, mode='console'):
        reward = 0
        # draw picture on screen
        self.screen.fill(self.back_ground_color)  # Fill color

        agent = Block(color=self.agent_color, width=0, height=0,
                      pos=(self.agent_pos[:2] - 20), size=self.agent_size, shape='circle')
        agent_list = pygame.sprite.Group()
        agent_list.add(agent)

        self.target_move()

        for i in range(self.target_num):
            # print(i)
            if pygame.sprite.collide_mask(agent, self.target_list.sprites()[i]):
                self.target_list.remove(self.target_list.sprites()[i])
                self.trash_num = self.trash_num - 1
                reward += 2
                break
        for j in range(self.obstacle_num):
            if pygame.sprite.collide_mask(agent, self.obstacle_list.sprites()[j]):
                reward -= 0.05
                self.agent_pos = self.agent_buf
                break
        # print(self.agent_pos)
        if not (any((self.agent_pos == x).all() for x in self.map_done)):
            self.map_done.append(np.array(self.agent_pos))
            # print(self.map_done)
            # reward += 0.01
        for i in self.map_done:
            pygame.draw.rect(self.screen, (90, 150, 90), (i[0], i[1], 3, 3))
        agent_list.draw(self.screen)
        self.target_list.draw(self.screen)
        self.obstacle_list.draw(self.screen)
        if self.display:
            pygame.display.update()  # Update all display

            time.sleep(0.1)
        agent_pos_array = np.array(self.agent_pos)
        target_array = np.array(self.target_pos[0])
        #print('a', agent_pos_array)
        #print('t', target_array)
        dis = np.sqrt(np.dot((agent_pos_array - target_array).T, (agent_pos_array - target_array)))
        #print(dis*1e-5)
        reward -= dis*3e-5
        #print(reward)
        return reward

    def close(self):
        pass

    def init_state(self):

        self.trash_num = self.target_num_init
        self.obstacle_num = self.obstacle_num_init
        self.reset_agent()

    def Capture(self, display, name, pos, size):  # (pygame Surface, String, tuple, tuple)
        image = pygame.Surface(size)  # Create image surface
        image.blit(display, (0, 0), ((pos[0]-size[0]/2,pos[1]-size[1]/2), size))  # Blit portion of the display to the image

        img_array = pygame.surfarray.array3d(image).transpose(1, 0, 2)
        save_im = Image.fromarray(np.uint8(img_array)).convert('L')
        #save_im.save('obs.png')
        im = Image.fromarray(np.uint8(img_array)).convert('L').resize((22, 22))
        im_array = np.array(im).reshape((22, 22, 1)).transpose(2, 0, 1)
        # print(im_array.shape)
        #im.save('obs.png')

        return im_array

    def _get_obs(self):

        im = self.Capture(display=self.screen, name='1', pos=self.agent_pos[:2], size=[self.obs_size, self.obs_size])

        # float_target = []
        # for i in self.target_pos[0]:
        #     float_target.append(float(i))
        target_array = np.array(self.target_pos[0])
        agent_pos_array = np.array(self.agent_pos)
        #print(target_array)
        #print(agent_pos_array)
        observation_dict = {'image': im,
                            'target': np.float32(np.concatenate([target_array/self.world_size, agent_pos_array/self.world_size]))}
        # print(im.shape)
        return observation_dict

    def excute_action(self, action):

        reward = 0

        self.agent_buf = self.agent_pos.copy()

        #     self.agent_pos[:2] = self.agent_pos[:2] + np.array([1,0])*8
        # elif action == 1:
        #     self.agent_pos[:2] = self.agent_pos[:2] + np.array([-1, 0])*8
        # elif action == 2:
        #     self.agent_pos[:2] = self.agent_pos[:2] + np.array([0, 1])*8
        # elif action == 3:
        #     self.agent_pos[:2] = self.agent_pos[:2] + np.array([0, -1])*8
        # # elif action == 4:
        #     self.agent_pos[:2] = self.agent_pos[:2] + np.array([0, 0])*5
        self.agent_pos = self.agent_pos + self.step_size * np.array([np.cos(action * np.pi), np.sin(action * np.pi)])[:,
                                                           0]
        # change robot pos if robot collide with the bound
        #print(self.agent_pos)
        if self.agent_pos[0] > (self.world_size - 1) or self.agent_pos[1] > (self.world_size - 1) or self.agent_pos[
            0] < 0 or (
                self.agent_pos[1] < 0):
            self.agent_pos = np.clip(self.agent_pos, 0, (self.world_size - 1))
            reward = reward - 0.05
        # print(f'current agent pos :{self.agent_pos}')
        reward += self.render()
        return reward

    def get_done(self):
        done = False
        if self.trash_num <= 0:
            done = True
        if self.count > self.max_step_num:
            done = True
        return done

    def pygame_init(self):
        pygame.init()
        size = width, height = self.world_size, self.world_size
        self.screen = pygame.display.set_mode(size)
        self.back_ground_color = (255, 255, 255)
        self.target_color = (0, 255, 0)
        self.obstacle_color = (255, 0, 0)
        self.agent_color = (0, 0, 255)

        self.clock = pygame.time.Clock()

        self.target_list = pygame.sprite.Group()
        for i in range(self.target_num):
            target_ = Block(color=self.target_color, width=self.target_size[i], height=self.target_size[i],
                            pos=self.target_pos[i], size=self.target_size[i], shape='circle')
            self.target_list.add(target_)

        self.obstacle_list = pygame.sprite.Group()

        for i in range(self.obstacle_num):
            obstacle_ = Block(color=self.obstacle_color,
                              width=self.obst_size[i], height=self.obst_size[i],
                              pos=self.obst_pos[i], size=self.obst_size[i], shape=self.obst_shape[i])
            self.obstacle_list.add(obstacle_)

    def render_rest(self):
        self.target_list = pygame.sprite.Group()
        for i in range(self.trash_num):
            target_ = Block(color=self.target_color, width=self.target_size[i], height=self.target_size[i],
                            pos=self.target_pos[i], size=self.target_size[i], shape='circle')
            self.target_list.add(target_)

        self.obstacle_list = pygame.sprite.Group()

        for i in range(self.obstacle_num):
            obstacle_ = Block(color=self.obstacle_color,
                              width=self.obst_size[i], height=self.obst_size[i],
                              pos=self.obst_pos[i], size=self.obst_size[i], shape=self.obst_shape[i])
            self.obstacle_list.add(obstacle_)
        return

    def record_video(self):
        # 。。。。。。。pygame表示各种物体代码块
        if self.count == 0:
            fps = 12
            size = (self.world_size, self.world_size)
            file_path = "./record_video" + str(int(time.time())) + ".avi"  # 导出路径
            print(file_path)
            fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
            self.video = cv2.VideoWriter(file_path, fourcc, fps, size)
        else:
            imagestring = pygame.image.tostring(self.screen, "RGB")
            pilImage = Image.frombytes("RGB", (self.world_size, self.world_size), imagestring)
            img = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)

            self.video.write(img)  # 把图片写进视频
            if self.count == self.max_step_num:
                self.video.release()  # 释放
        return

    def target_move(self):
        target_buf = (self.target_list.sprites()[0].rect.x, self.target_list.sprites()[0].rect.y)
        self.target_list.sprites()[0].rect.x += np.random.randint(low=-3, high=3)
        self.target_list.sprites()[0].rect.y += np.random.randint(low=-3, high=3)
        if self.target_list.sprites()[0].rect.x > (self.world_size - 1) or self.target_list.sprites()[0].rect.y > (
                self.world_size - 1) or self.target_list.sprites()[0].rect.x < 0 or (
                self.target_list.sprites()[0].rect.y < 0):
            self.target_list.sprites()[0].rect.x = np.clip(self.target_list.sprites()[0].rect.x, 0,
                                                           (self.world_size - 1))
            self.target_list.sprites()[0].rect.y = np.clip(self.target_list.sprites()[0].rect.y, 0,
                                                           (self.world_size - 1))
        self.target_pos[0] = [self.target_list.sprites()[0].rect.x, self.target_list.sprites()[0].rect.y]
        return


class Block(pygame.sprite.Sprite):
    def __init__(self, color, width, height, pos, size, shape):
        super().__init__()
        self.image = pygame.Surface([40, 40]).convert_alpha()
        self.image.set_colorkey((0, 0, 0))  # 设置透明色
        # pygame.draw.rect(self.image, color, (0,0,width,height))
        # print('pp',pos[0],pos[1])
        if shape == 'circle':
            pygame.draw.circle(self.image, color, (20, 20), size)
        else:
            pygame.draw.rect(self.image, color, (0, 0, size[0], size[1]))
        self.rect = self.image.get_rect()
        self.rect.x = pos[0]
        self.rect.y = pos[1]
        self.mask = pygame.mask.from_surface(self.image)


if __name__ == '__main__':

    env = Uav_env(world_size=240, step_size=20, obstacle_num=5, max_step_num=100, display=True, fixed=False, obs_size=66)
    check_env(env)
    obs = env.reset()
    #env.record_video()
    episode_reward = 0
    while True:
        # action = env.action_space.sample()
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN):
                if event.key == K_1:
                    action = 0
                elif event.key == K_2:
                    action = 0.2
                elif event.key == K_3:
                    action = -0.2
                elif event.key == K_4:
                    action = 0.5
                elif event.key == K_5:
                    action = -0.5
                elif event.key == K_6:
                    action = 0.7
                elif event.key == K_7:
                    action = -0.7
                # action = input("input:")
                # print(action)
                action = np.array([action])
                obs, reward, done, info = env.step(action)
                # env.record_video()
                # env.render()
                # print(action)
                episode_reward += reward
                if done:
                    print("Reward:", episode_reward)
                    episode_reward = 0.0
                    obs = env.reset()

    # a = pygame.image.tostring(env.screen, "RGB")
    # print(a)
    # while True:
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     #env.render()
    #     #print(action)
    #     episode_reward += reward
    #     if done:
    #         print("Reward:", episode_reward)
    #         episode_reward = 0.0
    #         obs = env.reset()
    #         # break

    ## rendering env step by step.
    # while True:  # An infinite loop ensures that the window is always displayed
    #     env.clock.tick(60)  # It is executed 60 times per second
    #     for event in pygame.event.get():  # Iterate over all events
    #         if event.type == pygame.QUIT:  # If you click to close the window, it is retired
    #             sys.exit()
    #         try:
    #             if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN):
    #                 action = env.action_space.sample()
    #                 obs, reward, done, info = env.step(action)
    #                 env.render()
    #                 print(action)
    #                 episode_reward += reward
    #                 if done:
    #                     print("Reward:", episode_reward)
    #                     episode_reward = 0.0
    #                     obs = env.reset()
    #                     #sys.exit()
    #         except:
    #             pass
