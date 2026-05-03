import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from suika.part2.config import config, CollisionTypes
    from suika.part2.cloud import Cloud
    from suika.part2.wall import Wall
    from suika.part2.particle import Particle
    from suika.part2.collision import collide
    from suika.part2.text import score as draw_score
    from suika.part2.text import gameover as draw_gameover
except ImportError as e:
    raise ImportError(f"Could not import game modules. Make sure you are running from the project root or have set PYTHONPATH correctly. Error: {e}")

MAX_TYPE = 11.0 
MAX_RADIUS = 150.0

class SuikaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.screen.fps}
    
    def __init__(self, render_mode=None, action_type="discrete", discrete_bins=448, max_fruits=50, spatial_features=False, debug=False, clustering=False):
        """
        Here I explain the default initialization parameters for the gymnasium environment
        render_mode = None - I do not want to render while training because it's a massive slow down
        action_type = "discrete" - Technically the game implementation only accepts integer value x-coordinates within the playable space that's 448 units wide so I use a discrete action type/space
        discrete_bins = 448 - The aformentioned size of the discrete action space to configure
        max_fruits = 50 - The observation space (game board representation) needs to be a fixed size/shape through a training run. However, the number of fruits on the board changes over time. 
          Each fruit is represented by 4 values in the observation space (discussed below) so there are always at least 4*max_fruits indexes in the observation space array. These indexes are zeroes
          until a fruit needs to be represented in those indexes. It's highly unlike for there to ever be 50 or more fruit on the board so this is a safe value to pick
        spatial_features = False - This turns on calculations for spatial features and adds them to the observation space. Defaulted to False because they can be computationally expensive which slows down training
        debug = False - My flag for toggling debug prints
        cluster = False - A separate flag just for clustering since it's the most computationally expensive feature to calculate (determined by profiling the program) and has a signficant training speed impact
        """
        self.render_mode = render_mode
        self.action_type = action_type
        self.discrete_bins = discrete_bins
        self.max_fruits = max_fruits
        self.spatial_features = spatial_features
        self.debug = debug
        self.clustering = clustering

        # Print useful info when training starts
        # print(f"ACTION_TYPE: {action_type}")
        # if action_type == "discrete":
        #     print(f"DISCRETE_BINS: {discrete_bins}")
        # print(f"SPATIAL_FEATURES: {spatial_features}")
        # print(f"CLUSTERING_CALCULATION: {clustering}")

        self.last_action = None
        self.repeat_count = 0
        
        pygame.init()
        pygame.display.init()
        
        self.screen_width = config.screen.width
        self.screen_height = config.screen.height
        
        # Human render mode is for rendering the game on screen
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Suika RL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()

        # I always use discrete for reasons above. This sets gymnasium's action_space - the range of action values the model should pick from when predict()-ing the optimal action given a state
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(self.discrete_bins)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_len = 0
        # With spatial features and/or clustering enabled, I have to adjust the observation shape accordingly so that 
        # gymnasium receives the expected observation space shape and doesn't crash
        if self.spatial_features and self.clustering:
            # changed from 9 + to 14 + to adjust for spatial features
            obs_len = 14 + (self.max_fruits * 4)
        elif self.spatial_features and not self.clustering:
            obs_len = 13 + (self.max_fruits * 4)
        else:
            obs_len = 9 + (self.max_fruits * 4)

        self.debug_print(f"OBS_LEN: {obs_len}")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        self.space = None
        self.walls = None
        self.cloud = None
        self.handler = None
        self.game_over = False
        self.game_over_timer = 0
        self.game_over_threshold = 3.0
    
    def render(self):
        """This was a workaround to get rendering to work with SBX (Stable-Baselines3 with JAX support) when evaluating a trained model"""
        # If you are already rendering in step(), this can be empty
        # or you can move your Pygame/drawing logic here.
        if self.render_mode is None:
            return
        
        # Example: if using pygame
        # self._draw_frame() 
        return
    
    def _normalize(self, val, max_val):
        return val / max_val

    def _get_obs(self):
        """
        This returns the current observation_space of the game. The output of this function is what DQN takes in as the `state` which it uses to provide
        the most optimal action to take using the Q-value function.
        """
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        W = float(self.screen_width)
        H = float(self.screen_height)
        
        # The current fruit ID available to be dropped and its size. Normalized between 0-1
        obs[0] = self.cloud.curr.n / MAX_TYPE
        obs[1] = self.cloud.curr.radius / MAX_RADIUS
        
        # The next fruit ID available to be dropped and its size. Normalized between 0-1
        obs[2] = self.cloud.next.n / MAX_TYPE
        obs[3] = self.cloud.next.radius / MAX_RADIUS
        
        # The left, right, and bottom bounds of the play are as well as the height at which the game ends if fruit collide at or above that height
        # All normalized to be between 0-1
        obs[4] = config.pad.left / W
        obs[5] = config.pad.right / W
        obs[6] = config.pad.bot / H 
        obs[7] = config.pad.killy / H
        self.debug_print(f"obs[7] = {config.pad.killy} / {H}")
        
        fruits = []
        min_y = H 
        
        # Calculate the highest position of a fruit on the board. I know the code says min. That's because the game is programmed
        # so that the top of the visual screen is height 0 and the y position increases as look lower at the screen
        for p in self.space.shapes:
            if isinstance(p, Particle) and p.alive:
                fruits.append(p)
                min_y = min(min_y, p.pos[1] - p.radius) # neater
        
        # This is essentially the same as height_severity/clearance_ratio
        # Normalized to screen height
        obs[8] = min_y / H 
        self.debug_print(f"obs[8] = {min_y} / {H} == {obs[8]}")
        self.debug_print(f"MIN_Y: {min_y}")

        idx = 0
        # Try adding spatial features directly into the observation space to help learning
        # This is ugly and not optimal but that's ok right now
        if self.spatial_features and self.clustering:
            spatial_features = self._calculate_spatial_features()
            obs[9] = spatial_features["height_severity"]
            obs[10] = spatial_features["clustering_score"]
            obs[11] = spatial_features["edge_occupation"]
            obs[12] = spatial_features["clearance_ratio"]
            obs[13] = spatial_features["avg_fruit_spread"]
            idx = 14
        elif self.spatial_features and not self.clustering:
            spatial_features = self._calculate_spatial_features()
            obs[9] = spatial_features["height_severity"]
            obs[10] = spatial_features["edge_occupation"]
            obs[11] = spatial_features["clearance_ratio"]
            obs[12] = spatial_features["avg_fruit_spread"]
            idx = 13
        else:
            idx = 9

        # Sort fruit to maintain ordering for the model
        fruits.sort(key=lambda p: (p.pos[1], p.pos[0]))
        
        fruit_count = 0
        for p in fruits:
            if fruit_count >= self.max_fruits:
                break
            obs[idx] = p.n / MAX_TYPE # The ID of a fruit on the board normalized by max type
            obs[idx+1] = p.pos[0] / W # The horizontal position of the fruit normalized by screen width
            obs[idx+2] = p.pos[1] / H # The vertical position of the fruit normalized by screen height
            obs[idx+3] = p.radius / MAX_RADIUS # The radius of the fruit normalized by max radius
            
            idx += 4
            fruit_count += 1

        return obs

    def _get_info(self):
        return {
            "score": self.handler.data["score"] if self.handler else 0,
            "game_over": self.game_over
        }

    def _calculate_spatial_features(self):
        """Calculate spatial features for reward shaping."""
        W = float(self.screen_width)
        H = float(self.screen_height)

        fruits = []
        for p in self.space.shapes:
            if isinstance(p, Particle) and p.alive:
                fruits.append(p)

        if not fruits:
            return {
                "max_height": 0,
                "height_severity": 0,
                "clustering_score": 0,
                "edge_occupation": 0,
                "clearance_ratio": 1.0,
                "avg_fruit_spread": 1.0
            }

        # HEIGHT PENALTY - how close is highest fruit to top?
        min_y = min(p.pos[1] - p.radius for p in fruits)
        height_severity = np.clip((config.pad.killy - min_y) / (config.pad.killy - config.pad.top), 0.0, 1.0)

        # CLUSTERING SCORE - reward similar fruits being close
        clustering_score = 0
        if self.clustering:
            clustering_score = self._calculate_clustering(fruits)

        # EDGE OCCUPATION - reward large fruits in corners
        edge_occupation = self._calculate_edge_occupation(fruits, W)

        # CLEARANCE RATIO - available vertical space at top
        clearance_ratio = self._calculate_clearance_ratio(min_y)

        # FRUIT SPREAD - penalize overly clustered center
        avg_fruit_spread = self._calculate_spread(fruits, W)

        return {
            "max_height": min_y,
            "height_severity": height_severity,
            "clustering_score": clustering_score,
            "edge_occupation": edge_occupation,
            "clearance_ratio": clearance_ratio,
            "avg_fruit_spread": avg_fruit_spread
        }

    def _calculate_clustering(self, fruits):
        """Reward similar-sized fruits being close to each other (encourages merging)."""
        if len(fruits) < 2:
            return 1.0

        clustering_reward = 0.0
        count = 0

        # TODO: might be able to optimize this calculation? Pretty compute intensize
        for i, f1 in enumerate(fruits):
            # Find other fruits of same or similar type (within 1 level)
            for f2 in fruits[i+1:]:
                if abs(f1.n - f2.n) <= 1:  # Same or adjacent type
                    dist = np.sqrt((f1.pos[0] - f2.pos[0])**2 + (f1.pos[1] - f2.pos[1])**2)
                    # Reward if close (within 2x combined radius)
                    combined_radius = f1.radius + f2.radius
                    proximity = max(0, 1.0 - (dist / (combined_radius * 2)))
                    clustering_reward += proximity
                    count += 1

        return clustering_reward / max(1, count)

    def _calculate_edge_occupation(self, fruits, W):
        """Reward large fruits being in corners/edges (keeps small fruits away)."""
        corner_zone = W * 0.15  # Corners are within 15% of width
        edge_reward = 0.0

        for p in fruits:
            # Check if fruit is in corner/edge zones
            in_left_corner = p.pos[0] < corner_zone
            in_right_corner = p.pos[0] > (W - corner_zone)
            is_in_edge = in_left_corner or in_right_corner

            # Reward proportional to fruit size if in edge
            if is_in_edge:
                size_factor = p.radius / 150.0  # normalized by max radius
                edge_reward += size_factor

        return edge_reward / max(1, len(fruits))

    def _calculate_clearance_ratio(self, min_y):
        """Ratio of distance of highest fruit to the screen top vs the distance of the kill zone to the screen top. Normalized/clipped between 0-1"""
        return np.clip((min_y - config.pad.top) / (config.pad.killy - config.pad.top), 0.0, 1.0)

    def _calculate_spread(self, fruits, W):
        """Calculate how well-distributed fruits are horizontally."""
        if len(fruits) < 2:
            return 1.0

        x_positions = [p.pos[0] for p in fruits]
        # Calculate coefficient of variation of x positions
        std_x = np.std(x_positions)
        spread_ratio = std_x / (W / 2) if W > 0 else 0  # Normalized spread
        return min(1.0, spread_ratio)  # Capped at 1.0

    def _is_simulation_at_rest(self, space, linear_threshold=0.1, angular_threshold=0.01):
        """
        Used to make sure all the fruit stop moving before letting the model make the next move. 
        This avoids issues with the previous implementation where the model would pick actions on fixed time intervals
        This would result in the model taking actions while fruit were still moving significantly, meaning the model
        was making decisions on a bad state because the observation space doesn't contain any information about the linear/angular velocity
        of fruits (only position, ID, and size) so the model might aim to hit a specific fruit that might not be there
        once the dropped fruit falls. This eliminates the need for the model to embed/learn too much physics in its model weights
        and makes each state -> action -> state' transition more consistent for training.
        """
        for body in space.bodies:
            if body.body_type != pymunk.Body.DYNAMIC:
                continue

            # Use squared length to avoid square root
            if body.velocity.get_length_sqrd() > (linear_threshold**2):
                return False

            if abs(body.angular_velocity) > angular_threshold:
                return False

        return True
    
    def debug_print(self, log_message):
        if self.debug and self.render_mode == "human":
            print(f"DEBUG: {log_message}")

    def reset(self, seed=None, options=None):
        """Resets the game state for a new training game"""
        super().reset(seed=seed)
        
        self.last_action = None
        self.repeat_count = 0

        self.space = pymunk.Space()
        self.space.gravity = (0, config.physics.gravity)
        self.space.damping = config.physics.damping
        self.space.collision_bias = config.physics.bias

        left = Wall(config.top_left, config.bot_left, self.space)
        bottom = Wall(config.bot_left, config.bot_right, self.space)
        right = Wall(config.bot_right, config.top_right, self.space)
        self.walls = [left, bottom, right]

        self.cloud = Cloud()
        
        # Randomly places a few fruit on the board instead of starting with an empty board
        # I kept this on because there's nothing for the model to learn from an empty board in my opinion
        do_random_start = True
        if options and "random_start" in options:
            do_random_start = options["random_start"]
            
        if do_random_start:
            rng = np.random.default_rng(seed)
            num_random = rng.integers(3, 9) 
            
            for _ in range(num_random):
                x_pos = rng.uniform(config.pad.left + 20, config.pad.right - 20)
                n_type = rng.integers(0, 6)
                
                p = Particle((x_pos, config.pad.top), n_type, self.space)
                
                for _ in range(30):
                    self.space.step(1/config.screen.fps)

        self.handler = self.space.add_collision_handler(CollisionTypes.PARTICLE, CollisionTypes.PARTICLE)
        self.handler.begin = collide
        self.handler.data["score"] = 0

        self.game_over = False
        self.game_over_timer = 0
        
        if self.render_mode == "human":
            self._draw_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        The most important function. It:
        1. Performs an action (fruit drop @location) chosen by the model for the current game state
        2. Simulates the fruit drop and collisions/merges until all fruits are at rest
        3. Calculates the reward to give the model for this action
        4. Calculates the new game state (observation_space) including any custom features and returns it to the training framework
            Also tells the training framework if the game is over
        """
        if self.game_over:
            return self._get_obs(), 0, True, False, self._get_info()

        act_val = 0.0
        if self.action_type == "discrete":
            bin_idx = action 
            act_val = -1.0 + (bin_idx / (self.discrete_bins - 1)) * 2.0
        else:
            act_val = np.clip(action[0], -1.0, 1.0)
        
        pad_width = config.pad.right - config.pad.left
        target_x = config.pad.left + (act_val + 1.0) * 0.5 * pad_width
        
        self.cloud.curr.set_x(round(target_x))
        
        # self.debug_print(f"act_val: {act_val}")
        # self.debug_print(f"target_x: {target_x}")
        # self.debug_print(f"int(target_x): {round(target_x)}\n")
            
        self.cloud.release(self.space)

        max_steps = 400

        reward = 0
        initial_score = self.handler.data["score"]

        # To prevent the game simulating too far ahead, set a maximum step limit.
        # This loop steps the game 1 frame at a time until all the fruit stop moving
        i = 0
        while i < max_steps:
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return self._get_obs(), 0, True, False, self._get_info()

            self.space.step(1/config.screen.fps)

            any_over = False
            for p in self.space.shapes:
                if isinstance(p, Particle):
                    if p.has_collided:
                         bottom_y = p.pos[1] + p.radius
                         if bottom_y < config.pad.killy:
                             any_over = True
                             break

            if any_over:
                self.game_over_timer += (1/config.screen.fps)
                if self.game_over_timer > self.game_over_threshold:
                    self.game_over = True
            else:
                self.game_over_timer = 0

            if self.game_over:
                break

            # Check if fruits have come to rest but give the fruit 90 frames to fall first
            # i % 30 to check less often because this is a costly check
            if i >= 90 and i % 30 == 0:
                if self._is_simulation_at_rest(self.space):
                    break

            if self.render_mode == "human":
                self._draw_frame(wait_val=1)
                self.clock.tick(config.screen.fps)

            i += 1

        # Advance cloud for next fruit after fruits settle
        self.cloud.step()

        # Final frame render if in human mode
        if self.render_mode == "human":
            self._draw_frame()

        final_score = self.handler.data["score"]

        # Scale the reward, might need to disable scaling when training an existing model that wasn't trained with scaling
        # step_reward = (final_score - initial_score)/100
        # step_reward = final_score - initial_score
        # maybe try scaling up the rewards by 50-100%
        step_reward = final_score - initial_score

        # BASE REWARD: Score + small bonus for surviving
        # scaled BASE reward
        self.debug_print(f"Reward: {reward:.6f}")
        # Was 0.25 before scaling. Then tried 0.025. Now trying 0.0025 to match the /100 scaling
        # reward = step_reward + 0.0025
        reward = step_reward + 0.15
        self.debug_print(f"Step Reward: {step_reward}")
        self.debug_print(f"Base Reward + Step Reward: {reward}")

        # 
        if self.spatial_features:
            # Get spatial features for reward shaping
            spatial_features = self._calculate_spatial_features()

            # SPATIAL REWARD SHAPING
            # Penalty for fruits getting too high
            if spatial_features["height_severity"] > 0.7:
                height_severity_penalty = spatial_features["height_severity"] * 0.5
                reward -= height_severity_penalty
                # self.debug_print(f"Height Severity Penalty: {height_severity_penalty}")

            # Bonus for good clustering (helps with merging)
            if self.clustering:
                reward += spatial_features["clustering_score"] * 0.1

            # Bonus for good edge occupation
            reward += spatial_features["edge_occupation"] * 0.15

            # Bonus for maintaining clearance at top
            reward += spatial_features["clearance_ratio"] * 0.1

            # Bonus for good spread (don't pile everything in center)
            reward += spatial_features["avg_fruit_spread"] * 0.1

        # Penalize model for repeating same action too many times so it doesn't loop
        if self.action_type == "discrete":
            if self.last_action is not None and action == self.last_action:
                self.repeat_count += 1
            else:
                self.repeat_count = 0
                self.last_action = action

            if self.repeat_count > 6:
                # repeat_penalty = 0.01 # adjusted for scaling from 1.0
                repeat_penalty = 1.0
                self.debug_print(f"Reward: {reward:.6f}")
                reward -= repeat_penalty
                self.debug_print(f"Repeat Move Penalty: -{repeat_penalty}")
                self.debug_print(f"Reward: {reward:.6f}")

        terminated = self.game_over

        # Equivalent of being penalized 300 score
        if terminated:
            self.debug_print("----------GAME OVER----------")
            # reward -= 1.0
            reward -= 100.0

        truncated = False

        # Leaving this for posterity: was experimenting with calculating the height severity only without the rest of the spatial featuress
        # Don't want to _get_obs() twice but want to use obs values for reward penalty so grabbing it here
        # obs = self._get_obs()

        # Replicate height_severity from _calculate_spatial_features() using already-computed obs values
        # obs[7] = killy/H, obs[8] = min_y/H. Same formula as line 204, normalized.

        # Disable height penalty for now for tuning
        # top_norm = config.pad.top / float(self.screen_height)
        # height_severity = np.clip((obs[7] - obs[8]) / (obs[7] - top_norm), 0.0, 1.0)
        # max_height_penalty = 0.005 # was 0.3, trying 0.05
        # self.debug_print(f"Height Severity: {height_severity:.6f} = ({obs[7]:.4f} - {obs[8]:.4f}) / ({obs[7]:.4f} - {top_norm:.4f})")
        # self.debug_print(f"Height Penalty: {max_height_penalty * height_severity:.6f}")
        # reward -= max_height_penalty * height_severity
        # self.debug_print(f"Reward after height penalty: {reward:.6f}")


        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _draw_frame(self, wait_val=0):
        self.screen.blit(config.background_blit, (0, 0))
        
        self.cloud.draw(self.screen, wait_val)
        
        for p in self.space.shapes:
            if isinstance(p, Particle):
                p.draw(self.screen)
        
        draw_score(self.handler.data['score'], self.screen)
        
        if self.game_over:
            draw_gameover(self.screen)

        pygame.display.update()

    def close(self):
        pygame.quit()
