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
    

    def __init__(self, render_mode=None, action_type="continuous", discrete_bins=128, max_fruits=100, spatial_features=False, debug=False):
        self.render_mode = render_mode
        self.action_type = action_type
        self.discrete_bins = discrete_bins
        self.max_fruits = max_fruits
        self.spatial_features = spatial_features
        self.debug = debug

        self.last_action = None
        self.repeat_count = 0
        
        pygame.init()
        pygame.display.init()
        
        self.screen_width = config.screen.width
        self.screen_height = config.screen.height
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Suika RL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()

        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(self.discrete_bins)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_len = 0
        if self.spatial_features:
            # changed from 9 + to 14 + to adjust for spatial features
            obs_len = 14 + (self.max_fruits * 4)
        else:
            obs_len = 9 + (self.max_fruits * 4)

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
        
    def _normalize(self, val, max_val):
        return val / max_val

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        W = float(self.screen_width)
        H = float(self.screen_height)
        
        obs[0] = self.cloud.curr.n / MAX_TYPE
        obs[1] = self.cloud.curr.radius / MAX_RADIUS
        
        obs[2] = self.cloud.next.n / MAX_TYPE
        obs[3] = self.cloud.next.radius / MAX_RADIUS
        
        obs[4] = config.pad.left / W
        obs[5] = config.pad.right / W
        obs[6] = config.pad.bot / H 
        obs[7] = config.pad.killy / H 
        
        fruits = []
        min_y = H 
        
        for p in self.space.shapes:
            if isinstance(p, Particle) and p.alive:
                fruits.append(p)
                if (p.pos[1] - p.radius) < min_y:
                    min_y = p.pos[1] - p.radius
        
        obs[8] = min_y / H
        
        if self.spatial_features:
            # Try adding spatial features directly into the observation space to help learning
            spatial_features = self._calculate_spatial_features()
            obs[9] = spatial_features["height_severity"]
            obs[10] = spatial_features["clustering_score"]
            obs[11] = spatial_features["edge_occupation"]
            obs[12] = spatial_features["clearance_ratio"]
            obs[13] = spatial_features["avg_fruit_spread"]

        fruits.sort(key=lambda p: (p.pos[1], p.pos[0]))
        
        idx = 0
        if self.spatial_features:
            idx = 14
        else:
            idx = 9
        
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
        height_severity = max(0, (config.pad.killy - min_y) / (config.pad.killy - config.pad.top))

        # CLUSTERING SCORE - reward similar fruits being close
        clustering_score = self._calculate_clustering(fruits, W, H)

        # EDGE OCCUPATION - reward large fruits in corners
        edge_occupation = self._calculate_edge_occupation(fruits, W, H)

        # CLEARANCE RATIO - available vertical space at top
        clearance_ratio = max(0, (min_y - config.pad.top) / (config.pad.killy - config.pad.top))

        # 5. FRUIT SPREAD - penalize overly clustered center
        avg_fruit_spread = self._calculate_spread(fruits, W)

        return {
            "max_height": min_y,
            "height_severity": height_severity,
            "clustering_score": clustering_score,
            "edge_occupation": edge_occupation,
            "clearance_ratio": clearance_ratio,
            "avg_fruit_spread": avg_fruit_spread
        }

    def _calculate_clustering(self, fruits, W, H):
        """Reward similar-sized fruits being close to each other (encourages merging)."""
        if len(fruits) < 2:
            return 1.0

        clustering_reward = 0.0
        count = 0

        # TODO: might be able to optimize this calculation?
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

    def _calculate_edge_occupation(self, fruits, W, H):
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

    def _calculate_spread(self, fruits, W):
        """Calculate how well-distributed fruits are horizontally."""
        if len(fruits) < 2:
            return 1.0

        x_positions = [p.pos[0] for p in fruits]
        # Calculate coefficient of variation of x positions
        std_x = np.std(x_positions)
        spread_ratio = std_x / (W / 2) if W > 0 else 0  # Normalized spread
        return min(1.0, spread_ratio)  # Capped at 1.0

    # Used to make sure all the fruit stop moving before making the next move
    def _is_simulation_at_rest(self, space, linear_threshold=0.1, angular_threshold=0.01):
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
        if self.debug:
            print(f"DEBUG: {log_message}")

    def reset(self, seed=None, options=None):
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
        
        if self.debug:
            self.debug_print(f"act_val: {act_val}")
            self.debug_print(f"target_x: {target_x}")
            self.debug_print(f"int(target_x): {round(target_x)}\n")
            
        self.cloud.release(self.space)

        max_steps = 400

        reward = 0
        initial_score = self.handler.data["score"]

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
            if i >= 90:
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
        step_reward = final_score - initial_score

        # BASE REWARD: Score + small bonus for surviving
        reward = step_reward + 0.25

        # Get spatial features for reward shaping
        spatial_features = self._calculate_spatial_features()

        # SPATIAL REWARD SHAPING
        # Penalty for fruits getting too high
        if spatial_features["height_severity"] > 0.7:
            height_severity_penalty = spatial_features["height_severity"] * 0.5
            reward -= height_severity_penalty
            self.debug_print(f"Height Severity Penalty: {height_severity_penalty}")

        # Bonus for good clustering (helps with merging)
        reward += spatial_features["clustering_score"] * 0.1

        # Bonus for good edge occupation
        reward += spatial_features["edge_occupation"] * 0.15

        # Bonus for maintaining clearance at top
        reward += spatial_features["clearance_ratio"] * 0.1

        # Bonus for good spread (don't pile everything in center)
        reward += spatial_features["avg_fruit_spread"] * 0.1

        if self.action_type == "discrete":
            if self.last_action is not None and action == self.last_action:
                self.repeat_count += 1
            else:
                self.repeat_count = 0
                self.last_action = action

            if self.repeat_count > 2:
                reward -= 1

        terminated = self.game_over

        if terminated:
            reward -= 100.0

        truncated = False

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
