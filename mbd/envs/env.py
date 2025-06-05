import numpy as np
import jax.numpy as jnp


class Env:
    def __init__(self, width=36.0, height=36.0, known_obs=[], resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.x_range = (-width/2, width/2)  # Centered around origin
        self.y_range = (-height/2, height/2)  # Centered around origin
        
        # Set up obstacles
        self.obs_boundary = self.set_obs_boundary(width, height)
        self.obs_circle = self.set_obs_circle(known_obs)
        self.obs_rectangle = self.set_obs_rectangle()

    @staticmethod
    def set_obs_boundary(width, height):
        """Set boundary obstacles (walls around the environment)"""
        # For now, no boundary walls - tractor-trailer operates in open space
        obs_boundary = []
        return obs_boundary

    @staticmethod
    def set_obs_rectangle():
        """Set rectangular obstacles"""
        obs_rectangle = []
        return obs_rectangle
    
    @staticmethod
    def set_obs_circle(known_obs):
        """Set circular obstacles"""
        if len(known_obs) > 0:
            return known_obs
        
        # Default obstacle configuration (scaled for 6x environment)
        r_obs = 1.8
        obs_cir = [
            [-r_obs * 3, r_obs * 2, r_obs],
            [-r_obs * 2, r_obs * 2, r_obs],
            [-r_obs * 1, r_obs * 2, r_obs],
            [0.0, r_obs * 2, r_obs],
            [0.0, r_obs * 1, r_obs],
            [0.0, 0.0, r_obs],
            [0.0, -r_obs * 1, r_obs],
            [-r_obs * 3, -r_obs * 2, r_obs],
            [-r_obs * 2, -r_obs * 2, r_obs],
            [-r_obs * 1, -r_obs * 2, r_obs],
            [0.0, -r_obs * 2, r_obs],
        ]
        return obs_cir

    def get_obstacles(self):
        """Get obstacles for collision checking"""
        return jnp.array(self.obs_circle)

    def is_in_bounds(self, x, y):
        """Check if point is within environment bounds"""
        return (self.x_range[0] <= x <= self.x_range[1] and 
                self.y_range[0] <= y <= self.y_range[1])

    def get_plot_limits(self):
        """Get plot limits for visualization"""
        return self.x_range, self.y_range
