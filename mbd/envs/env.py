import numpy as np
import jax.numpy as jnp
import logging

"""
Created on June 11th, 2025
@author: Taekyung Kim

@description: 
Visulization for the diffusion, and some functions for model-based diffusion (originated from the original MBD code).
"""


class Env:
    def __init__(self, width=45.0, height=45.0, case="parking", parking_config=None, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.case = case
        self.x_range = (-width/2, width/2)
        self.y_range = (-height/2, height/2)
        
        # Set up obstacles based on case
        self.obs_boundary = self.set_obs_boundary(width, height)
        
        if case == "parking":
            if parking_config is None:
                parking_config = self.get_default_parking_config()
            self.parking_config = parking_config
            self.obs_circles = self.set_obs_circle_parking(parking_config)
            self.obs_rectangles = self.set_obs_rectangle_parking()
        elif case == "navigation":
            # Navigation test case - obstacles will be set via set_rectangle_obs
            self.obs_circles = []  # No circular obstacles for navigation
            self.obs_rectangles = []  # Will be set later via set_rectangle_obs
        else:
            raise ValueError(f"Unknown case: {case}")
            
        # if self.obs_circles is empty, set a far away obstacle with radius 0
        if len(self.obs_circles) == 0:
            self.obs_circles = [[0.0, 0.0, 0.0]]

    def get_default_parking_config(self):
        """Default parking configuration for parking scenario"""
        return {
            'parking_rows': 2,
            'parking_cols': 8,
            'space_width': 3.5,     # Width of each parking space
            'space_length': 7.0,    # Length of each parking space
            'parking_y_offset': 4.0, # Distance from start area to parking lot
            'occupied_spaces': [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15 ],  # 1-indexed, 4 and 9 are vacant
            #'occupied_spaces': [4,12],  # 1-indexed, 4 and 9 are vacant
            #'occupied_spaces': [],  # 1-indexed, 4 and 9 are vacant
            'target_spaces': [3, 11],  # Target: tractor in 4, trailer in 9
            'obstacle_radius': 1.0,   # Radius of obstacles in occupied spaces
        }

    @staticmethod
    def set_obs_boundary(width, height):
        """Set boundary obstacles (walls around the environment)"""
        # For now, no boundary walls - tractor-trailer operates in open space
        obs_boundary = []
        return obs_boundary

    @staticmethod
    def set_obs_rectangle():
        """Set rectangular obstacles (deprecated - use case-specific methods)"""
        return []

    def set_obs_rectangle_parking(self):
        """Set rectangular obstacles for parking scenario"""
        obs_rectangles = [
            [0.0, 15.0, 30.0, 1.0, 0.0],
            [0.0, -14.0, 30.0, 1.0, 0.0]
        ]
        obs_rectangles = [
        ]
        return obs_rectangles
    
    def set_obs_circle_parking(self, parking_config):
        """Set circular obstacles for parking scenario"""
        obs_cir = []
        
        # Extract parking configuration
        rows = parking_config['parking_rows']
        cols = parking_config['parking_cols']
        space_width = parking_config['space_width']
        space_length = parking_config['space_length']
        y_offset = parking_config['parking_y_offset']
        occupied_spaces = parking_config['occupied_spaces']
        obstacle_radius = parking_config['obstacle_radius']
        
        # Calculate parking lot position (centered horizontally, offset vertically)
        parking_lot_width = cols * space_width
        parking_lot_height = rows * space_length
        
        # Starting position of parking lot (bottom-left corner)
        parking_start_x = -parking_lot_width / 2
        parking_start_y = self.y_range[0] + y_offset
        
        # Create obstacles for occupied spaces
        for space_num in occupied_spaces:
            # Convert 1-indexed space number to row, col (0-indexed)
            space_idx = space_num - 1
            row = space_idx // cols
            col = space_idx % cols
            
            # Calculate center of parking space
            space_center_x = parking_start_x + (col + 0.5) * space_width
            space_center_y = parking_start_y + (row + 0.5) * space_length
            
            # Add three obstacles - one at center and two offset vertically by 1m
            delta_y = 2.0
            obs_cir.append([space_center_x, space_center_y, obstacle_radius])  # Center
            obs_cir.append([space_center_x, space_center_y + delta_y, obstacle_radius])  # Above
            obs_cir.append([space_center_x, space_center_y - delta_y, obstacle_radius])  # Below
        
        return obs_cir

    def get_parking_space_center(self, space_num):
        """Get the center coordinates of a specific parking space"""
        if self.case != "parking":
            raise ValueError("Parking spaces only available in parking scenario")
            
        parking_config = self.parking_config
        rows = parking_config['parking_rows']
        cols = parking_config['parking_cols']
        space_width = parking_config['space_width']
        space_length = parking_config['space_length']
        y_offset = parking_config['parking_y_offset']
        
        # Calculate parking lot position
        parking_lot_width = cols * space_width
        parking_lot_height = rows * space_length
        parking_start_x = -parking_lot_width / 2
        parking_start_y = self.y_range[0] + y_offset
        
        # Convert 1-indexed space number to row, col (0-indexed)
        space_idx = space_num - 1
        row = space_idx // cols
        col = space_idx % cols
        
        # Calculate center of parking space
        space_center_x = parking_start_x + (col + 0.5) * space_width
        space_center_y = parking_start_y + (row + 0.5) * space_length
        
        return space_center_x, space_center_y

    def get_default_init_pos(self, case):
        """Get default initial position"""
        if case == "parking":
            # Start from left top corner
            start_x = self.x_range[0] + 2.0  # Small offset from boundary
            start_y = self.y_range[1] - 2.0  # Small offset from top
            theta1 = 0.1  # Nearly zero angle
            theta2 = 0.1  # Nearly zero angle
        else:
            raise ValueError(f"Unknown case: {case}")
        return jnp.array([start_x, start_y, theta1, theta2])

    def get_default_goal_pos(self, case):
        """Get default goal position"""
        if case == "parking":
            target_space = self.parking_config['target_spaces'][0]  # Tractor target (space 9)
            goal_x, goal_y = self.get_parking_space_center(target_space)
            # Goal orientation: facing into the parking space (approximately pi/2 for vertical spaces)
            theta1 = -np.pi / 2
            theta2 = -np.pi / 2
        else:
            raise ValueError(f"Unknown case: {case}")
        return jnp.array([goal_x, goal_y, theta1, theta2])

    def get_obstacles(self):
        """Get obstacles for collision checking - returns both circular and rectangular"""
        obstacles = {
            'circles': jnp.array(self.obs_circles) if len(self.obs_circles) > 0 else jnp.array([]).reshape(0, 3),
            'rectangles': jnp.array(self.obs_rectangles) if len(self.obs_rectangles) > 0 else jnp.array([]).reshape(0, 5)
        }
        return obstacles

    def get_circular_obstacles(self):
        """Get only circular obstacles (for backward compatibility)"""
        return jnp.array(self.obs_circles) if len(self.obs_circles) > 0 else jnp.array([]).reshape(0, 3)

    def get_rectangular_obstacles(self):
        """Get only rectangular obstacles"""
        return jnp.array(self.obs_rectangles) if len(self.obs_rectangles) > 0 else jnp.array([]).reshape(0, 5)

    def is_in_bounds(self, x, y):
        """Check if point is within environment bounds"""
        return (self.x_range[0] <= x <= self.x_range[1] and 
                self.y_range[0] <= y <= self.y_range[1])

    def get_plot_limits(self):
        """Get plot limits for visualization"""
        return self.x_range, self.y_range

    def print_parking_layout(self):
        """Print the parking layout for visualization"""
        if self.case != "parking":
            logging.info("Parking layout only available for parking scenario")
            return
            
        config = self.parking_config
        rows = config['parking_rows']
        cols = config['parking_cols']
        occupied = set(config['occupied_spaces'])
        target = config['target_spaces']
        
        logging.debug("Parking Layout:")
        logging.debug("O = Occupied, E = Empty, T = Target")
        logging.debug("-" * (cols * 4 + 1))
        
        for row in range(rows-1, -1, -1):  # Iterate rows in reverse order
            line = "|"
            for col in range(cols):
                space_num = row * cols + col + 1
                if space_num in target:
                    symbol = "T"
                elif space_num in occupied:
                    symbol = "O"
                else:
                    symbol = "E"
                line += f" {symbol} |"
            logging.debug(line)
            logging.debug("-" * (cols * 4 + 1))

    def set_rectangle_obs(self, obstacles, coordinate_mode="left-top", padding=0.0):
        """
        Set rectangular obstacles for navigation scenario
        
        Args:
            obstacles: List of obstacles, each can be:
                - [x, y, width, height] (4 elements, rotation = 0)
                - [x, y, width, height, rotation] (5 elements)
            coordinate_mode: "left-top" or "center"
                - "left-top": x,y represent upper-left corner (CarMaker convention)
                - "center": x,y represent center (MBD convention)
        """
        converted_obstacles = []
        
        for obs in obstacles:
            if len(obs) == 4:
                x, y, width, height = obs
                rotation = 0.0
            elif len(obs) == 5:
                x, y, width, height, rotation = obs
            else:
                raise ValueError(f"Obstacle must have 4 or 5 elements, got {len(obs)}")
            
            if coordinate_mode == "left-top":
                # Convert from left-top to center coordinates
                center_x = x + width / 2
                center_y = y - height / 2  # Note: y decreases downward in CarMaker
                converted_obstacles.append([center_x, center_y, width+padding*2, height+padding*2, rotation])
            elif coordinate_mode == "center":
                # Already in center coordinates
                converted_obstacles.append([x, y, width+padding*2, height+padding*2, rotation])
            else:
                raise ValueError(f"coordinate_mode must be 'left-top' or 'center', got {coordinate_mode}")
        
        self.obs_rectangles = np.array(converted_obstacles)
        logging.debug(f"Set {len(converted_obstacles)} rectangular obstacles with {coordinate_mode} coordinates")
        return self.obs_rectangles

    def set_plot_limits(self, x_range, y_range):
        """Set custom plot limits for visualization"""
        self.x_range = x_range
        self.y_range = y_range
        logging.debug(f"Plot limits set to x_range: {self.x_range}, y_range: {self.y_range}")


