import numpy as np
import jax.numpy as jnp

"""
Created on June 11th, 2025
@author: Taekyung Kim

@description: 
Visulization for the diffusion, and some functions for model-based diffusion (originated from the original MBD code).
"""


class Env:
    def __init__(self, width=36.0, height=36.0, case="case1", parking_config=None, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.case = case
        self.x_range = (-width/2, width/2)
        self.y_range = (-height/2, height/2)
        
        # Set up obstacles based on case
        self.obs_boundary = self.set_obs_boundary(width, height)
        
        if case == "case1":
            self.obs_circle = self.set_obs_circle_case1()
            self.obs_rectangle = self.set_obs_rectangle_case1()
        elif case == "case2":
            if parking_config is None:
                parking_config = self.get_default_parking_config()
            self.parking_config = parking_config
            self.obs_circle = self.set_obs_circle_case2(parking_config)
            self.obs_rectangle = self.set_obs_rectangle_case2()
        else:
            raise ValueError(f"Unknown case: {case}")
            
        # if self.obs_circle is empty, set a far away obstacle with radius 0
        if len(self.obs_circle) == 0:
            self.obs_circle = [[0.0, 0.0, 0.0]]

    def get_default_parking_config(self):
        """Default parking configuration for case 2"""
        return {
            'parking_rows': 2,
            'parking_cols': 8,
            'space_width': 3.5,     # Width of each parking space
            'space_length': 7.0,    # Length of each parking space
            'parking_y_offset': 8.0, # Distance from start area to parking lot
            'occupied_spaces': [1, 2,  4, 5, 6, 7, 9, 10, 12, 14, 15 ],  # 1-indexed, 4 and 9 are vacant
            #'occupied_spaces': [],  # 1-indexed, 4 and 9 are vacant
            'target_spaces': [3, 11],  # Target: tractor in 4, trailer in 9
            'obstacle_radius': 1.2,   # Radius of obstacles in occupied spaces
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

    def set_obs_rectangle_case1(self):
        """Set rectangular obstacles for case 1 (navigation scenario)"""
        # Default rectangular obstacles: [x_center, y_center, width, height, rotation_angle]
        obs_rectangle = [
            [-8.0, 8.0, 4.0, 2.0, 0.2],    # Rectangle 1: slightly rotated
            [6.0, -4.0, 3.0, 6.0, -0.3],   # Rectangle 2: vertical-ish, rotated
            [0.0, 12.0, 8.0, 1.5, 0.0],    # Rectangle 3: horizontal barrier
        ]
        return obs_rectangle

    def set_obs_rectangle_case2(self):
        """Set rectangular obstacles for case 2 (parking scenario)"""
        obs_rectangle = [
            [0.0, -5.0, 10.0, 4.0, 0.0]
        ]
        return obs_rectangle
    
    def set_obs_circle_case1(self):
        """Set circular obstacles for case 1 (original scenario)"""
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
        obs_cir = [[-r_obs * 3, r_obs * 2, r_obs]]
        return obs_cir

    def set_obs_circle_case2(self, parking_config):
        """Set circular obstacles for case 2 (parking scenario)"""
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
        parking_start_y = self.y_range[1] - y_offset - parking_lot_height
        
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
        if self.case != "case2":
            raise ValueError("Parking spaces only available in case2")
            
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
        parking_start_y = self.y_range[1] - y_offset - parking_lot_height
        
        # Convert 1-indexed space number to row, col (0-indexed)
        space_idx = space_num - 1
        row = space_idx // cols
        col = space_idx % cols
        
        # Calculate center of parking space
        space_center_x = parking_start_x + (col + 0.5) * space_width
        space_center_y = parking_start_y + (row + 0.5) * space_length
        
        return space_center_x, space_center_y

    def get_initial_position_case2(self):
        """Get initial position for case 2 (parking scenario)"""
        # Start from left top corner
        start_x = self.x_range[0] + 2.0  # Small offset from boundary
        start_y = self.y_range[1] - 2.0  # Small offset from top
        theta1 = 0.1  # Nearly zero angle
        theta2 = 0.1  # Nearly zero angle
        return jnp.array([start_x, start_y, theta1, theta2])

    def get_goal_position_case2(self):
        """Get goal position for case 2 (parking scenario)"""
        target_space = self.parking_config['target_spaces'][0]  # Tractor target (space 9)
        goal_x, goal_y = self.get_parking_space_center(target_space)
        # Goal orientation: facing into the parking space (approximately pi/2 for vertical spaces)
        theta1 = -np.pi / 2
        theta2 = -np.pi / 2
        return jnp.array([goal_x, goal_y, theta1, theta2])

    def get_obstacles(self):
        """Get obstacles for collision checking - returns both circular and rectangular"""
        obstacles = {
            'circles': jnp.array(self.obs_circle) if len(self.obs_circle) > 0 else jnp.array([]).reshape(0, 3),
            'rectangles': jnp.array(self.obs_rectangle) if len(self.obs_rectangle) > 0 else jnp.array([]).reshape(0, 5)
        }
        return obstacles

    def get_circular_obstacles(self):
        """Get only circular obstacles (for backward compatibility)"""
        return jnp.array(self.obs_circle) if len(self.obs_circle) > 0 else jnp.array([]).reshape(0, 3)

    def get_rectangular_obstacles(self):
        """Get only rectangular obstacles"""
        return jnp.array(self.obs_rectangle) if len(self.obs_rectangle) > 0 else jnp.array([]).reshape(0, 5)

    def is_in_bounds(self, x, y):
        """Check if point is within environment bounds"""
        return (self.x_range[0] <= x <= self.x_range[1] and 
                self.y_range[0] <= y <= self.y_range[1])

    def get_plot_limits(self):
        """Get plot limits for visualization"""
        return self.x_range, self.y_range

    def print_parking_layout(self):
        """Print the parking layout for visualization"""
        if self.case != "case2":
            print("Parking layout only available for case2")
            return
            
        config = self.parking_config
        rows = config['parking_rows']
        cols = config['parking_cols']
        occupied = set(config['occupied_spaces'])
        target = config['target_spaces']
        
        print("Parking Layout:")
        print("O = Occupied, E = Empty, T = Target")
        print("-" * (cols * 4 + 1))
        
        for row in range(rows):
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
            print(line)
            print("-" * (cols * 4 + 1))
