import jax
#from brax.io import html
import os
import glob
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerPatch
from tqdm import tqdm
import mbd
import jax.numpy as jnp


# evaluate the diffused uss
def eval_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rew_seq = jax.lax.scan(step, state, us)
    return rew_seq

def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)
    # Warm-initialize carry to match shape if necessary by a no-op step
    zero_u = jnp.zeros_like(us[0]) if us.shape[0] > 0 else jnp.zeros((2,))
    state_init = step_env(state, zero_u)
    final_state, (rew_seq, pipeline_state_seq) = jax.lax.scan(step, state_init, us) # NOTE: returns stack of (rew, pipeline_state). _ is the final carry, which is final state in this case
    return rew_seq, pipeline_state_seq

def rollout_us_with_terminal(step_env, env, state, us):
    """Rollout with terminal reward computed separately from stage rewards"""
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)
    zero_u = jnp.zeros_like(us[0]) if us.shape[0] > 0 else jnp.zeros((2,))
    state_init = step_env(state, zero_u)
    final_state, (rew_seq, pipeline_state_seq) = jax.lax.scan(step, state_init, us)
    
    # Compute mean of stage rewards first, then add separate terminal reward
    stage_reward_mean = rew_seq.mean()
    terminal_reward = env.get_terminal_reward(final_state.pipeline_state)
    total_reward = stage_reward_mean + env.terminal_reward_weight * terminal_reward
    
    # Return total reward instead of per-timestep rewards for consistency with new logic
    return total_reward, pipeline_state_seq

# def render_us(step_env, sys, state, us):
#     rollout = []
#     rew_sum = 0.0
#     Hsample = us.shape[0]
#     for i in range(Hsample):
#         rollout.append(state.pipeline_state)
#         state = step_env(state, us[i])
#         rew_sum += state.reward
#     # rew_mean = rew_sum / (Hsample)
#     # print(f"evaluated reward mean: {rew_mean:.2e}")
#     return html.render(sys, rollout)


def setup_animation_saving(env_name, animation_type="trajectory"):
    """Setup directories for animation saving"""
    current_directory_path = os.getcwd()
    animation_path = f"{mbd.__path__[0]}/../results/{env_name}/animations"
    if not os.path.exists(animation_path):
        os.makedirs(animation_path)
    
    # Create subdirectory for specific animation type
    type_path = f"{animation_path}/{animation_type}"
    if not os.path.exists(type_path):
        os.makedirs(type_path)
    
    # if file exists, delete all
    if os.path.exists(type_path):
        for file_name in glob.glob(f"{type_path}/*.png"):
            os.remove(file_name)
    return type_path


def export_video(env_name, animation_type="trajectory", video_name=None):
    """Convert image sequence to video using ffmpeg"""
    # Use the same path structure as setup_animation_saving
    animation_path = f"{mbd.__path__[0]}/../results/{env_name}/animations/{animation_type}"
    
    if video_name is None:
        if animation_type == "denoising":
            video_name = "denoising_process_animation.mp4"
        else:
            video_name = "tractor_trailer_animation.mp4"
    
    # Debug: Check if frames exist
    frame_files = glob.glob(f"{animation_path}/frame_*.png")
    print(f"Animation path: {animation_path}")
    print(f"Found {len(frame_files)} frame files")
    if len(frame_files) > 0:
        print(f"First frame: {frame_files[0]}")
        print(f"Last frame: {frame_files[-1]}")
    else:
        print("ERROR: No frame files found!")
        return
    
    # Create video using ffmpeg
    fps = 10 if animation_type == "trajectory" else 30 # 10 fps for trajectory, 30 fps for denoising
    result = subprocess.call(['ffmpeg', '-y',  # -y to overwrite existing files
                     '-framerate', str(fps),  # Input framerate
                     '-i', f'{animation_path}/frame_%04d.png',
                     '-vf', 'scale=1920:1080,fps=30',  # Scale and set output framerate
                     '-pix_fmt', 'yuv420p',
                     f'{animation_path}/{video_name}'])
    
    if result == 0:
        print("Video created successfully!")
        # Clean up individual frames
        for file_name in glob.glob(f"{animation_path}/*.png"):
            os.remove(file_name)
        print(f"Animation saved to: {animation_path}/{video_name}")
    else:
        print(f"FFmpeg failed with return code: {result}")


def create_animation(env, trajectory_states, trajectory_actions, args, guided_trajectory_overlay=None):
    """Create animation of the tractor-trailer trajectory
    
    Args:
        env: Environment object
        trajectory_states: Main trajectory states for vehicle animation (usually unguided for guidance case)
        trajectory_actions: Actions corresponding to trajectory states  
        args: Configuration arguments
        guided_trajectory_overlay: Optional guided trajectory states to overlay as path
    """
    print("Creating animation...")
    
    # Setup animation saving if enabled
    if args.save_animation:
        animation_path = setup_animation_saving(args.env_name, "trajectory")
    
    # Setup interactive plotting
    plt.close() # close any existing figures
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get plot limits from environment
    x_range, y_range = env.env.get_plot_limits()
    
    # Set up plot properties
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    #ax.grid(True)

    # Set title based on environment type
    if args.env_name == "kinematic_bicycle2d":
        ax.set_title("Kinematic Bicycle Parking")
    else:
        ax.set_title("Tractor-Trailer Parking")

    
    # Add parking space boundaries for parking scenario
    if args.case == "parking" and hasattr(env.env, 'parking_config'):
        config = env.env.parking_config
        rows = config['parking_rows']
        cols = config['parking_cols']
        space_width = config['space_width']
        space_length = config['space_length']
        y_offset = config['parking_y_offset']
        
        # Calculate parking lot position
        parking_lot_width = cols * space_width
        parking_lot_height = rows * space_length
        parking_start_x = -parking_lot_width / 2
        parking_start_y = y_range[0] + y_offset
        
        # Draw parking space boundaries
        for row in range(rows + 1):
            y = parking_start_y + row * space_length
            ax.plot([parking_start_x, parking_start_x + parking_lot_width], [y, y], 'k-', alpha=0.3, linewidth=1)
        
        for col in range(cols + 1):
            x = parking_start_x + col * space_width
            ax.plot([x, x], [parking_start_y, parking_start_y + parking_lot_height], 'k-', alpha=0.3, linewidth=1)
        
        # Add parking space numbers
        for row in range(rows):
            for col in range(cols):
                space_num = row * cols + col + 1
                space_center_x = parking_start_x + (col + 0.5) * space_width
                space_center_y = parking_start_y + (row + 0.5) * space_length
                
                # Color code: target spaces in green, occupied in red, empty in white
                if space_num in config['target_spaces']:
                    color = 'lightgreen'
                    text_color = 'black'
                elif space_num in config['occupied_spaces']:
                    color = 'lightcoral'
                    text_color = 'white'
                else:
                    color = 'lightblue'
                    text_color = 'black'
                
                # Add colored background for space number
                if space_num not in config['occupied_spaces']:  # Don't show numbers on occupied spaces (they have obstacles)
                    ax.text(space_center_x, space_center_y, str(space_num), 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                           color=text_color)
    
    # Add obstacles
    obstacles = env.env.get_obstacles()
    obs_circles = obstacles['circles']
    obs_rectangles = obstacles['rectangles']
    
    # Render circular obstacles
    if obs_circles.shape[0] > 0:
        for i in range(obs_circles.shape[0]):
            circle = plt.Circle(
                obs_circles[i, :2], obs_circles[i, 2], color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
    
    # Render rectangular obstacles
    if obs_rectangles.shape[0] > 0:
        for i in range(obs_rectangles.shape[0]):
            x_center, y_center, width, height, angle = obs_rectangles[i]
            
            # Create rectangle patch
            rect = plt.Rectangle((-width/2, -height/2), width, height,
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            
            # Apply rotation and translation
            transform = (Affine2D()
                       .rotate(angle)
                       .translate(x_center, y_center) + ax.transData)
            rect.set_transform(transform)
            ax.add_patch(rect)
    
    # # Add demonstration trajectory if available
    # if args.enable_demo and hasattr(env, 'xref') and env.xref is not None:
    #     ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", linewidth=2, label="Demonstration path", alpha=0.7)
    
    # Add start and goal markers
    ax.scatter(env.x0[0], env.x0[1], c='blue', s=150, marker='o', edgecolor='black', linewidth=2, label='Start', zorder=5)
    ax.scatter(env.xg[0], env.xg[1], c='red', s=150, marker='*', edgecolor='black', linewidth=2, label='Goal', zorder=5)
    
    # Setup animation patches
    env.setup_animation_patches(ax)
    
    # Add trajectory trace
    if guided_trajectory_overlay is not None:
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.8, linewidth=3, label='Unguided trajectory', zorder=4)
    else:
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.8, linewidth=3, label='Trajectory', zorder=4)
    
    # Add guided trajectory overlay if provided
    guided_trajectory_line = None
    if guided_trajectory_overlay is not None:
        guided_trajectory_line, = ax.plot([], [], 'r--', alpha=0.8, linewidth=2, 
                                        label='Guided trajectory', zorder=3)
    
    # Add violation markers to legend using colored squares
    # Create square patches for violation markers
    collision_patch = Rectangle((0, 0), 1, 1, facecolor='#FF3C3C', edgecolor='#FF3C3C', linewidth=0.5)
    is_bicycle = args.env_name == "kinematic_bicycle2d"
    
    if not is_bicycle:
        jackknife_patch = Rectangle((0, 0), 1, 1, facecolor='#CC00FF', edgecolor='#CC00FF', linewidth=0.5)
    
    # Custom handler to ensure squares remain square in legend
    class SquareHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Make it square by using the smaller of width/height
            size = min(width, height)
            # Center the square
            x_offset = (width - size) / 2
            y_offset = (height - size) / 2
            
            square = Rectangle((x_offset, y_offset), size, size, 
                             facecolor=orig_handle.get_facecolor(),
                             edgecolor=orig_handle.get_edgecolor(),
                             linewidth=orig_handle.get_linewidth(),
                             transform=trans)
            return [square]
    
    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Add violation markers to legend (only collision for bicycle, both for tractor-trailer)
    handles.append(collision_patch)
    labels.append('Collision')
    handler_map = {collision_patch: SquareHandler()}
    
    if not is_bicycle:
        handles.append(jackknife_patch)
        labels.append('Jackknifing')
        handler_map[jackknife_patch] = SquareHandler()
    
    # Create legend with custom handler for square patches
    legend = ax.legend(handles, labels, handler_map=handler_map)
    fig.tight_layout()
    
    # Animation loop
    traj_x = []
    traj_y = []
    guided_traj_x = []
    guided_traj_y = []
    violation_markers = []  # Store references to exclamation mark text objects
    
    for frame_idx, (state, action) in enumerate(zip(trajectory_states, trajectory_actions)):
        # Convert jax arrays to numpy for matplotlib
        state_np = np.array(state)
        action_np = np.array(action) if action is not None else None
        
        # Clear previous violation markers
        for marker in violation_markers:
            marker.remove()
        violation_markers.clear()
        
        # Update robot visualization
        env.update_animation_patches(state_np, action_np)
        
        # Handle different state dimensions
        is_bicycle = args.env_name == "kinematic_bicycle2d"
        is_multi_trailer = hasattr(env, 'num_trailers') and getattr(env, 'num_trailers', 1) >= 2
        if is_bicycle:
            state_for_collision = state_np[:3]
        elif is_multi_trailer:
            state_for_collision = state_np  # use full N-trailer state
        else:
            state_for_collision = state_np[:4]
        
        # Check obstacle collision
        obstacle_collision = env.check_obstacle_collision(state_for_collision, env.obs_circles, env.obs_rectangles)
        
        # Check hitch angle violation (only for tractor-trailer)
        if is_bicycle:
            hitch_violation = False  # No hitch angle for bicycle
        else:
            hitch_violation = env.check_hitch_violation(state_for_collision)
        
        # Add exclamation marks for violations
        current_position = state_np[:2]
        
        if obstacle_collision:
            marker = ax.text(current_position[0] + 2.0, current_position[1] + 0.5, 
                           '!', color='#FF3C3C', weight='bold', fontsize=32, 
                           ha='center', va='center', zorder=10)
            violation_markers.append(marker)
        
        if hitch_violation:
            marker = ax.text(current_position[0] + 3.0, current_position[1] + 0.5, 
                           '!', color='#CC00FF', weight='bold', fontsize=32, 
                           ha='center', va='center', zorder=10)
            violation_markers.append(marker)
        
        # Update trajectory trace
        traj_x.append(state_np[0])
        traj_y.append(state_np[1])
        trajectory_line.set_data(traj_x, traj_y)
        
        # Update guided trajectory overlay progressively if provided
        if guided_trajectory_line is not None and frame_idx < len(guided_trajectory_overlay):
            guided_state = guided_trajectory_overlay[frame_idx]
            guided_traj_x.append(guided_state[0])
            guided_traj_y.append(guided_state[1])
            guided_trajectory_line.set_data(guided_traj_x, guided_traj_y)
        
        # Update plot
        if args.show_animation:
            plt.pause(0.01)  # Small pause for animation effect
        else:
            plt.draw()
        
        # Save frame if animation saving is enabled
        if args.save_animation:
            frame_filename = f"{animation_path}/frame_{frame_idx:04d}.png"
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
    
    # Clear any remaining violation markers
    for marker in violation_markers:
        marker.remove()
    violation_markers.clear()
    
    # Show final result
    if args.show_animation:
        plt.show()
    
    plt.ioff()
    plt.close()
    
    # Export video if saving animation
    if args.save_animation:
        export_video(args.env_name, "trajectory")


def create_denoising_animation(env, Yi, args, step_env_jit, state_init, frame_skip=1):
    """Create animation showing the denoising process through all diffusion steps"""
    print("Creating denoising process animation...")
    
    # Setup animation saving
    animation_path = setup_animation_saving(args.env_name, "denoising")
    
    # Setup plotting
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get plot limits from environment
    x_range, y_range = env.env.get_plot_limits()
    
    # Set up plot properties
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    #ax.grid(True)
    
    # Add parking space boundaries for parking scenario
    if args.case == "parking" and hasattr(env.env, 'parking_config'):
        config = env.env.parking_config
        rows = config['parking_rows']
        cols = config['parking_cols']
        space_width = config['space_width']
        space_length = config['space_length']
        y_offset = config['parking_y_offset']
        
        # Calculate parking lot position
        parking_lot_width = cols * space_width
        parking_lot_height = rows * space_length
        parking_start_x = -parking_lot_width / 2
        parking_start_y = y_range[0] + y_offset
        
        # Draw parking space boundaries
        for row in range(rows + 1):
            y = parking_start_y + row * space_length
            ax.plot([parking_start_x, parking_start_x + parking_lot_width], [y, y], 'k-', alpha=0.3, linewidth=1)
        
        for col in range(cols + 1):
            x = parking_start_x + col * space_width
            ax.plot([x, x], [parking_start_y, parking_start_y + parking_lot_height], 'k-', alpha=0.3, linewidth=1)
        
        # Add parking space numbers
        for row in range(rows):
            for col in range(cols):
                space_num = row * cols + col + 1
                space_center_x = parking_start_x + (col + 0.5) * space_width
                space_center_y = parking_start_y + (row + 0.5) * space_length
                
                # Color code: target spaces in green, occupied in red, empty in white
                if space_num in config['target_spaces']:
                    color = 'lightgreen'
                    text_color = 'black'
                elif space_num in config['occupied_spaces']:
                    color = 'lightcoral'
                    text_color = 'white'
                else:
                    color = 'lightblue'
                    text_color = 'black'
                
                # Add colored background for space number
                if space_num not in config['occupied_spaces']:  # Don't show numbers on occupied spaces (they have obstacles)
                    ax.text(space_center_x, space_center_y, str(space_num), 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                           color=text_color)
    
    # Add obstacles
    obstacles = env.env.get_obstacles()
    obs_circles = obstacles['circles']
    obs_rectangles = obstacles['rectangles']
    
    # Render circular obstacles
    if obs_circles.shape[0] > 0:
        for i in range(obs_circles.shape[0]):
            circle = plt.Circle(
                obs_circles[i, :2], obs_circles[i, 2], color="k", fill=True, alpha=0.5
            )
            ax.add_artist(circle)
    
    # Render rectangular obstacles
    if obs_rectangles.shape[0] > 0:
        for i in range(obs_rectangles.shape[0]):
            x_center, y_center, width, height, angle = obs_rectangles[i]
            
            # Create rectangle patch
            rect = plt.Rectangle((-width/2, -height/2), width, height,
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            
            # Apply rotation and translation
            transform = (Affine2D()
                       .rotate(angle)
                       .translate(x_center, y_center) + ax.transData)
            rect.set_transform(transform)
            ax.add_patch(rect)
    
    # Add demonstration trajectory if available
    if args.enable_demo and hasattr(env, 'xref') and env.xref is not None:
        ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", linewidth=2, label="Demonstration path", alpha=0.7)
    
    # Add start and goal markers
    ax.scatter(env.x0[0], env.x0[1], c='blue', s=150, marker='o', edgecolor='black', linewidth=2, label='Start', zorder=5)
    ax.scatter(env.xg[0], env.xg[1], c='red', s=150, marker='*', edgecolor='black', linewidth=2, label='Goal', zorder=5)
    
    # Create plot handles for trajectory
    trajectory_scatter = ax.scatter([], [], c=[], cmap="Reds", s=45, zorder=5)
    trajectory_line, = ax.plot([], [], 'r-', linewidth=1.5, label='Tractor path', zorder=6)
    
    ax.legend()
    fig.tight_layout()
    
    # Loop through each denoising step
    frame_idx = 0
    total_steps = Yi.shape[0]
    
    with tqdm(range(total_steps), desc="Creating denoising animation") as pbar:
        for step in pbar:
            # Only process frames that aren't skipped
            if step % frame_skip == 0:
                # Get actions for current denoising step
                actions = Yi[step]  # Shape: [Hsample, Nu]
                
                # Rollout trajectory for current denoising step
                trajectory_states = [state_init.pipeline_state]
                state = state_init
                
                traj_x = [np.array(state_init.pipeline_state)[0]]
                traj_y = [np.array(state_init.pipeline_state)[1]]
                
                for t in range(actions.shape[0]):
                    action_np = np.array(actions[t])
                    state = step_env_jit(state, actions[t])
                    state_np = np.array(state.pipeline_state)
                    trajectory_states.append(state_np)
                    traj_x.append(state_np[0])
                    traj_y.append(state_np[1])
                
                # Update scatter plot with colored points along trajectory
                if len(traj_x) > 0:
                    positions = np.column_stack([traj_x, traj_y])
                    colors = np.arange(len(traj_x))  # Color progression along trajectory
                    trajectory_scatter.set_offsets(positions)
                    trajectory_scatter.set_array(colors)
                    
                # Update trajectory line using set_data
                trajectory_line.set_data(traj_x, traj_y)
                # Update opacity to show progression
                alpha = 0.3 + 0.7 * (step / max(1, total_steps - 1))  # Gradually increase opacity
                trajectory_line.set_alpha(alpha)
                
                # Update title to show current denoising step
                diffusion_step = args.Ndiffuse - 1 - step
                title = f"Denoising Step {diffusion_step}/{args.Ndiffuse-1}"
                ax.set_title(title)
                
                # Save frame
                plt.draw()
                frame_filename = f"{animation_path}/frame_{frame_idx:04d}.png"
                plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
                frame_idx += 1
                
                pbar.set_postfix({"step": f"{diffusion_step}"})
    plt.ioff()
    plt.close()
    
    # Export video
    export_video(args.env_name, "denoising", "denoising_process_animation.mp4")

