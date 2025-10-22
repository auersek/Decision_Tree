import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import Dict, Any

def get_coords(tree: Dict[str, Any], x_pos: float, y_pos: float, y_step: float, leaf_x_coords: list):
    """
    Recursive helper function to assign (x, y) coordinates
    to each node in the tree for plotting.
    """
    if tree['type'] == 'leaf':
        x = leaf_x_coords[0]
        tree['coords'] = (x, y_pos)
        leaf_x_coords[0] += 1
        return x
    
    y_child = y_pos - y_step
    
    x_left = get_coords(tree['left'], x_pos, y_child, y_step, leaf_x_coords)
    x_right = get_coords(tree['right'], x_pos, y_child, y_step, leaf_x_coords)
    
    x_self = (x_left + x_right) / 2
    tree['coords'] = (x_self, y_pos)
    
    return x_self

def draw_tree(tree: Dict[str, Any], ax):
    """
    Recursive helper function to draw the nodes and
    connecting lines.
    """
    x, y = tree['coords']
    
    leaf_style = dict(boxstyle='round,pad=0.6', fc='lightgreen', ec='black')
    node_style = dict(boxstyle='round4,pad=0.6', fc='lightblue', ec='black')
    
    if tree['type'] == 'leaf':
        text = f"Room {int(tree['label'])}"
        ax.text(x, y, text, ha='center', va='center', bbox=leaf_style, fontsize=11)
        return
        
    text = f"WIFI {tree['attr']} <= {tree['thr']:.2f}"
    ax.text(x, y, text, ha='center', va='center', bbox=node_style, fontsize=11)
    
    # Draw left child and connecting line
    x_left, y_left = tree['left']['coords']
    ax.plot([x, x_left], [y, y_left], 'g-')
    draw_tree(tree['left'], ax)
    
    # Draw right child and connecting line
    x_right, y_right = tree['right']['coords']
    ax.plot([x, x_right], [y, y_right], 'r-')
    draw_tree(tree['right'], ax)

def visualize_tree(tree: Dict[str, Any], depth: int, filename: str = "decision_tree_visualization.png"):
    """
    Main function to visualize and save the decision tree plot.
    """
    
    leaf_x_coords = [0] 
    get_coords(tree, x_pos=0, y_pos=depth, y_step=1, leaf_x_coords=leaf_x_coords)
    
    num_leaves = leaf_x_coords[0]
    
    fig_width = max(num_leaves * 1.0, 12)
    fig_height = max((depth + 1) * 1.2, 7)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.set_ylim(-1, depth + 1)
    ax.set_xlim(-1, num_leaves)
    
    draw_tree(tree, ax)
    
    plt.title("Decision Tree Visualization (Clean Dataset)", fontsize=16)
    
    # Add key/legend for lines
    green_line = mlines.Line2D([], [], color='green', label='True (<=)')
    red_line = mlines.Line2D([], [], color='red', label='False (>)')
    ax.legend(handles=[green_line, red_line], loc='lower right', fontsize=12)
    
    plt.savefig(filename, bbox_inches='tight', dpi=250)
