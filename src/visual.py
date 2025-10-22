import matplotlib.pyplot as plt
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
    
    leaf_style = dict(boxstyle='round,pad=0.3', fc='lightgreen', ec='black')
    node_style = dict(boxstyle='sawtooth,pad=0.3', fc='lightblue', ec='black')
    
    if tree['type'] == 'leaf':
        text = f"Room {int(tree['label'])}"
        ax.text(x, y, text, ha='center', va='center', bbox=leaf_style, fontsize=8)
        return
        
    text = f"WIFI {tree['attr']} <= {tree['thr']:.2f}"
    ax.text(x, y, text, ha='center', va='center', bbox=node_style, fontsize=8)
    
    # Draw left child and connecting line
    x_left, y_left = tree['left']['coords']
    ax.plot([x, x_left], [y, y_left], 'k-')
    ax.text((x + x_left) * 0.5, (y + y_left) * 0.5 + 0.1, 'True', ha='center', va='center', fontsize=7)
    draw_tree(tree['left'], ax)
    
    # Draw right child and connecting line
    x_right, y_right = tree['right']['coords']
    ax.plot([x, x_right], [y, y_right], 'k-')
    ax.text((x + x_right) * 0.5, (y + y_right) * 0.5 + 0.1, 'False', ha='center', va='center', fontsize=7)
    draw_tree(tree['right'], ax)

def visualize_tree(tree: Dict[str, Any], depth: int, filename: str = "decision_tree_visualization.png"):
    """
    Main function to visualize and save the decision tree plot.
    """
    print("Generating tree visualization...")
    
    leaf_x_coords = [0] 
    get_coords(tree, x_pos=0, y_pos=depth, y_step=1, leaf_x_coords=leaf_x_coords)
    
    num_leaves = leaf_x_coords[0]
    
    fig_width = max(num_leaves * 1.5, 20)
    fig_height = max((depth + 1) * 2, 10)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.set_ylim(-1, depth + 1)
    ax.set_xlim(-1, num_leaves)
    
    draw_tree(tree, ax)
    
    plt.title("Decision Tree Visualization (Clean Dataset)", fontsize=20)
    
    try:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to '{filename}'")
    except Exception as e:
        print(f"Error saving visualization: {e}")