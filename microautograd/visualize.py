from graphviz import Digraph

def get_node_color(value, nodes):
    # A node with no parents is an input (leaf)
    if len(value._prev) == 0:
        return '#E0F7FA'  # Soft mint – input
    # A node with no children in the graph is an output
    elif not any(value in parent._prev for parent in nodes):
        return '#FFF3E0'  # Light peach – output
    else:
        return '#FFFAFA	'  # Pale lilac – intermediate


def trace(root):
    """
    Trace through the computational graph starting from root node.
    Returns all nodes and edges in the graph.
    
    This function performs a depth-first traversal of the computational graph,
    collecting all Value nodes and the edges between them. This is essential
    for visualization because we need to see the complete structure of how
    values depend on each other.
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)  # Recursive traversal
    
    build(root)
    return nodes, edges

def draw_dot(root, show_gradients=True, precision=4):
    """
    Create a graphviz visualization of the computational graph.
    
    Parameters:
    - root: The root Value node to start visualization from
    - show_gradients: Whether to display gradient values (useful after backward())
    - precision: Number of decimal places to show for data/grad values
    
    Returns:
    - Digraph object that can be rendered or saved
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        
        # Create label with data and optionally gradients
        if show_gradients:
            format_str = "{%s | data %." + str(precision) + "f | grad %." + str(precision) + "f}"
            label = format_str % (n.label or 'Value', n.data, n.grad)
        else:
            format_str = "{%s | data %." + str(precision) + "f}"
            label = format_str % (n.label or 'Value', n.data)
        
        # Create a rectangular node for each Value
        dot.node(
            name=uid,
            label=label,
            shape='record',
            fillcolor=get_node_color(n, nodes),
            style='filled',
            fontname='Helvetica',
            fontsize='10'
        )


        
        # If this value resulted from an operation, create an operation node
        if n._op:
            op_uid = uid + n._op
            
            activation_color = '#D1C4E9'  # Unified soft lavender for all activations

            op_colors = {
                '+': '#A8D5BA',        # Soft green – Addition
                '*': '#F7B2B7',        # Soft red/pink – Multiplication
                '**': '#FFD59E',       # Pastel orange – Power
                'tanh': activation_color,
                'relu': activation_color,
                'sigmoid': activation_color,
                'exp': activation_color,
                'log': activation_color
            }



            
            # Handle clamp operations and other complex ops
            op_color = 'white'  # default
            op_label = n._op
            
            if n._op.startswith('clamp'):
                op_color = 'orange'
            elif n._op.startswith('**'):
                op_color = op_colors.get('**', 'white')
                op_label = 'pow'  # Cleaner display
            else:
                op_color = op_colors.get(n._op, 'white')
            
            dot.node(
                name=op_uid,
                label=op_label,
                shape='ellipse',
                fillcolor=op_color,
                style='filled',
                fontsize='10'
            )
            
            # Connect operation node to result value
            dot.edge(op_uid, uid)
    
    # Add edges between Value nodes through operation nodes
    for n1, n2 in edges:
        if n2._op:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        else:
            # Direct edge for nodes without operations (shouldn't happen normally)
            dot.edge(str(id(n1)), str(id(n2)))
    
    return dot

def visualize_computation(root, filename=None, show_gradients=True, view=True, inline=None):
    """
    Convenience function to create and optionally save/view the computational graph.
    
    Parameters:
    - root: Root Value node to visualize
    - filename: If provided, save the graph to this file (without extension)
    - show_gradients: Whether to show gradient values
    - view: Whether to automatically open the visualization
    - inline: If True, display inline in Jupyter. If None, auto-detect Jupyter environment
    
    Example usage:
    >>> x = Value(2.0, label='x')
    >>> y = Value(3.0, label='y') 
    >>> z = x * y + x
    >>> z.backward()
    >>> visualize_computation(z, show_gradients=True, inline=True)  # For Jupyter
    """
    dot = draw_dot(root, show_gradients=show_gradients)
    
    # Auto-detect if we're in a Jupyter environment
    if inline is None:
        try:
            from IPython import get_ipython
            inline = get_ipython() is not None
        except ImportError:
            inline = False
    
    # Handle inline display for Jupyter notebooks
    if inline:
        try:
            from IPython.display import SVG, display
            # Render to SVG string and display inline
            svg_string = dot.pipe(format='svg')
            display(SVG(svg_string))
            return dot
        except ImportError:
            print("IPython not available. Falling back to file output.")
            inline = False
    
    # Original file-based visualization
    if filename:
        dot.render(filename, view=view, cleanup=True)
        print(f"Graph saved as {filename}.svg")
    
    if view and not filename and not inline:
        dot.view(cleanup=True)
    
    return dot

def compare_before_after_backward(root, inline=None):
    """
    Show the computational graph before and after calling backward().
    This is great for understanding how gradients flow through the graph.
    
    Parameters:
    - root: Root Value node (before backward() has been called)
    - inline: If True, display inline in Jupyter. If None, auto-detect
    """
    # Auto-detect Jupyter environment
    if inline is None:
        try:
            from IPython import get_ipython
            inline = get_ipython() is not None
        except ImportError:
            inline = False
    
    if inline:
        try:
            from IPython.display import SVG, display, HTML
            
            print("Before backward():")
            dot_before = draw_dot(root, show_gradients=False)
            svg_before = dot_before.pipe(format='svg')
            display(SVG(svg_before))
            
            print("\nCalling backward()...")
            root.backward()
            
            print("After backward():")
            dot_after = draw_dot(root, show_gradients=True)
            svg_after = dot_after.pipe(format='svg')
            display(SVG(svg_after))
            
            return dot_before, dot_after
            
        except ImportError:
            print("IPython not available. Falling back to file output.")
            inline = False
    
    # Original file-based approach
    if not inline:
        print("Creating visualization before backward()...")
        dot_before = draw_dot(root, show_gradients=False)
        dot_before.render('before_backward', view=False, cleanup=True)
        
        print("Calling backward()...")
        root.backward()
        
        print("Creating visualization after backward()...")
        dot_after = draw_dot(root, show_gradients=True)
        dot_after.render('after_backward', view=False, cleanup=True)
        
        print("Visualizations saved as 'before_backward.svg' and 'after_backward.svg'")
        print("You can open these files to see how gradients were computed!")
        
        return dot_before, dot_after

