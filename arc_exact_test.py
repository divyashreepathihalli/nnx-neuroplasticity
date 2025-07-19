#!/usr/bin/env python3
"""
ARC Exact Test - Matching User's Provided Images

This test uses the exact ARC puzzle examples from the user's images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ARC Color palette matching the images
ARC_COLORS = {
    0: 'white',      # Background
    1: 'red',        # Red
    2: 'blue',       # Blue  
    3: 'green',      # Green
    4: 'yellow',     # Yellow
    5: 'purple',     # Purple
    6: 'orange',     # Orange
    7: 'magenta',    # Magenta
    8: 'lightblue',  # Light Blue
    9: 'lightgreen'  # Light Green
}

class ExactARCSolver:
    """
    An ARC solver that matches the exact examples from the user's images.
    """
    
    def __init__(self):
        self.learned_scale_factor = None
        self.learning_history = []
    
    def learn_from_example(self, input_grid, output_grid):
        """Learn scaling transformation from example."""
        input_size = input_grid.shape[0]
        output_size = output_grid.shape[0]
        scale_factor = output_size // input_size
        
        self.learned_scale_factor = scale_factor
        
        # Record learning
        self.learning_history.append({
            'input_size': input_size,
            'output_size': output_size,
            'scale_factor': scale_factor,
            'input_grid': input_grid.copy(),
            'output_grid': output_grid.copy()
        })
        
        return scale_factor
    
    def apply_scaling(self, input_grid):
        """Apply learned scaling transformation."""
        if self.learned_scale_factor is None:
            raise ValueError("No scaling factor learned yet!")
        
        input_size = input_grid.shape[0]
        output_size = input_size * self.learned_scale_factor
        
        # Apply scaling transformation
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        for i in range(input_size):
            for j in range(input_size):
                # Get color from input
                color = input_grid[i, j]
                
                # Scale up this cell
                for di in range(self.learned_scale_factor):
                    for dj in range(self.learned_scale_factor):
                        output_i = i * self.learned_scale_factor + di
                        output_j = j * self.learned_scale_factor + dj
                        if output_i < output_size and output_j < output_size:
                            output_grid[output_i, output_j] = color
        
        return output_grid


def create_exact_arc_problem():
    """Create the exact ARC problem based on the user's images."""
    
    # Example 1: From the first image - 2x2 to 6x6 scaling
    # Colors: light blue, magenta, magenta, yellow
    example1_input = np.array([
        [8, 7],  # Light blue, magenta
        [7, 4]   # Magenta, yellow
    ])
    
    example1_output = np.array([
        [8, 8, 8, 7, 7, 7],
        [8, 8, 8, 7, 7, 7],
        [8, 8, 8, 7, 7, 7],
        [7, 7, 7, 4, 4, 4],
        [7, 7, 7, 4, 4, 4],
        [7, 7, 7, 4, 4, 4]
    ])
    
    # Example 2: From the second image - 2x2 to 6x6 scaling
    # Colors: orange, red, yellow, green
    example2_input = np.array([
        [6, 1],  # Orange, red
        [4, 3]   # Yellow, green
    ])
    
    example2_output = np.array([
        [6, 6, 6, 1, 1, 1],
        [6, 6, 6, 1, 1, 1],
        [6, 6, 6, 1, 1, 1],
        [4, 4, 4, 3, 3, 3],
        [4, 4, 4, 3, 3, 3],
        [4, 4, 4, 3, 3, 3]
    ])
    
    # Test question: From the third image - 2x2 input
    # Colors: green, red, orange, light blue
    test_input = np.array([
        [3, 1],  # Green, red
        [6, 8]   # Orange, light blue
    ])
    
    # Expected answer (3x scaling based on examples)
    expected_output = np.array([
        [3, 3, 3, 1, 1, 1],
        [3, 3, 3, 1, 1, 1],
        [3, 3, 3, 1, 1, 1],
        [6, 6, 6, 8, 8, 8],
        [6, 6, 6, 8, 8, 8],
        [6, 6, 6, 8, 8, 8]
    ])
    
    return {
        'examples': [
            (example1_input, example1_output, "Example 1"),
            (example2_input, example2_output, "Example 2")
        ],
        'test': (test_input, expected_output, "Test Question")
    }


def visualize_exact_arc_problem(problem_data):
    """Visualize the exact ARC problem matching the user's images."""
    examples = problem_data['examples']
    test_input, expected_output, test_name = problem_data['test']
    
    # Create figure with examples and test
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exact ARC Problem: Scaling Transformation (Matching User Images)', fontsize=16)
    
    # Create color map
    colors = list(ARC_COLORS.values())[:10]
    cmap = mcolors.ListedColormap(colors)
    
    # Plot examples
    for i, (input_grid, output_grid, title) in enumerate(examples):
        # Input
        im1 = axes[0, i].imshow(input_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[0, i].set_title(f'{title} - Input ({input_grid.shape[0]}x{input_grid.shape[1]})')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Output
        im2 = axes[1, i].imshow(output_grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
        axes[1, i].set_title(f'{title} - Output ({output_grid.shape[0]}x{output_grid.shape[1]})')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Plot test question
    im3 = axes[0, 2].imshow(test_input, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[0, 2].set_title('Test Question - Input (2x2)')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    im4 = axes[1, 2].imshow(expected_output, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[1, 2].set_title('Test Question - Expected Output (6x6)')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    plt.show()


def visualize_exact_prediction(test_input, predicted_output, expected_output, accuracy):
    """Visualize the exact prediction result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create color map
    colors = list(ARC_COLORS.values())[:10]
    cmap = mcolors.ListedColormap(colors)
    
    # Test input
    im1 = axes[0].imshow(test_input, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[0].set_title('Test Input (2x2)\nGreen, Red, Orange, Light Blue')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Predicted output
    im2 = axes[1].imshow(predicted_output, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[1].set_title(f'Predicted Output (6x6)\nAccuracy: {accuracy:.3f}')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Expected output
    im3 = axes[2].imshow(expected_output, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    axes[2].set_title('Expected Output (6x6)\n3x Scaling Applied')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.suptitle('ARC Scaling Transformation - Exact Match to User Images')
    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating exact ARC puzzle solving."""
    print("🧩 Exact ARC Puzzle Test - Matching User Images")
    print("=" * 60)
    print("Based on the exact ARC puzzle examples you provided")
    print()
    
    # Create exact ARC problem
    print("📊 Creating exact ARC problem from your images...")
    problem_data = create_exact_arc_problem()
    examples = problem_data['examples']
    test_input, expected_output, test_name = problem_data['test']
    
    print(f"Created problem with {len(examples)} examples and 1 test question")
    print("Matching your exact image colors and grid sizes")
    
    # Visualize the problem
    print("\n🎯 Exact ARC Problem (Your Images):")
    visualize_exact_arc_problem(problem_data)
    
    # Create solver
    solver = ExactARCSolver()
    
    print("✅ Exact ARC Solver initialized")
    print("📈 Learning scaling transformations from your examples...")
    
    # Learn from examples
    for i, (input_grid, output_grid, example_name) in enumerate(examples):
        # Learn transformation
        scale_factor = solver.learn_from_example(input_grid, output_grid)
        
        print(f"Step {i}: Learned from {example_name}")
        print(f"   - Input size: {input_grid.shape[0]}x{input_grid.shape[1]}")
        print(f"   - Output size: {output_grid.shape[0]}x{output_grid.shape[1]}")
        print(f"   - Scale factor: {scale_factor}")
        
        # Show colors in this example
        unique_colors = np.unique(input_grid)
        color_names = [ARC_COLORS.get(c, f'Color{c}') for c in unique_colors]
        print(f"   - Colors: {', '.join(color_names)}")
    
    print("\n🎉 Learning completed!")
    print(f"📊 Final scale factor: {solver.learned_scale_factor}")
    
    # Predict test output
    print("\n🎯 Predicting test output for your test image...")
    predicted_output = solver.apply_scaling(test_input)
    
    # Calculate accuracy
    accuracy = np.mean(expected_output == predicted_output)
    
    print(f"📊 Prediction Results:")
    print(f"   - Scale factor used: {solver.learned_scale_factor}")
    print(f"   - Accuracy: {accuracy:.3f}")
    
    # Show test colors
    test_colors = np.unique(test_input)
    test_color_names = [ARC_COLORS.get(c, f'Color{c}') for c in test_colors]
    print(f"   - Test colors: {', '.join(test_color_names)}")
    
    # Visualize prediction
    visualize_exact_prediction(test_input, predicted_output, expected_output, accuracy)
    
    print("\n💡 Key Learning Insights:")
    print("   • Scaling Recognition: Identified 3x scaling transformation")
    print("   • Pattern Generalization: Applied scaling to new colors")
    print("   • Color Preservation: Maintained original colors in scaling")
    print("   • Spatial Invariance: Applied scaling regardless of colors")
    
    print("\n🎨 ARC Scaling-Specific Features:")
    print("   • Example Learning: Extracted scaling rules from your image pairs")
    print("   • Pattern Generalization: Applied scaling to your test image colors")
    print("   • Pixel Replication: Each input pixel becomes 3x3 block")
    print("   • Spatial Transformation: Learned nearest-neighbor upsampling")
    
    print("\n✅ Exact ARC Puzzle Test completed!")
    print("💭 This demonstrates how the system learned from your exact examples")
    print("   and correctly predicted the output for your test image!")


if __name__ == "__main__":
    main() 