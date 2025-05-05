import torch
import torch.nn as nn
import json
import argparse
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from google import genai
from google.genai import types
from IPython.display import HTML,Markdown,display
from google.api_core import retry
from EightPuzzle_model import EightPuzzleMLP
import matplotlib.pyplot as plt


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



client = genai.Client(api_key=GEMINI_API_KEY)


is_retriable=lambda e: (isinstance(e,genai.errors.APIError) and e.code in {429,503})
genai.models.Models.generate_content=retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='8-Puzzle Solver using NN and LLM')
    parser.add_argument('--model_weights', nargs='+', type=str, required=True, 
                    help='Path(s) to the neural network weights (.pth files)')
    parser.add_argument('--initial_state', type=str, required=True,
                        help='Path to the image of the initial state')
    parser.add_argument('--goal_state', type=str, required=True,
                        help='Path to the image of the goal state')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to the model configuration file')
    return parser.parse_args()

def load_model(weights_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = EightPuzzleMLP(config)
    
    # Load the checkpoint
    checkpoint = torch.load(weights_path)
    
    # Check if the checkpoint contains a model_state_dict key
    if 'model_state_dict' in checkpoint:
        # Option 1: Try to load with strict=False to ignore mismatched keys
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Option 2: Try to load the checkpoint directly with strict=False
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess the image for the neural network using similar transforms as in your code"""
    image = Image.open(image_path).convert('L')  
    
    
    preprocess = transforms.Compose([
        transforms.Resize((84, 84)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])
    
    return preprocess(image).unsqueeze(0)  

def predict_puzzle_state(model, image_tensor):
    """Use the model to predict the puzzle state"""
    with torch.no_grad():
        output = model(image_tensor)
    
    
    
    predicted_digits = torch.argmax(output, dim=2)[0].reshape(3, 3).tolist()
    
    return predicted_digits

def format_puzzle_state(state):
    """Format the puzzle state for display"""
    formatted = []
    for row in state:
        formatted.append(" ".join(map(str, row)))
    return "\n".join(formatted)

def visualize_puzzle(state):
    """Create a simple text visualization of the puzzle state"""
    result = "-" * 13 + "\n"
    for row in state:
        result += "| "
        for cell in row:
            if cell == 0:
                result += "  | "  
            else:
                result += f"{cell} | "
        result += "\n" + "-" * 13 + "\n"
    return result




def visualize_states(states, output_dir, checkpoint_name):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for each state
    num_states = len(states)
    fig_width = min(20, num_states * 3)  # Limit width for many states
    
    fig, axes = plt.subplots(1, num_states, figsize=(fig_width, 3))
    
    # Handle case where there's only one state
    if num_states == 1:
        axes = [axes]
    
    # Plot each state
    for i, state in enumerate(states):
        # Extract the configuration from the state dictionary
        config = state["configuration"]  # This is the key change
        
        # Convert state to 3x3 grid
        grid = np.array(config).reshape(3, 3)
        
        # Plot the grid
        ax = axes[i]
        ax.imshow(grid, cmap='Blues', vmin=0, vmax=8)
        
        # Add text labels
        for r in range(3):
            for c in range(3):
                val = grid[r, c]
                ax.text(c, r, str(val) if val != 0 else '', 
                       ha='center', va='center', fontsize=20, color='black')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {i}")
    
    # Add overall title
    plt.suptitle(f"Solution Path - {os.path.basename(checkpoint_name)}")
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(output_dir, f"solution_path_{os.path.basename(checkpoint_name).split('.')[0]}.png")
    plt.savefig(filename)
    plt.close()
    
    print(f"Visualization saved to {filename}")


def call_gemini_llm(initial_state, goal_state):
    """Call the Gemini API to solve the 8-puzzle"""
    
    initial_formatted = format_puzzle_state(initial_state)
    goal_formatted = format_puzzle_state(goal_state)
    
    initial_visual = visualize_puzzle(initial_state)
    goal_visual = visualize_puzzle(goal_state)
    
    prompt = f"""
    I need you to solve a modified 8-puzzle problem using the A* search algorithm with Manhattan distance heuristic in the most optimal way possible.
    The 8-puzzle consists of a 3x3 grid with 8 numbered tiles and one blank space. The goal is to move the tiles around until they are in a specific order.
    The tiles can be moved into the blank space, and the goal is to find the shortest sequence of moves that leads from the initial state to the goal state.
    The initial and goal states are represented as 3x3 matrices, with numbers 1-8 and 0 representing the blank space. 
    Valid moves are to slide tiles into the blank space (up, down, left, right) and each move will cost 1 unit.
    
    I will give you the initial and goal states of the puzzle, and I need you to provide the solution path from the initial state to the goal state.

    Initial state:
    {initial_state}
    
    Goal state:
    {goal_state}
    
    Please follow these steps:
    1. Use the A* search algorithm with Manhattan distance as the heuristic.
    2. Show ALL intermediate states at every step of the solution path from initial to goal
    3. For each state, include:
       - The current puzzle configuration
       - The move made to reach this state (except for the initial state)
       - The cost of the path taken to reach this state (number of moves)
       - The Manhattan distance from this state to the goal
       - The total cost (path cost + Manhattan distance)
    
    Format your response as a JSON-parseable list of states, with each state having these fields:
    - "configuration": The 3x3 grid as a list of lists
    - "move": The move made to reach this state (null for initial state)
    - "cost": The cost of the path taken to reach this state (number of moves)
    - "manhattan_distance": The Manhattan distance to the goal
    - "total_cost": The total cost (path cost + Manhattan distance)
    
    Note: 
    1. To calculate the  manhattan distance from the current state to the goal state:
    - For each tile (including 0), find its current position in the grid and its target position in the goal state.
    - The Manhattan distance for each tile is the sum of the absolute differences of their row and column indices.
    - The total Manhattan distance is the sum of the distances for all tiles.

    2. The general way the puzzle is solved is:
    - The algorithm starts with the initial state and explores all possible moves (up, down, left, right) to generate new states.
    - For each new state, it calculates the Manhattan distance to the goal state and adds it to the cost of the path taken to reach that state.
    - Then it selects the state with the lowest total cost (path cost + manhattan distance) to explore next.  
    - If all the possiible states have the same total cost, then thealgorithm will select any of them to explore next.
    - This process continues until the goal state is reached or all possible states have been explored.
    - You should Explore  all possible states at each depth but only return the states that are part of the solution path - meaning only the state that has least total cost.
    - The solution path should be the most optimal one, meaning it has the least number of moves to reach the goal state.

    Example format:
    suppose the initial state is:
    [[1, 8, 0], [4, 5, 3], [7, 6, 2]]
    and the goal state is:
    [[1, 8, 5], [7, 6, 4], [0, 2, 3]]]
    ```json
    [
      {{
        "configuration": [[1, 8, 0], [4, 5, 3], [7, 6, 2]],
        "move": null,
        "manhattan_distance": 10
      }},
      {{
        "configuration": [[1, 0, 8], [4, 5, 3], [7, 6, 2]],
        "move": "left",
        "manhattan_distance": 9
      }},
      ...
    ]
    ```
    
    Only include the JSON output and no other text. Make sure your JSON format is valid and can be parsed.
    """
    
    
    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt
    )
    
    return response.text

def parse_llm_response(response_text):
    """Parse the LLM response to get the states"""
    
    json_content = response_text.strip()
    
    
    if "```json" in json_content:
        json_content = json_content.split("```json")[1]
    elif "```" in json_content:
        json_content = json_content.split("```")[1]
    
    if "```" in json_content:
        json_content = json_content.split("```")[0]
    
    json_content = json_content.strip()
    
    try:
        states = json.loads(json_content)
        return states
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response content (first 200 chars): {json_content[:200]}...")
        
        
        try:
            
            
            import re
            json_match = re.search(r'(\[.*\])', json_content, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group(1)
                return json.loads(cleaned_json)
        except Exception:
            pass
            
        return []
def main():
    
    args = parse_arguments()
    
    
    for weight_path in args.model_weights:
        print(f"\n--- Processing checkpoint: {weight_path} ---")
        model = load_model(weight_path, args.config)
        
        # Process images and get predictions
        initial_image = preprocess_image(args.initial_state)
        goal_image = preprocess_image(args.goal_state)
        
        # Predict puzzle states (uncomment these and remove hardcoded states)
        initial_state = predict_puzzle_state(model, initial_image)
        goal_state = predict_puzzle_state(model, goal_image)
    
        print("Predicted Initial State:")
        print(initial_state)
        print("Predicted Goal State:")
        print(goal_state)
        print("Recognized Initial State:")
        print(visualize_puzzle(initial_state))
        print("\nRecognized Goal State:")
        print(visualize_puzzle(goal_state))
        
        
        print("\nSolving puzzle with Gemini LLM...")
        llm_response = call_gemini_llm(initial_state, goal_state)
        
        
        states = parse_llm_response(llm_response)
        
        if states:
            with open('states.json', 'w') as f:
                json.dump(states, f, indent=2)
            print(f"Solution saved to states.json with {len(states)} states")
        else:
            print("Failed to generate a valid solution from LLM response")
            print("Saving raw LLM response for debugging...")
            with open('llm_response_raw.txt', 'w') as f:
                f.write(llm_response)
            print("Raw response saved to llm_response_raw.txt")

        with open('llm_response_raw.txt', 'w') as f:
            f.write(llm_response)

        checkpoint_name = os.path.basename(weight_path).split('.')[0]
        states_filename = f'states_{checkpoint_name}.json'
        with open(states_filename, 'w') as f:
            json.dump(states, f, indent=2)
        print(f"Solution saved to {states_filename} with {len(states)} states")
        visualize_states(states, "visualizations", weight_path)

if __name__ == "__main__":
    main()