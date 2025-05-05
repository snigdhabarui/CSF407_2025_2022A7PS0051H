# 8-Puzzle Problem Solver using Neural Networks and LLM (Gemini)

## Project Description

This project solves the **8-puzzle problem** using a hybrid approach:
- **Neural Network (NN)** for interpreting puzzle configurations from images.
- **Large Language Model (LLM)** (Google's Gemini) to simulate reasoning and solve the puzzle using an A*-like search algorithm with a Manhattan distance heuristic.

Instead of coding a traditional A* solver, we delegate the search reasoning to the LLM. The entire flow is automated from image input to state path output.

---

## What is the 8-Puzzle?

- It consists of a 3×3 grid containing numbers 1–8 and one empty tile (0).
- The goal is to transform a given **initial configuration** to a **goal configuration** by sliding tiles.
- Only one tile can be moved into the blank space at a time.

---
### Code Workflow
1. Argument Parsing
The program uses the argparse library to handle command-line arguments, allowing flexible usage from the terminal.

2. Loading the Neural Network
The NN is defined as a Multi-Layer Perceptron (MLP) model. It loads weights from the checkpoint file and switches to evaluation mode.

3. Image Preprocessing
Images of the puzzle state are converted to grayscale, resized to 84x84 pixels, normalized, and then converted to a tensor format compatible with PyTorch.

4. Predicting the Puzzle State
The model takes the preprocessed tensor and outputs a probability distribution over digits (0-8) for each of the 9 tiles. The highest probability digit is chosen per tile using argmax, then reshaped into a 3x3 grid.

5. Visualizing Predicted State
The 3x3 puzzle grid is displayed in a human-readable format using characters (like borders and spacing).

6. Calling the LLM (Gemini)
The LLM prompt:

Describes the 8-puzzle and how it works.

Requests solving using A* with Manhattan distance.

Asks for a JSON-formatted response that includes:

Each intermediate state.

The move taken to reach the state.

The cost so far.

The Manhattan distance from the goal.

The total cost (cost + heuristic).

Gemini responds with a JSON-like list of all intermediate steps from start to finish.

7. Parsing the LLM Response
The JSON portion of the LLM's response is extracted. If the output is wrapped in markdown formatting (e.g., ```json), it is stripped out. The resulting string is parsed into Python objects using the json library. If parsing fails, a backup approach using regular expressions is used.

8. Output Handling
If the response is valid JSON, it is saved to states.json.

If not, the raw LLM output is saved to llm_response_raw.txt to inspect and debug the failure.
