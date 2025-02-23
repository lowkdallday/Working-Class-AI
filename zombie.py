import random
import numpy as np
import time
import pyautogui
import cv2
import torch
import threading
from collections import deque
import keyboard
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from scipy.linalg import expm
import mss  # For screen capture
from transformers import GPTJForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

input_text = "Explain quantum computing in simple terms."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
# Define log file path
log_file_path = "C:\\Users\\Chandler\\Desktop\\logfrank.txt"

# Frank Castle: Main AI that learns and plays
class FrankCastle:
    def __init__(self):
        self.memory = deque(maxlen=100)
        self.name = "Frank Castle"
        self.mcts = MonteCarloTreeSearch()
        self.decision_tree = DecisionTree()
        self.micro = MicroAI()
        self.daredevil = DaredevilAI()
        self.qml = QuantumML()
        self.dialogue_history = deque(maxlen=10)  # Store last 10 dialogues

    def give_advice(self, state):
        health, ammo, zombie_count, closest_enemy_dist = state

        if random.random() < 0.5:  # Randomly choose advisor to talk (Micro or Daredevil)
            advisor = self.daredevil
        else:
            advisor = self.micro

        advice = advisor.give_advice(state)
        self.dialogue_history.append(f"{self.name}: {advice}")
        return advice

    def take_action(self, state, action_size):
        action = self.mcts.select_action(state, action_size)
        self.execute_action(action)

    def execute_action(self, action):
        if action == 0:  # Move forward
            pyautogui.keyDown("w")
            time.sleep(0.1)
            pyautogui.keyUp("w")
        elif action == 1:  # Move backward
            pyautogui.keyDown("s")
            time.sleep(0.1)
            pyautogui.keyUp("s")
        elif action == 2:  # Shoot
            pyautogui.keyDown("space")
            time.sleep(0.1)
            pyautogui.keyUp("space")
        elif action == 3:  # Look around
            pyautogui.moveTo(random.randint(0, 1920), random.randint(0, 1080))


class MonteCarloTreeSearch:
    def __init__(self, simulations=50):
        self.simulations = simulations
        self.tree = {}

    def simulate(self, state, action):
        reward = random.random()  # Placeholder reward function
        return reward

    def select_action(self, state, action_size):
        best_action = None
        best_reward = -float('inf')

        for _ in range(self.simulations):
            action = random.randint(0, action_size - 1)
            reward = self.simulate(state, action)

            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action


class DecisionTree:
    def __init__(self):
        self.tree = self.build_tree()

    def build_tree(self):
        return {
            "health": {
                "low": {
                    "ammo": {
                        "low": "avoid_combat",
                        "high": "find_health_pack"
                    }
                },
                "high": {
                    "ammo": {
                        "low": "find_ammo",
                        "high": "engage"
                    }
                }
            }
        }

    def decide_action(self, state):
        health, ammo, zombie_count, closest_enemy_dist = state
        decision = "default_action"

        if health < 20:
            decision = self.tree["health"]["low"]
            if ammo < 5:
                decision = self.tree["health"]["low"]["ammo"]["low"]
            else:
                decision = self.tree["health"]["low"]["ammo"]["high"]


        elif health >= 20:
            decision = self.tree["health"]["high"]
            if ammo < 5:
                decision = self.tree["health"]["high"]["ammo"]["low"]
            else:
                decision = self.tree["health"]["high"]["ammo"]["high"]


        return decision


class MicroAI:
    def __init__(self):
        self.model = self.create_model()
        self.memory = deque(maxlen=100)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs: avoid, find_health, engage
        )
        model.apply(self.init_weights)
        return model

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def give_advice(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        output = self.model(state_tensor)
        action = torch.argmax(output).item()

        if action == 0:
            return "Micro says: Avoid combat, find health packs immediately!"
        elif action == 1:
            return "Micro says: You need ammo, Frank. Find it now."
        else:
            return "Micro says: Keep it up, Frank. Don't stop fighting."


class DaredevilAI:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs: retreat, lead_choke_point, engage
        )
        model.apply(self.init_weights)
        return model

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def give_advice(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        output = self.model(state_tensor)
        action = torch.argmax(output).item()

        if action == 0:
            return "Daredevil advises: Retreat and take cover, Frank. Regroup."
        elif action == 1:
            return "Daredevil advises: Use your surroundings. Lead the zombies into a choke point."
        else:
            return "Daredevil advises: Engage them head-on, Frank. Stay sharp."


class QuantumML:
    def __init__(self):
        # Quantum machine learning model setup
        self.model = self.build_quantum_model()

    def build_quantum_model(self):
        # Placeholder function: Quantum machine learning should integrate quantum data.
        # For now, it's a basic model that simulates a quantum approach.
        model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs: simulate quantum decision-making
        )
        return model

    def simulate_quantum_decision(self, state):
        # Apply Quantum-like decision-making here (could involve quantum superposition, etc)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        output = self.model(state_tensor)
        action = torch.argmax(output).item()
        return action


class GameController:
    def __init__(self, frank):
        self.frank = frank

    def get_advice(self, state):
        return self.frank.give_advice(state)


def capture_screen():
    # Capture both screens using MSS (if available)
    with mss.mss() as sct:
        # Grab all monitors
        monitors = sct.monitors  # list of monitors, including primary and secondary
        screen_images = []

        for monitor in monitors[1:]:  # skip the first monitor (primary screen)
            screen = sct.grab(monitor)  # Capture screenshot of the current monitor
            screen_np = np.array(screen)  # Convert image to numpy array
            screen_images.append(screen_np)
        
        # Now, capture the primary monitor (main screen)
        primary_screen = sct.grab(sct.monitors[1])  # Monitor 1 is usually the primary screen
        primary_screen_np = np.array(primary_screen)
        screen_images.append(primary_screen_np)

        # Combine the images of both monitors into one
        full_screen_image = np.concatenate(screen_images, axis=1)  # Stack images horizontally

        # Convert to BGR (OpenCV format) for further processing
        full_screen_image = cv2.cvtColor(full_screen_image, cv2.COLOR_RGBA2BGR)
        return full_screen_image


def detect_zombies(screen):
    """Replace with YOLO or other advanced detection model."""
    # Load YOLOv3 weights and config files
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(screen, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    zombie_count = 0
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming '0' is zombie class in YOLO
                zombie_count += 1

    return zombie_count


def get_state(zombie_count):
    health = random.randint(0, 100)
    ammo = random.randint(0, 100)
    closest_enemy_dist = random.random()
    return [health, ammo, zombie_count, closest_enemy_dist]


def print_slow(text, delay=1.5):
    """Slowly print text for readability in console."""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay / 10)
    print()  # Ensure a newline after the message


def log_update(message):
    with open(log_file_path, 'a') as log_file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")


def draw_transparent_text(image, text, position, font, font_scale, font_color, thickness):
    """Draw text with a transparent background."""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.putText(image, text, (x, y + text_height), font, font_scale, font_color, thickness)


def passive_observation_loop(agent, action_size, game_controller):
    last_update_time = time.time()

    while True:
        if keyboard.is_pressed('space'):
            print("Manual control activated!")
            break  # Exit loop if 'space' is pressed

        # Capture screen and detect zombies
        screen = capture_screen()
        zombie_count = detect_zombies(screen)
        state = get_state(zombie_count)

        # Get advice from Frank Castle, Micro, and Daredevil
        frank_advice = game_controller.get_advice(state)
        print_slow(frank_advice)

        # Choose action based on MCTS or QuantumML (based on QML decision)
        decision = agent.qml.simulate_quantum_decision(state)

        # Log updates
        log_update(f"Frank's quantum decision: {decision}")

        # Draw transparent chat overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # White text
        thickness = 2

        # Draw each line of dialogue
        y = 30
        for line in agent.dialogue_history:
            draw_transparent_text(screen, line, (10, y), font, font_scale, font_color, thickness)
            y += 30

        # Display the screen with the chat overlay
        cv2.imshow("Frank's Dialogue", screen)
        cv2.moveWindow("Frank's Dialogue", 1920 - 400, 0)  # Position overlay on the top-right of the second screen
        cv2.waitKey(1)

        # Sleep for a bit before next update
        time.sleep(0.1)


# Initialize and start game controller loop
frank = FrankCastle()
game_controller = GameController(frank)

# Run the passive observation loop in a separate thread
observation_thread = threading.Thread(target=passive_observation_loop, args=(frank, 4, game_controller))
observation_thread.start()