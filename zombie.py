import time
import threading
import queue
import keyboard
import pyautogui
import cv2
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
import mss  # For screen capture

# Game State for player and enemy positions
class GameState:
    def __init__(self):
        self.player_position = [0, 0]  # Example: player position
        self.enemy_positions = [[5, 5], [10, 10]]  # Example: enemy positions
        self.action_space = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot']
        
    def get_possible_actions(self):
        return self.action_space  # Returning the possible actions
        
    def apply_action(self, action):
        # This applies the action and returns the new state
        new_state = GameState()  # A new instance of GameState
        new_state.player_position = self.player_position.copy()
        new_state.enemy_positions = [pos.copy() for pos in self.enemy_positions]
        
        if action == 'move_up':
            new_state.player_position[1] += 1
        elif action == 'move_down':
            new_state.player_position[1] -= 1
        elif action == 'move_left':
            new_state.player_position[0] -= 1
        elif action == 'move_right':
            new_state.player_position[0] += 1
        elif action == 'shoot':
            # Simulate shooting action
            print("Frank shoots!")
            pyautogui.click()  # Simulate mouse click for shooting
        
        return new_state

# RL Model (adjusted input size to 2 features)
class RLModel(nn.Module):
    def __init__(self):
        super(RLModel, self).__init__()
        # Input size is 2 (player position), output size is 5 (number of actions)
        self.fc1 = nn.Linear(2, 128)  # Input size is now 2
        self.fc2 = nn.Linear(128, 5)  # Output size is 5 (number of actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# AI Agent for advising on actions
class AIAgent:
    def __init__(self, model):
        self.model = model

    def advise(self, state):
        # Convert state to tensor with proper shape (1, 2)
        state_tensor = torch.tensor([state.player_position], dtype=torch.float32)
        action_values = self.model(state_tensor)
        action = torch.argmax(action_values).item()
        return action

# Frank control mechanism using keyboard and mouse
class FrankControl:
    def __init__(self, state):
        self.state = state
        self.controlled_by_frank = True  # Frank takes control immediately
        self.key_mapping = {
            'w': 'move_up',
            's': 'move_down',
            'a': 'move_left',
            'd': 'move_right',
            'space': 'shoot',
        }
        self.ai_agent = AIAgent(RLModel())  # Initialize AI agent with the RL model

    def ai_move(self):
        # Let AI advise on the action based on the current state
        action = self.ai_agent.advise(self.state)
        self.state = self.state.apply_action(self.state.get_possible_actions()[action])
        return self.state

    def control(self):
        # Frank controls the game directly via keyboard and mouse
        if self.controlled_by_frank:
            action = None
            if keyboard.is_pressed('w'):  # Move Up
                action = self.key_mapping['w']
            elif keyboard.is_pressed('s'):  # Move Down
                action = self.key_mapping['s']
            elif keyboard.is_pressed('a'):  # Move Left
                action = self.key_mapping['a']
            elif keyboard.is_pressed('d'):  # Move Right
                action = self.key_mapping['d']
            elif keyboard.is_pressed('space'):  # Shoot
                action = self.key_mapping['space']

            if action:
                self.state = self.state.apply_action(action)
                return self.state
        return self.state

# Overlay chat window setup
class ChatOverlay:
    def __init__(self, message_queue):
        self.root = tk.Tk()
        self.root.title("Game Control Console")
        self.root.attributes("-topmost", True)  # Always on top
        self.root.attributes("-transparentcolor", "white")  # Transparent background
        self.root.attributes("-fullscreen", True)  # Fullscreen overlay
        self.root.overrideredirect(True)  # Remove window decorations

        # Create a frame for the chat window
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a scrolled text area
        self.text_area = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, width=70, height=20, bg="black", fg="white")
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "Game Control Console\n")

        # Message queue for thread-safe updates
        self.message_queue = message_queue

    def update_chat(self):
        # Check for new messages in the queue
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.text_area.insert(tk.END, message + "\n")
            self.text_area.yview(tk.END)  # Auto scroll to the bottom
        self.root.after(100, self.update_chat)  # Schedule the next update

    def run(self):
        self.update_chat()  # Start the update loop
        self.root.mainloop()

# Screen capture using OpenCV
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor
        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow("Screen Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        cv2.destroyAllWindows()

# Display conversation in the chat window
def display_conversation(message_queue):
    conversations = [
        "Daredevil: 'Frank, it's your time now. Take control.'",
        "Micro: 'You got this, Frank. Move fast, stay sharp.'",
        "Frank: 'Alright, here I go...'",
        "Daredevil: 'Remember, Frank, they won’t give you an inch. Be relentless!'",
        "Micro: 'Frank, the enemies are closing in!'",
    ]
    
    for line in conversations:
        message_queue.put(line)  # Add message to the queue
        time.sleep(2)  # 2-second delay between each message

# Main function to start the game and control by Frank
def main():
    state = GameState()
    frank_control = FrankControl(state)

    # Create a thread-safe message queue
    message_queue = queue.Queue()

    # Create and run the chat overlay
    chat_overlay = ChatOverlay(message_queue)
    chat_thread = threading.Thread(target=chat_overlay.run)
    chat_thread.daemon = True  # Ensure the thread exits when the main program exits
    chat_thread.start()

    # Start screen capture
    screen_thread = threading.Thread(target=capture_screen)
    screen_thread.daemon = True
    screen_thread.start()

    # Display the conversation
    conversation_thread = threading.Thread(target=display_conversation, args=(message_queue,))
    conversation_thread.daemon = True
    conversation_thread.start()

    # Game loop
    while True:
        try:
            # Let Frank control the game
            state = frank_control.control()
            message_queue.put(f"Current state: {state.player_position}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

# Run the game
if __name__ == "__main__":
    main()