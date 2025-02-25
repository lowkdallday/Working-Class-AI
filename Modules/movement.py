import random
import logging
from pynput import keyboard

class Movement:
    def __init__(self):
        self.keyboard = keyboard.Controller()

    def move_randomly(self):
        """Move randomly for exploration."""
        direction = random.choice(['w', 'a', 's', 'd'])
        self.keyboard.press(direction)
        logging.info(f"Moving in direction: {direction}")
        time.sleep(0.5)
        self.keyboard.release(direction)

    def move_to_enemy(self):
        """Move towards the detected enemy."""
        logging.info("Moving towards enemy.")
        # In actual code, this should be replaced with more accurate movement logic
        self.move_randomly()  # Placeholder for now
