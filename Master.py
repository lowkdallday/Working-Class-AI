import time
import random
from modules.pathfinding import Pathfinding
from modules.room_scanner import RoomScanner
from modules.ammo_handler import AmmoHandler
from modules.movement import Movement
from modules.micro_ai import MicroAI
from modules.daredevil_ai import DaredevilAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.FileHandler("franklog.txt"),
    logging.FileHandler("masterlog.txt")
])

class FrankAI:
    def __init__(self):
        self.state = "Idle"
        self.pathfinder = Pathfinding(grid_size=10, obstacles=[(3, 3), (4, 4), (5, 5)])
        self.room_scanner = RoomScanner()
        self.ammo_handler = AmmoHandler()
        self.movement = Movement()
        self.micro_ai = MicroAI()  # Micro's AI
        self.daredevil_ai = DaredevilAI()  # Daredevil's AI
        self.max_health = 100
        self.health = self.max_health
        self.ammo = 100

    def take_control(self):
        self.state = "Player"
        logging.info("Frank Castle has taken control!")
        self.play_game()

    def play_game(self):
        """Main game loop where Frank plays."""
        while True:
            self.scan_environment()
            self.make_decision()

            time.sleep(0.1)  # Smooth action updates

    def scan_environment(self):
        """Scan the environment for enemies, ammo, and obstacles."""
        self.room_scanner.scan_for_enemies()
        self.room_scanner.scan_for_ammo()

        # Micro and Daredevil AI interactions
        self.micro_ai.assess_ammo(self.ammo_handler.ammo)
        self.daredevil_ai.detect_threats(self.room_scanner.room_data)

    def make_decision(self):
        """Make decisions based on reports from other modules."""
        if self.daredevil_ai.threat_detected:
            self.daredevil_ai.combat_advice()
            self.movement.move_to_enemy()
        elif self.micro_ai.ammo_strategy == 'conservative':
            self.micro_ai.suggest_tactical_move()
            self.movement.move_randomly()
        else:
            self.movement.move_randomly()

if __name__ == "__main__":
    frank_ai = FrankAI()
    frank_ai.take_control()
