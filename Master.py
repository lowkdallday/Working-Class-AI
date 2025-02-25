import time
import logging
from pathfinding import Pathfinding
from Modules.room_scanner import RoomScanner
from Modules.ammo_handler import AmmoHandler
from Modules.movement import Movement
from Modules.micro_ai import MicroAI
from Modules.daredevil_ai import DaredevilAI

# Logging setup
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
        self.micro_ai = MicroAI()
        self.daredevil_ai = DaredevilAI()
        self.health = 100
        self.ammo = 100
        self.points = 500  # Starting points for buying weapons and perks

    def take_control(self):
        self.state = "Player"
        logging.info("Frank Castle has taken control!")
        self.play_game()

    def play_game(self):
        """Main loop for Frank's actions."""
        while True:
            self.scan_environment()
            self.make_decision()
            time.sleep(0.1)  # Smooth game updates

    def scan_environment(self):
        """Gather data on surroundings."""
        self.room_scanner.scan_for_enemies()
        self.room_scanner.scan_for_ammo()
        self.micro_ai.assess_ammo(self.ammo_handler.ammo)
        self.daredevil_ai.detect_threats(self.room_scanner.room_data)

    def make_decision(self):
        """Decision-making based on AI reports."""
        if self.daredevil_ai.threat_detected:
            self.daredevil_ai.combat_advice()
            self.movement.move_to_enemy()
        elif self.micro_ai.ammo_strategy == 'conservative':
            self.micro_ai.suggest_tactical_move()
            self.movement.move_tactically()
        else:
            self.movement.move_randomly()

if __name__ == "__main__":
    frank_ai = FrankAI()
    frank_ai.take_control()
