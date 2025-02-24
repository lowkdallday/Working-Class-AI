import cv2
import mss
import numpy as np

class RoomScanner:
    def __init__(self):
        self.enemy_detected = False
        self.ammo_detected = False
        self.room_data = None

    def scan_for_enemies(self):
        """Simulate scanning for enemies."""
        self.enemy_detected = random.choice([True, False])
        logging.info(f"Enemy detected: {self.enemy_detected}")

    def scan_for_ammo(self):
        """Simulate scanning for ammo."""
        self.ammo_detected = random.choice([True, False])
        logging.info(f"Ammo detected: {self.ammo_detected}")

    def update_room_data(self, room_data):
        """Store room data (e.g., obstacles, enemies)."""
        self.room_data = room_data
