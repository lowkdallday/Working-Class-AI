import logging

class DaredevilAI:
    def __init__(self):
        self.threat_detected = False

    def detect_threats(self, room_data):
        """Detect threats based on room data (e.g., enemy count or sounds)."""
        self.threat_detected = random.choice([True, False])  # Simulate threat detection
        if self.threat_detected:
            logging.info("Daredevil: Threat detected, preparing for combat.")
        else:
            logging.info("Daredevil: No immediate threats detected.")
    
    def combat_advice(self):
        """Daredevil provides combat advice."""
        if self.threat_detected:
            logging.info("Daredevil suggests: Use close combat tactics, stay agile.")
        else:
            logging.info("Daredevil suggests: Continue scanning for threats.")
