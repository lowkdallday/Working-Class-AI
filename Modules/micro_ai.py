import logging

class MicroAI:
    def __init__(self):
        self.ammo_strategy = "balanced"  # Can be 'aggressive', 'balanced', 'conservative'

    def assess_ammo(self, ammo_count):
        """Assess ammo status and provide recommendations."""
        if ammo_count < 20:
            self.ammo_strategy = 'conservative'
            logging.info("Micro: Ammo running low, switch to conservative tactics.")
        else:
            self.ammo_strategy = 'balanced'
            logging.info("Micro: Ammo sufficient, maintaining balanced tactics.")
    
    def suggest_tactical_move(self):
        """Micro's tactical move suggestion."""
        if self.ammo_strategy == 'conservative':
            logging.info("Micro suggests: Conserve ammo, avoid unnecessary fire.")
        else:
            logging.info("Micro suggests: Keep moving, stay aggressive.")
