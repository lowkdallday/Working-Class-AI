class AmmoHandler:
    def __init__(self):
        self.ammo = 100
        self.low_ammo = False

    def check_ammo(self):
        """Check ammo status and update."""
        if self.ammo < 20:
            self.low_ammo = True
        else:
            self.low_ammo = False

    def reload_weapon(self):
        """Simulate reloading the weapon."""
        if self.low_ammo:
            logging.info("Reloading weapon...")
            self.ammo = 100  # Full reload
            self.low_ammo = False
