import sys
import os
from datetime import datetime


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


output_dir = "model_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def log_output():
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"bike_rental_results_{timestamp}.txt")

    # Set up the logger
    sys.stdout = Logger(output_filename)

    # Print header with timestamp
    print(
        f"Bike Rental Model Comparison - Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("-" * 80)
