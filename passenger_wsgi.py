import os
import sys

# Add the directory containing your predict.py to the system path
sys.path.insert(0, os.path.dirname(__file__))

# Import the 'application' variable from your predict.py file
from predict import application