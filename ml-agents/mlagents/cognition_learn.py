"""
CognitionLearn main entry point module
This provides the main entry point for the CognitionLearn GUI application
"""
from mlagents.trainers.learn import main as learn_main

def main():
    """Main entry point for CognitionLearn"""
    learn_main()

if __name__ == "__main__":
    main()