# src/hello.py

class HelloUCF:
    def __init__(self, name="UCF101 Project"):
        self.name = name

    def print_hello(self):
        print(f"Hello from {self.name}!")
        print("src/ folder loaded successfully into Colab.")
