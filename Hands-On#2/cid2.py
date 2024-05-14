import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys
import os


class Dataset:
    def __init__(self):
        self.X = []
        self.Y = []

    def receive_dataset(self, filename):
        try:
            df = pd.read_excel(filename)
            self.X = df['X'].tolist()
            self.Y = df['Y'].tolist()
        except FileNotFoundError:
            print("File not found!")
            return False
        return True

class SLRCalculator:
    def __init__(self, dataset):
        self.dataset = dataset

    def sum_X(self):
        return sum(self.dataset.X)

    def sum_Y(self):
        return sum(self.dataset.Y)

    def compute_XY(self):
        return [x * y for x, y in zip(self.dataset.X, self.dataset.Y)]

    def sum_XY(self):
        return sum(self.compute_XY())

    def square_X(self):
        return [x ** 2 for x in self.dataset.X]

    def sum_X_squared(self):
        return sum(self.square_X())

    def square_Y(self):
        return [y ** 2 for y in self.dataset.Y]

    def sum_Y_squared(self):
        return sum(self.square_Y())

    def calculate_coefficients(self):
        n = len(self.dataset.X)
        X = self.sum_X()
        Y = self.sum_Y()
        X2 = self.sum_X_squared()
        XY = self.sum_XY()
        Y2 = self.sum_Y_squared()

        B0 = (Y * X2 - X * XY) / (n * X2 - X ** 2)
        B1 = (n * XY - X * Y) / (n * X2 - X ** 2)

        numerator =  (n * XY) - (X * Y) 
        denominator = (((n*X2) - (X**2)) * ((n * Y2) - (Y ** 2))) ** (1/2)

        correlation = numerator/denominator
        determination = correlation **2
        


        return B0, B1,correlation, determination

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filename = filedialog.askopenfilename(initialdir=os.path.expanduser("~/Desktop"), title="Select File", filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
    return filename

def main():
    filename = select_file()
    if not filename:
        print("No file selected. Exiting.")
        return

    dataset_obj = Dataset()
    success = dataset_obj.receive_dataset(filename)
    if not success:
        return

    print("\nReceived dataset:")
    print("X:", dataset_obj.X)
    print("Y:", dataset_obj.Y)

    calculator = SLRCalculator(dataset_obj)
    B0, B1, correlation, determination = calculator.calculate_coefficients()

    print("\nCoefficients of Simple Linear Regression:")
    print("B0:", B0)
    print("B1:", B1)
    print("Correlation:", correlation)
    print("Determination:", determination)


    while True:
        x = input("\nEnter a value of 'x' to predict SALES(or 'done' to finish): ")
        if x.lower() == 'done':
            break
        try:
            x = float(x)
            prediction = B0 + B1 * x
            print("Predicted value for SALES =", x, ":", prediction)
        except ValueError:
            print("Invalid input! Please enter a numerical value or 'done'.")

if __name__ == "__main__":
    main()
