import pandas as pd
import platform

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"
else:
    webDriveFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/"
    outputFolder = "/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf/output/"

if __name__ == "__main__":
    print("This is working")
    file = pd.read_csv(webDriveFolder + "feature_sets/test_seq_128.csv")
    print(file.head(1))
    file.head().to_csv(outputFolder + 'file.csv')