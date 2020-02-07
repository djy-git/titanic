from env import *


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_PROC_CSV_PATH)
    y = train_data['Survived']
    X = train_data.drop(columns='Survived')

    # Model selection
