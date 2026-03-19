import pandas as pd

def main():
    URL = 'https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/abalone.csv'
    df = pd.read_csv(URL)
    df['Sex'] = df['Sex'].replace('f', 'F')
    df.to_csv('abalone.csv', index=False)

if __name__ == "__main__":
    main()
