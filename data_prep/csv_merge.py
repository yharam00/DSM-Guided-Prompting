import pandas as pd

df1 = pd.read_csv('data/daic_woz/gpt4o_depression/experiment_5_train_DAIC_train.csv')
df2 = pd.read_csv('data/results/human_selected/e_daic_transcript_train_experiment_5_depression_gpt-4o_expand.csv')


df = pd.concat([df1, df2])

df.to_csv('data/daic_woz/gpt4o_depression/experiment_5_train_final.csv', index=False)
