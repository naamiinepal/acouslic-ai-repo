import pandas as pd
from sklearn.model_selection import train_test_split

stats_csv_path = 'data_stats/artifacts/ACOUSLIC_AI_data_stats.csv'


df = pd.read_csv(stats_csv_path)


# train val split
train_df , test_df = train_test_split(df,test_size=0.1,random_state=42)
assert len(train_df) == 270
assert len(test_df) == 30


train_df['filestem'].to_csv('data_stats/artifacts/train_test_split/ACOUSLIC_AI-train_split.csv',index=False)
test_df['filestem'].to_csv('data_stats/artifacts/train_test_split/ACOUSLIC_AI-test_split.csv',index=False)





