import pandas as pd
import numpy as np
# df = pd.read_csv('FER/fer2013/fer2013.csv')
# df2 = pd.read_csv('FERPlus/fer2013new.csv')
# df2 = df2.rename(columns={"Usage": "Usage_2"})
# df_tot = pd.concat([df, df2], axis=1)
# print(df_tot["Usage_2"].eq(df_tot["Usage"]).all())
#
# df_train = df_tot[df_tot["Usage"] == "Training"]
# df_test = df_tot[df_tot["Usage"] == "PublicTest"]
# df_valid = df_tot[df_tot["Usage"] == "PrivateTest"]
#
# keep_cols = ['pixels',  'neutral','happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
# df_train = df_train[keep_cols]
# df_test = df_test[keep_cols]
# df_valid = df_valid[keep_cols]
#
# df_train['pixels'] = np.array(df_train['pixels'].map(lambda x: [int(y) for y in x.split(" ")]))
# df_test['pixels'] = np.array(df_test['pixels'].map(lambda x: [int(y) for y in x.split(" ")]))
# df_valid['pixels'] = np.array(df_valid['pixels'].map(lambda x: [int(y) for y in x.split(" ")]))
#
# df_train.to_csv("data/train.csv")
# df_test.to_csv("data/test.csv")
# df_valid.to_csv("data/valid.csv")

lab_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown',
            'NF']
for split in ['test','train','valid']:
    print(split)
    df = pd.read_csv("data/{}.csv".format(split))
    data = np.array([eval(x) for x in df['pixels']])
    df_pixels = pd.DataFrame(data)
    df_pixels.to_csv("data/{}_pixels.csv".format(split), index=False)

    labels = df[lab_cols]
    labels.to_csv("data/{}_labs.csv".format(split),index=False)
    labels.to_csv("data/{}_labs.csv".format(split),index=False)