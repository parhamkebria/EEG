import ast
import scipy.io
import numpy as np
import pandas as pd

# mat = scipy.io.loadmat("path to the .mat file")
# x = np.array(mat)
# print(mat.keys())


def mat2csv(path):
    file_name = path.split("/")[-1].split(".")[0]
    csv_path = path.replace(".mat", ".csv")
    mat = scipy.io.loadmat(path, squeeze_me=True)

    records = mat["data"]
    rows = []

    for i, rec in enumerate(records):
        x = np.asarray(rec["X"]).item()
        x = np.asarray(x, dtype=float)

        gender = np.asarray(rec["gender"]).item()
        age = np.asarray(rec["age"]).item()
        fs = np.asarray(rec["fs"]).item()
        classes = np.asarray(rec["classes"]).reshape(-1).tolist()
        trial = np.asarray(rec["trial"]).reshape(-1).tolist()
        y = np.asarray(rec["y"]).reshape(-1).tolist()

        flat = x.reshape(-1)
        row = {
            "record_index": i,
            "gender": gender,
            "age": age,
            "fs": fs,
            "classes": str(classes),
            "trial": str(trial),
            "y": str(y),
            "signal": flat.tolist(),
        }
        rows.append(row)

    full_df = pd.DataFrame(rows)
    print(full_df.head(1))
    full_df.to_csv(csv_path, index=False)
    print(full_df.shape)
    print("CSV file saved successfully to", csv_path)

# path = "<path to the .mat file>"
# mat2csv(path)