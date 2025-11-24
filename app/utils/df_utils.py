import datetime
import numpy as np
import pandas as pd


from utils.utils import print_colored, print_dict, BAR, TimeExecution





def get_even_rows(DF:pd.DataFrame) -> pd.DataFrame:
    '''
    Deletes every other row, only **even rows** remain
    '''
    return DF.iloc[::2].reset_index(drop=True)




def get_dataframe(file_path:str, sheet_name:str=None, date_column:str=None, date_format:str="%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
    import pandas as pd
    try:
        xl = pd.ExcelFile(file_path)
        DF = xl.parse(sheet_name=sheet_name)
    except ValueError:
        if file_path.endswith(".csv"):
            DF = pd.read_csv(file_path)
        elif file_path.endswith(".tsv"):
            DF = pd.read_csv(file_path, sep='\t')
        else:
            DF = TypeError(f"Unsupported file extension for file '{file_path}'")
    if not (date_column is None):
        DF[date_column] = [datetime.datetime.strptime(d, date_format) for d in DF[date_column]]

    return DF
    





def dict_to_dataframe_row(data:dict) -> pd.DataFrame:
    new_data:dict = dict()
    for k,v in data.items():
        new_data[k] = [v]
    return pd.DataFrame(new_data)




def concat_dfs(df1:pd.DataFrame|None, df2:pd.DataFrame|dict, reset_index:bool=False) -> pd.DataFrame:
    assert not (df2 is None)
    if isinstance(df2, dict):
        has_list:bool = True
        for k, v in df2.items():
            if not isinstance(v, list):
                has_list = False
                break
        if has_list:
            processed_df2 = pd.DataFrame(df2)
        else:
            processed_df2 = dict_to_dataframe_row(df2)
    else:
        processed_df2 = df2
    if df1 is None:
        return processed_df2
    else:
        if not reset_index:
            return pd.concat(objs=[df1, processed_df2], ignore_index=True)
        else:
            return pd.concat(objs=[df1, processed_df2], ignore_index=True).reset_index()
   




def dict_to_df(d:dict, key_column_name:str="key", value_column_name:str="value") -> pd.DataFrame:
    df:pd.DataFrame|None = None
    for k, v in d.items():
        df = concat_dfs(df, {key_column_name:k, value_column_name: v})
    return df




def fix_class_imbalance(DF:pd.DataFrame,
                        target_column:str,
                        downsample:bool=True,
                        noise_magnitude:float|None=None,
                        verbose:bool=False,
                        ) -> pd.DataFrame:
    labels:set = DF[target_column].drop_duplicates()
    label_count:dict[int|str, int] = dict()
    new_DF:pd.DataFrame = None
    for l in labels:
        label_count[l] = len(DF[DF[target_column] == l])
    if verbose:
        print("Current class counts:")
        print_dict(label_count)
    # FIX CLASS IMBALANCE
    if downsample:
        if verbose:
            print("Fixing class imbalance with ", end="")
            print_colored("down", color="red", end="-sampling ... ")
        new_DF = DF.copy()
        min_count:int = min(label_count.values())
        remove_idx:list[int] = list()
        for l in label_count.keys():
            current_count:int = label_count[l]
            if current_count == min_count:
                continue
            for idx, row in DF.iterrows():
                # remove random samples
                if row[target_column] == l:
                    remove_idx.append(idx)
                    current_count -= 1
                    if current_count <= min_count:
                        break
        new_DF.drop(remove_idx, inplace=True)
        new_DF.reset_index(inplace=True)
        new_DF.drop(columns=['index'], inplace=True)
    else:
        # UPSAMPLE
        if verbose:
            print("Fixing class imbalance with ", end="")
            print_colored("up", color="blue", end="-sampling ... ")
        max_count:int = max(label_count.values())
        for l in label_count.keys():
            if max_count - label_count[l] > 0:
                labeled_samples = DF[DF[target_column] == l] # get samples with current label
                new_rows = labeled_samples.sample(n=max_count-label_count[l], replace=True) # get random rows
                if noise_magnitude:
                    if verbose:
                        print_colored("perturbating", color="red", end=" ")
                        print("the newly generated rows:")
                        bar = BAR(length=len(new_rows))
                    # perturbate samples randomly
                    perturbed = None
                    for idx, row in new_rows.iterrows():
                        new = dict()
                        for f in set(new_rows.columns):
                            if f == target_column:
                                new[f] = row[f]
                                continue
                            old_value = row[f]
                            perturbation = (noise_magnitude-(-noise_magnitude))*np.random.rand()-noise_magnitude
                            try:
                                new_value = old_value + (old_value*perturbation)
                            except TypeError:
                                new_value = old_value
                            new[f] = new_value
                        perturbed = concat_dfs(perturbed, new)
                        if verbose:
                            bar.update()
                    new_rows = perturbed
                new_DF = concat_dfs(new_DF, new_rows)
        new_DF = concat_dfs(new_DF, DF)
        new_DF.reset_index(inplace=True)
        new_DF.drop(columns=['index'], inplace=True)
    if verbose:
        print("done.")
    return new_DF




def missing_values_table(df:pd.DataFrame) -> pd.DataFrame:
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

