# comparison.py

import pandas as pd
from beautifultable import BeautifulTable


def display_comparison_tables(records, strategies):
    """
    Displays accuracy, time, and combined results tables using BeautifulTable.
    """
    # Create DataFrame from records
    df = pd.DataFrame(records)

    # 1) Pivot accuracy and time tables
    acc_df = df.pivot(index='keep_pct', columns='strategy', values='accuracy')
    time_df = df.pivot(index='keep_pct', columns='strategy', values='avg_time')

    # 2) Format values
    acc_df = acc_df.applymap(lambda x: f"{x*100:.2f}%")
    time_df = time_df.applymap(lambda x: f"{x:.4f}s")

    # 3) Print Accuracy Table
    print("Accuracy Table:")
    _print_beautiful_table(acc_df, include_index=True)

    # 4) Print Time Table
    print("\nAverage Inference Time Table:")
    _print_beautiful_table(time_df, include_index=True)

    # # 5) Combined wide table
    # wide = df.groupby('keep_pct').agg({
    #     f'{strat}_acc': 'first' for strat in strategies
    # })
    # for strat in strategies:
    #     wide[f'{strat}_avg_time'] = df.groupby('keep_pct')[f'{strat}_avg_time'].first()
    # wide = wide.reset_index()

    # # 6) Format combined values
    # for strat in strategies:
    #     wide[f'{strat}_acc']      = wide[f'{strat}_acc'].map(lambda x: f"{x*100:.2f}%")
    #     wide[f'{strat}_avg_time'] = wide[f'{strat}_avg_time'].map(lambda x: f"{x:.4f}s")

    # print("\nCombined Results Table:")
    # _print_beautiful_table(wide, include_index=False)


def _print_beautiful_table(df, include_index=True):
    """
    Helper to print a pandas DataFrame using BeautifulTable.
    If include_index, the DataFrame's index is shown as the first column.
    """
    table = BeautifulTable()
    if include_index:
        headers = ['keep_pct'] + list(df.columns)
        table.column_headers = headers
        for idx, row in df.iterrows():
            table.append_row([idx] + list(row.values))
    else:
        table.column_headers = list(df.columns)
        for row in df.itertuples(index=False):
            table.append_row(list(row))
    print(table)


if __name__ == '__main__':
    # Example usage:
    # records = [
    #     {'strategy': 'random', 'keep_pct': 0.9, 'accuracy': 0.75, 'avg_time': 0.1234,
    #      'random_acc': 0.75, 'random_avg_time': 0.1234},
    #     ...
    # ]
    # strategies = ['random', 'uniform', 'similarity']
    # display_comparison_tables(records, strategies)
    pass

# Note: install dependencies via:
# pip install beautifultable
