import pandas as pd

def left_merge_new_fields(left, right, key):
    new_cols = [col for col in right.columns if col not in left.columns or col == key]
    return left.merge(right[new_cols], on=key, how='left')
