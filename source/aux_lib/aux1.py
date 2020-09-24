import csv
from collections import defaultdict

def own_read_csv(path_file: str) -> dict:
    """[summary]

    Args:
        path_file (str): [description]

    Returns:
        dict: [description]
    """
    with open(path_file, newline='') as csvfile:        
        reader = csv.DictReader(csvfile)
        dict_values = defaultdict(list)
        for row in reader:
            for key, value in row.items():
                dict_values[key].append(float(value))
    return dict_values

#d2 = own_read_csv('../data/boston1.csv')


            

            

