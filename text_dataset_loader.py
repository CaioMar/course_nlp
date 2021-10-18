"""
Loads text file as dataset that can be used for training models.
"""
import pandas as pd


def load_dataset() -> pd.DataFrame:
    """
    Loads poems from both Edgar Allan Poe and Robert Frost and creates
    a pandas dataframe where each line of a poem is an entry. It also
    creates an author column that can be employed in training supervised
    models.
    """

    with open('./edgar_allan_poe.txt','r') as file:
        edgar = file.readlines()

    with open('./robert_frost.txt','r') as file:
        robert = file.readlines()

    dataset = pd.DataFrame(zip(edgar, ['edgar allan poe']*len(edgar)), columns=['poem_line', 'author'])
    dataset = dataset.append(
        pd.DataFrame(zip(edgar, ['robert frost']*len(robert)), columns=['poem_line', 'author'])
    ).reset_index(drop=True)
    dataset.poem_line = dataset.poem_line.apply(lambda x: x.replace('\n',''))

    return dataset