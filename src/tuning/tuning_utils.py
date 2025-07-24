import os
import json
import pandas as pd

def load_stopper_info(filepath):
    """Loads historical early stopper objective and count."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        return state['_best_objective'], state['_n_lower']
    print('Stopper filepath not found.')
    return None, 0

def save_stopper_info(stopper, filepath):
    """Save optimization early stopper objective and count."""
    state = {'_best_objective': stopper._best_objective, '_n_lower': stopper._n_lower}
    with open(filepath, 'w') as f:
        json.dump(state, f)

def save_problem(problem, filepath):
    """Save deephyper HpProblem hyperparameters specifications to json.
    
    Parameters
    ----------
    problem : deephyper.hpo.HpProblem
        The problem to save.
    filepath : str
        The path to save the problem to.
    """
    # Save config space as object
    problem.space.to_json(filepath)

def print_save_best_params(save_file, file_path):
    """Prints the parameters of the best tuning run so far."""
    df = pd.read_csv(save_file)
    best_idx = df["objective"].idxmax()
    best_row = df.loc[best_idx]

    # Filter for columns starting with "p:"
    p_vars = best_row.filter(like="p:").to_dict()
    p_vars['objective'] = best_row['objective']
    print(p_vars)

    # save best params to file
    with open(file_path, 'w') as f:
        json.dump(p_vars, f)