import pickle 

# Solution from: https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_saved_mdp(filename):
    with open(filename, 'rb') as inp:
          mdp = pickle.load(inp)
          return mdp