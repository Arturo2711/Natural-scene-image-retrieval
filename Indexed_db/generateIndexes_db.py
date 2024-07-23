import pandas as pd


### To generate the csv

def write_Instance(path, vector, name):
    with open(path, mode='+a', encoding='utf-8') as file:
        line_to_write = name + ',' + str(vector[0]) + ',' + str(vector[1]) + ',' + str(vector[2])+ ',' + str(vector[3]) + ',' + str(vector[4]) + ',' + str(vector[5]) + ',' + str(vector[6]) + ',' + str(vector[7]) + ',' + str(vector[8]) + ',' + str(vector[9]) + '\n'
        file.write(line_to_write)



# Load the CSV file
df = pd.read_csv('Classifier/describing_vectors_clustered.csv')

# Select the feature columns (adjust this based on your actual column names)
features = ['Name', 'cluster']
X = df[features]

listImageComponents = [[1 for _ in range(10)] for _ in range(1468)]


def getComponents_Db(row):
    global listImageComponents
    #print(row['Name'][6])
    
    # Assuming 'Name' contains a string from which you want to extract a number
    name = row['Name']
    digitos = ''.join([char for char in name if char.isdigit()])
    number =  int(digitos)# Extracting the 6th character as a number (adjust as needed)
    cluster = row['cluster']
    
    # Increment the count
    #print(number)
    #print(cluster)
    listImageComponents[number][cluster] += 1

# Apply the function to each row
X.apply(getComponents_Db, axis=1)

# Now listImageComponents contains the accumulated counts
for i in range(len(listImageComponents)):
    i += 1
    name = 'Image{}'.format(i)
    write_Instance('Indexed_db\indexed.csv', listImageComponents[i], name)