

class tsv_generator (): 
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def generate_tsv(self):
        with open(self.path, 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerows(self.data)