import os

def make_anotation(filename):
    txt_file = filename.split('.')[0] + '.txt'
    with open(txt_file, 'w') as file:
        text = f"0 0.5 0.5 1 1"
        file.write(text)

files = os.listdir('.')
for filename in files:
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        make_anotation(filename)