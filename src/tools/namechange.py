def edit_txt_file(filename):
    """
    This function edits a text file.
    """
    with open(filename, 'r') as file:
        text = file.read()
    text = 'cup' + text[1:]
    with open(filename, 'w') as file:
        file.write(text)


# make a list of all .txt files in directory
import os
files = os.listdir('.')
txt_files = [f for f in files if f.endswith('.txt')]

for file in txt_files:
    edit_txt_file(file)