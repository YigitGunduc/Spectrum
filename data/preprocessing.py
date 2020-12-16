import os
filenames = os.listdir()



with open("raplyrics.txt", "wb") as outfile:
    for f in filenames:
        with open(f, "rb") as infile:
            outfile.write(infile.read())