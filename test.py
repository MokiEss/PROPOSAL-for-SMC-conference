import csv
import numpy as np
import DEV2
# open the file in the write mode


result10D = np.zeros((30,25))
result30D = np.zeros((30,25))
result50D = np.zeros((30,25))
result100D = np.zeros((30,25))

print("experimentation on 10D")
D = 10
for i in range(30):
    print("function ",i, "dimension", D)
    for j in range(25):
        result10D[i,j] = DEV2.optimize(i,D)

print("experimentation on 30D")
D = 30
for i in range(30):
    print("function ", i, "dimension", D)
    for j in range(25):
        result30D[i,j] = DEV2.optimize(i,D)
print("experimentation on 50D")
D = 50
for i in range(30):
    print("function ", i, "dimension", D)
    for j in range(25):
        result50D[i,j] = DEV2.optimize(i,D)
print("experimentation on 100D")
D = 100
for i in range(30):
    print("function ", i, "dimension", D)
    for j in range(25):
        result100D[i,j] = DEV2.optimize(i,D)

f = open('results10DCEC2017.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(result10D)

# close the file
f.close()

f = open('results30DCEC2017.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(result30D)

# close the file
f.close()

f = open('results50DCEC2017.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(result50D)

# close the file
f.close()

f = open('results100DCEC2017.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(result100D)

# close the file
f.close()