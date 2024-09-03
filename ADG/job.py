import os
from collections import defaultdict
data = '/HitLocatorLibrary/pdbqt'
temp = defaultdict(list)
pdb_list = os.listdir(data)
for pdb in pdb_list:
path = os.path.join( data, pdb )
for file in os.listdir( path ):
file_path = os.path.join( path, file )
temp[pdb].append( file_path )
for key in temp:
print(key)
# os.makedirs( f'./results/{key}' )
with open( f'job{key}', 'w') as F:
for value in temp[key]:
F.write( f'receptor.maps.fld\n' )
F.write( f'{value}\n' )
F.write( f'/home/sylee/project/GLP1R/results/{key}/{value.split("/")[-1][:-6]}\n' )