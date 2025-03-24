import sys
import os

# path to the root folder.
dir ="/home/shabeen/data/Dataset/"

# take the root dir path as command line argument if passed.
if len(sys.argv) > 1:
	dir = sys.argv[1]

print('root directory -',dir)

classes  = os.listdir(dir)

# filter out any files and other folders from classes and do sorting
# classes = sorted(list(set([item.lower() for item in classes if '.' not in item and item!='AllDatasets'])))
classes=sorted([item  for item in classes if '.' not in item and item!='AllDatasets'])

print('Total classes:',len(classes),'\nClasses :',classes,'\n')

# to print number of images in each folder or classes
inp=input("Want to print number of plants of classes (y/n) : ")
if inp in ['y','Y']:
	nmbr_images={i:len(os.listdir(os.path.join(dir,i))) for i in  classes}
	# sorting folders based on number of images 
	sorted_dict = dict(sorted(nmbr_images.items(), key=lambda item: item[1],reverse=True))
	for k,v in sorted_dict.items():
		print(f'{k:-<20}{v}') #if "." not in k  else  print(k)
	# Display the number of total images available.		
	print(f"Total Images : {sum(sorted_dict.values())}")
