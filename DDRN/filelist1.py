import os

DIR='F:\\jiangkui\\shujuji\\val'
all_name=os.listdir(DIR)
file_object = open('filelist_val.txt', 'w')
file_list=''
for f in all_name:
    if os.path.isdir(os.path.join(DIR,f)):
        #file_list+=os.path.join(DIR,f)+'\n'
        file_list+=DIR+'/'+f+'\n'
        print(f)
file_object.write(file_list)
file_object.close( )