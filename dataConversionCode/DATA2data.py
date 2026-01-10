#　dataConversionCode/DATA2data.py
#  ファイル構造シンプル化
import os
import shutil

date = "2025_11_03_2"
need_txt = True

output_dir = 'C:/imageProcessing/data_sort/' + date + "/" 
os.makedirs(output_dir, exist_ok=True)

ex_output = output_dir + "DATA/"
os.makedirs(ex_output, exist_ok=True)

dir1 = "C:/imageProcessing/DATA/" + date + "/DATA"
folders = [f for f in os.listdir(dir1)] #100559, 109021...

for i in range(len(folders)): #len(folders)
    dir2  = dir1 + "/" + folders[i] #DATA/100559
    folders2 = [f for f in os.listdir(dir2)] 

    for j in range(len(folders2)):
        dir3  = dir2 + "/" + folders2[j] # DATA/100559/20241003
        folders3 = [f for f in os.listdir(dir3)]

        for k in range(len(folders3)):
            dir4  = dir3 + "/" + folders3[k] # DATA/100559/20241003/130714
            folders4 = [f for f in os.listdir(dir4)]

            for l in range(len(folders4)):
                EXdir  = dir4 + "/" + folders4[l] # DATA/100559/20241003/130714/EX26
                new_name = folders[i] + folders2[j] + folders4[l] # 10055920241003EX26 ID+date+EX
                shutil.move(EXdir, ex_output + new_name)

if need_txt:
    txtfile = "C:/imageProcessing/DATA/" + date + "/Data.txt"
    shutil.move(txtfile, output_dir)

    dicomdir = "C:/imageProcessing/DATA/" + date + "/DICOMDIR"
    shutil.move(dicomdir, output_dir)
