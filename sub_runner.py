import subprocess
import os

def run_script(script_name,args=''):
    try:
        if args!='':
            subprocess.run(['python', script_name]+args, check=True)
        else:
            subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

filenames=['RS_2014-05.zst','RS_2014-06.zst','RS_2014-07.zst']

os.system("rm data/out.csv")

for i in filenames:
    args=[f'data/reddit/submissions/{i}','data/zst_as_csv.csv','author,title,score,created,link,text,url']
    run_script('to_csv.py',args)
    run_script('post_finder.py')