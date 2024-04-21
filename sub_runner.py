import subprocess
import os
import pandas as pd

def run_script(script_name,args=''):
    try:
        if args!='':
            subprocess.run(['python', script_name]+args, check=True)
        else:
            subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

filenames = [
    "RS_2014-01.zst",
    "RS_2014-02.zst",
    "RS_2014-03.zst",
    "RS_2014-04.zst",
    "RS_2014-05.zst",
    "RS_2014-06.zst",
    "RS_2014-07.zst",
    "RS_2014-08.zst",
    "RS_2014-09.zst",
    "RS_2014-10.zst",
    "RS_2014-11.zst",
    "RS_2014-12.zst",
    "RS_2015-01.zst",
    "RS_2015-02.zst",
    "RS_2015-03.zst",
    "RS_2015-04.zst",
    "RS_2015-05.zst",
    "RS_2015-06.zst",
    "RS_2015-07.zst",
    "RS_2015-08.zst",
    "RS_2015-09.zst",
    "RS_2015-10.zst",
    "RS_2015-11.zst",
    "RS_2015-12.zst",
    "RS_2016-01.zst",
    "RS_2016-02.zst",
    "RS_2016-03.zst",
    "RS_2016-04.zst",
    "RS_2016-05.zst",
    "RS_2016-06.zst",
    "RS_2016-07.zst",
    "RS_2016-08.zst",
    "RS_2016-09.zst",
    "RS_2016-10.zst",
    "RS_2016-11.zst",
    "RS_2016-12.zst",
    "RS_2017-01.zst",
    "RS_2017-02.zst",
    "RS_2017-03.zst",
    "RS_2017-04.zst",
]


os.system("del data/out.csv")

for i in filenames:
    args=[f'data/reddit/submissions/{i}','data/zst_as_csv.csv','author,title,score,created,link,text,url']
    run_script('to_csv.py',args)
    run_script('post_finder.py')
