from setuptools import find_packages,setup
from typing import List

def get_requirment(file_path:str)->list[str]:
    requirments=[]
    with open (file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("\n"," ") for req in requirments]

    return requirments



setup (
    name='Customer upsell prediction',
    version='0.0.1',
    author='vikaskumar',
    author_email='vikaskumar072006@gmail.com',
    packages=find_packages(),
    install_requires=get_requirment('requirment.txt')

)