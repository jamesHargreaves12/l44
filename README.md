#This code is located at github.com/jamesHargreaves12/l44 please use latest version from here before running tests
Run Test
Steps: 
- python3 -m venv venv
- source venv/bin/activate
- pip3 install -r requirements.txt
- mkdir FER2013
- cd FER2013
- kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
- unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip
- tar -zxvf fer2013.tar.gz
- cd ..
- mkdir models
- python setup_data_FER.py
- python GAN_example.py
 