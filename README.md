#This code is located at github.com/jamesHargreaves12/l44 please use latest version from here before running tests
Run Test
Steps: 
- python3 -m venv venv
- source venv/bin/activate
- pip3 install -r requirements.txt
- mkdir FER2013
- cd FER2013
- for the next step you will aquire kaggle api to be set up and the credentials located in ~/.kaggle/kaggle.json
- kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
- unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip
- tar -zxvf fer2013.tar.gz
- cd ..
- mkdir models
- mkdir output_images
- python3 setup_data_FER.py
- python3 train_models.py config_test.yaml
To see the change emotion results:
- git clone https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch.git
- follow the instructions under FER2013 Dataset
- mv Facial-Expression-Recognition.Pytorch classifer_fer
- python3 emot_change.py config_test.yaml
 