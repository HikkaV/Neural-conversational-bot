##setting up requirements
sudo apt-get update
sudo apt install virtualenv
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
