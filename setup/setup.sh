# create the virutual environment in the project root
pip3 install virtualenv
virtualenv -p python3 project_name_env
# activate the virtual environment
source project_name_env/bin/activate
# install packages you will need
pip3 install -r setup/requirements.txt