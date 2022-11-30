set -e
apt-get update
apt install python3.8-venv
python -m venv /projects/active-adversarial-tests/venv3.8tf --system-site-packages
source /projects/active-adversarial-tests/venv3.8tf/bin/activate
pip install robustml
pip install cleverhans==3.0.1 # updating causes package inconsistencies
pip install keras==2.2.4 # updating causes package inconsistencies
pip install line_profiler
/projects/active-adversarial-tests/install_requirements.sh

