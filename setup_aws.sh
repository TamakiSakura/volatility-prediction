set -e

function cleanup {
  echo "Error somewhere see above for more detail"
  set +e
}
trap cleanup EXIT

sudo yum -y groupinstall "Development Tools"
sudo yum -y install tmux atlas-sse3-devel lapack-devel

# Provide from https://gist.github.com/dacamo76/4780765
wget https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.11.6.tar.gz
tar xzf virtualenv-1.11.6.tar.gz 
rm -f virtualenv-1.11.6.tar.gz 
python27 virtualenv-1.11.6/virtualenv.py sk-learn
. sk-learn/bin/activate

pip install -U pip

# This part is only for if you are using a instance with memory less than 2GB
# Creating a swap file of 2GB for scipy installation
# dd if=/dev/zero of=/swapfile1 bs=1024 count=2097152
# chown root:root /swapfile1
# chmod 0600 /swapfile1
# mkswap /swapfile1
# swapon /swapfile1

pip install cython numpy
pip install scipy
pip install scikit-learn pymc
pip install lda textmining

# swapoff /swapfile1
# swapon -s

echo ". ~/sk-learn/bin/activate" >> ~/.bashrc
echo "export PS1=\"Tamaki_Sakura@\w:\"" >> ~/.bashrc

git clone https://github.com/guohengw/volatility-prediction.git
cd volatility-prediction
git checkout -b hmm_lda origin/hmm_lda
