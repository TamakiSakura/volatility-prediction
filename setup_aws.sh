sudo yum groupinstall "Development Tools"
sudo yum install tmux atlas-sse3-devel lapack-devel

# Provide from https://gist.github.com/dacamo76/4780765
wget https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.11.6.tar.gz
tar xzf virtualenv-1.11.6.tar.gz 
python27 virtualenv-1.11.6/virtualenv.py sk-learn
. sk-learn/bin/activate

sudo pip install cython scipy numpy
sudo pip install scikit-learn pymc
sudo pip install lda
sudo pip install textmining
sudo pip install -U pip
