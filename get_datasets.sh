# Get CIFAR10
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
mkdir -p data/CIFAR10
tar -xzvf cifar-10-binary.tar.gz -C data/CIFAR10 --strip=1
rm cifar-10-binary.tar.gz 
