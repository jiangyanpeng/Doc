# ubuntu20.04安装g++-4.9

增加软件源
sudo vim /etc/apt/sources.list

deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe

sudo apt update
sudo apt-get install gcc-4.9

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 --slave /usr/bin/g++ g++ /usr/bin/g++-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc-7 40 --slave /usr/bin/g++ g++ /usr/local/bin/g++-7

sudo update-alternatives --config gcc