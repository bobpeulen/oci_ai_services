#Oracle Linux 8

#open firewall
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload


#install docker
dnf install -y dnf-utils zip unzip
dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo

dnf remove -y runc
dnf install -y docker-ce --nobest


systemctl enable docker.service
systemctl start docker.service

#run Label Studio
mkdir mydata
sudo docker run -it -p 8080:8080 -v mydata:/label-studio/data heartexlabs/label-studio:latest

#Open on public IP:8080
