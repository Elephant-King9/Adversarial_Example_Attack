# 下载color_net的预训练文件

MODEL_FILE=./latest_net_G.pth
URL=http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth

wget -N $URL -O $MODEL_FILE
