#!/bin/tcsh

#Toronto 
set image = $1
# set image = /localStore/SyncShare/SIFshare/Ubuntu2004+GPU/espnet2-kaldi-whisper-cuda11.3.1-cudnn8-devel-ubuntu20.04-20240411.sif
# set image = /home-local/minh/myhome/wenet/examples/librispeech/s0/wenet.sif

#Graz
# set image = /local/Containers/SyncShare/SIFshare/Ubuntu2004+GPU/espnet2-kaldi-whisper-cuda11.3.1-cudnn8-devel-ubuntu20.04-20240411.sif

#Seoul
# set image = /local/Containers/Ubuntu2004+GPU/espnet2-kaldi-whisper-cuda11.3.1-cudnn8-devel-ubuntu20.04-20240411.sif
# set image = /home-local/minh/myhome/MultiModal_LLM/slp2025/slp.sif
# set image = /home-local/minh/myhome/CTC-assisted-ASR-LLM/ctc-asr-llm.sif

# set image = $1

# set i = $#
# if($i != 1)then
# 	echo "ERROR: 引数にsingularityのイメージファイルを1つ指定"
# 	exit 1
# endifs

# singularity run --nv --bind /home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror --no-home $image
singularity run --writable-tmpfs --nv --bind /home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror --no-home $image
# ,/localStore