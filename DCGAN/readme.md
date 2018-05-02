# DCGAN
* original implementation : https://github.com/pytorch/examples/tree/master/dcgan

## Changes
* image size에 맞춰서 네트워크가 동적으로 구성되도록 변경

# Experiments
1. image size 를 64 에서 128로 변경

  일정 상태에 수렴 해 버리면서 더이상 학습이 진행되지 않는다.
원인은 모르겠지만 iscriminator가 너무 강하거나 Generator가 너무 약해서 그럴수도 있을것같다.

2. gradient magnitue 이미지를 같이 사용하여 학습

  `loss = loss + loss_G`
  
  (gradient image 예시)![](http://)
  결과 사진 비교
  
  

