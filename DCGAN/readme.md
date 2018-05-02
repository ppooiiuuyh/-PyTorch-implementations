#DCGAN
original implementation : https://github.com/pytorch/examples/tree/master/dcgan

#changes
imageSize에 맞춰서 네트워크가 자동으로 구성되도록 변경


#D(G(z))는 거의 0.1 미만이었다(잘 구별한다). => D(G(z))가 낮아야 학습이 가능, 또는 D(G(z))를 낮추는게 더 강하게 작용

maximize log(D(G(z))) 만 사용하였을 경우에는 200 epo쯤에서 학습이 중단되었다 (D(G(z))가 너무 높아졌을 가능성이있음)
minimize log(1-D(G(z))) 만 사용하였을 경우에는 아예 학습이 시작조차 안되었다
둘을 같이 사용한다면? 또는 둘중 유효한것으로만( |loss| 가 큰것) 사용한다면?



''' 왜 안되는지 모르겠음
imagesize 128 => 안됨. gan을 키우는것이아닌 D를 바보로 만드는것에 주력해버린다.
학습률을 낮추면? => 여전히 안됨
D만 학습률을 낮추면? ==>여전히 안됨 (어차피 G의 optimizer에서 학습하기 때문에)
G만 학습률을 낮추면? ==> 당연히 안됨 (역시나 어차피 D를 바보로만드는것이 더 좋음)
