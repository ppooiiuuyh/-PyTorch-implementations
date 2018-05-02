# DCGAN
* original implementation : https://github.com/pytorch/examples/tree/master/dcgan

# Changes
* image size에 맞춰서 네트워크가 동적으로 구성되도록 변경

# My Experiments


##### gradient magnitue 이미지를 같이 사용하여 학습
* 실험 개요
 사람의 경우에는 대상을 구분할때 주요하게 여기는 부분과 주요하게 여기지 않는 부분이 다르다. 예를들어 사람 얼굴에서 살이 쪄있는정도, 머리카락의 길이 등의 차이에는 크게 이상함을 느끼지 않지만, 코뼈의 모양이나 눈의 위치과 같은 골격의 뭉게짐에는 쉽게 민감하게 반응하고 혐오스럽게 느끼게된다.

 하지만 Generator와, Discriminator의 입장에서는 골격과 같은 정보는 전혀 모르며 그저 단순히 loss를 줄일 수 있는 공통적인 feature에 우선시 하게된다. 때문에 골격의 뭉게짐에 더욱 가중치를 둘 수 있다면 사람이 보기에도 더 실제같은 이미지를 생성할 수 있을것이라 생각했다.

 이를위해서 골격의 정보라 할 수 있는 gradient image를 쌍으로 학습시켰다. 이를 통해서 G와 D는 골격(gradient) feature에 더 신경쓰게 될것이다. 이 골격정보는 hard한 feature가 될것이며 살쪄있는 정도나 머리색 등은 soft한 feature가 될것이다.

* magnitude of gradient 예시

  ![](https://raw.githubusercontent.com/ppooiiuuyh/-PyTorch-implementations/master/DCGAN/asset/test.jpg)
  
  ![](https://raw.githubusercontent.com/ppooiiuuyh/-PyTorch-implementations/master/DCGAN/asset/test_grad.png)


* loss function
  `loss = loss + loss_G`시시
  
* 결과 사진 비교
 * originalDCGAN

  ![](https://raw.githubusercontent.com/ppooiiuuyh/-PyTorch-implementations/master/DCGAN/asset/fake_samples_epoch_078.png)

 * DCGAN with gradient

 ![](https://raw.githubusercontent.com/ppooiiuuyh/-PyTorch-implementations/master/DCGAN/asset/fake_samples_epoch_078_grad.png)

 여전히 뮤턴트같지만 조금더 틀이 잡혀있는듯이 보이기도 한다. loss가 하나 추가되어 단순히 더 빨리 학습된것인지 (실제로 학습시간은 배로 더걸리게된다) 눈의 착각인지는 모르겠다.
 
 사실 골격정보로 전체 gradient 이미지를 사용한것은 타당하지 못하다. 이 gradient이미지는 배경에 대한 정보도 가지고있으며 nosie로 작용할것이다. 하지만 학습데이터가 많아질수록 이 gradient 이미지들에서도 공통적인 feature들을 뽑아낼 것이고 이는 골격정보로서 작용할것이라고 생각하였다.
 
 그러나 여전히 gradient 이미지에는 문제가 남아있다. 애초에 뼈와 같은 골격정보와 살과같은 가변요소에 대한 가중치를 구분하려함이었지만 원본 이미지에 대한 gradient는 이미 살과같은 가변요소 자체를 골격으로 처리하게된다. 더욱 목적에 맞도록 하려면 정말로 얼굴<->뼈 이미지를 쌍으로 사용하는것이 좋을것이다. 이는 나중에 여유가된다면 해봐야겠다.
 
 이를 위해서는 얼굴<-> 뼈 이미지 쌍의 데이터셋이 필요한데, 이는 얼굴에서 뼈이미지를 생성하는 프로그램이나 모델을 이용할 수 있을것이다. 추가로 다양한 뼈사진을 입력으로 넣어주면 더욱 뭉게짐이 덜한 얼굴이미지를 생성할 수 있을것이라 기대한다.
 
 
# Discussion (other experiments)
##### 1. image size 를 64 에서 128로 변경

  초기부터 일정 상태에 수렴 해 버리면서 더이상 학습이 진행되지 않는다.
원인은 모르겠지만 iscriminator가 너무 강하거나 Generator가 너무 약해서 그럴수도 있을것같다.

##### 2. D와 G의 학습률 조정

  식적으로도 당연하게도 의미 없었다. 같이 학습이 느려지거나 같이 빨라지기 때문에 둘 사이의 밸런스를 조절하는 효과는 얻을 수 없었다.
  
##### 3. -log(1-D(G(z))) 와 log(D(G(z))) 의 혼용.

  애초에 -log(1-D(G(z))) 대신에 log(D(G(z)))를 사용한 이유가 초기에 D가 G(z)를 구별하기 너무 쉬운상태에서는 학습이 안되었기 때문이다. 그말은 나중에 D가 구별해내기 너무어려워졌을때는 오히려 log(D(G(z)))가 효과가 낮아진다는것을 의미할것같아서 max( abs(loss1), abs(loss2)) 로 유리한것을 사용하도록 해보았다. 하지만 크게 효과는 없었다. 나중에 다시 생각해봐야할것같다.
  
  
##### 4. G의 output activation function으로 sigmoid 대신 clamp(output, -1,1) 사용

  tanh 함수의 특성상 -1과 1의 값의 등장확률은 다른값들에 비해 낮을것이라 생각했다. 이에 대하여 검은색과 흰색(극단색)의 생성 빈도가 낮게되지 않을까 하는 생각을 해보았다. 따라서 모든 분포확률이 동일하도록 y=x함수에 -1이하에선 -1, 1이상에선 1이 나오도록 사용해 보았지만 잘 되지 않았다. G의 output자체를 cross entrophy로 역전파하지는 않을테니 역전파문제는 아닐것같은데 정확한 이유는 모르겠다. 물론 안되는게 맞기는하지만 아직은 이해는 안된다.
  
 # Author
 Dohyun Kim