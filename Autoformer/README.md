
- [Review](https://seollane22.tistory.com/22)

이 논문은 기존 트랜스포머 모델에 시계열의 특징을 추가한 변형 모델인 Autoformer를 제안한 논문이다.

트랜스포머의 등장 이후, self-attention을 장착한 트랜스포머 기반 모델은 기존 RNN, CNN 기반의 딥러닝 모델들의 성능보다 더 뛰어난 성능을 내고있다.

여러 트랜스포머 기반의 변형모델들은 장기(long-term, long-range) 시계열의 예측성능을 "효율적으로" 높이기 위해 attention module이나 그 architecture를 개조하여 좋은 성능을 입증하였다.



그러나, 이 논문은 지난 변형모델들과는 다르게 전통적인 "시계열 데이터의 특징을 입힌", "시계열 분석에 더욱 특화된" 모델인 Autoformer를 제안한다.

Autoformer의 차별적인 요소는 바로 "Decomposition"과 "Auto-Correlation"의 매커니즘을 이용한다는 것이다.



결과적으로 이 모델은 다른 모델 대비 가장 탄탄한 성능을 보여주는 등, 장기 시계열 Forecasting에서 "SOTA"성능을 달성하였고 앞으로 딥러닝에서 시계열이 어떠한 방향으로 나아가야 할 지, 그 가능성까지 시사하고 있는 중요한 논문이다. 
