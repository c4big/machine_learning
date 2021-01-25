캐글 단축키
https://www.kaggle.com/fulrose/smart/comments	// 아무도 가르쳐주지 않는 Smart한 커널 사용법(단축키 버전)
editor mode: enter
command mode: 

현재 셀 위에(Above) 셀 만들기 : ( Command Mode ) + A
현재 셀 아래(Below) 셀 만들기 : ( Command Mode ) + B

Code 셀로 변경 :      ( Command Mode ) + Y
Markdown 셀로 변경 :  ( Command Mode ) + M

셀 삭제 :  ( Command Mode ) + D, D (두 번)

셀 삭제 실행취소 : ( Command Mode ) + Z

셀 복사 :    ( Command Mode ) + C
셀 잘라내기 : ( Command Mode ) + X
셀 붙여넣기 : ( Command Mode ) + V
// 붙여넣기는 선택 셀 바로 아래에 생성됩니다. Shift + V를 누르면 위에 생성됩니다.

셀 복수 선택 : ( Command Mode ) + Shift + 화살표
// Shift를 누른 상태에서 화살표를 눌러서 셀을 이동하면 시작 셀 부터 이동한 지점 까지의 셀이 모두 선택됩니다.
// 선택된 셀을 지우거나 형식 변경, 복사, 잘라내기를 위와 똑같이 하면 됩니다.

바로 아래의 셀과 현재 셀 병합 : ( Command Mode ) + Shift + M
// 위 명령어로 방금 배운 복수 선택 후 선택한 셀을 모두 병합하는 것도 가능합니다.

셀 분리 : ( Editor Mode ) + Ctrl + Shift + - (마이너스, 하이픈)

라인 넘버 노출 : ( Command Mode ) + L
// 마크다운, 코드 모두 표시할 수 있습니다.

Find, Replace : ( Command Mode ) + F // 크롬을 사용하는 경우라면 크롬에서 제공하는 찾기를 사용해도 되지만 커널 내부에서는

커맨드 팔레트: ( Command Mode ) + P

단축키 도움말: ( Command Mode ) + H		// 간단하게 숏컷 도움말을 물러올 수 있다는데 어째 실행을 어떻게 하누..




# 코드 툴팁이나 코드 도움말은 반드시 다른 셀에서 한번 실행이 된 상태에서만 가능합니다. 이 셀을 먼저 실행 하세요
import pandas as pd
# 연습
# 1. 에디터 모드
# 2. 작성된 코드에서 Shift + TAB

# pd.concat 확인 : concat 의 마지막 t에 커서를 놓은 후 Shift + TAB
# 툴 창이 뜨고 ^를 누르면 아래에서 모두 확인 가능
pd.concat		// 2초 이상 shift + tab 을 누르면 뜬다능

# 이렇게도 볼 수 있습니다.
pd.concat?		// 실행하면 됨

# 하드하게 짜여진 코드 까지 보고 싶다면?
pd.concat??		// 실행하면 됨


"//----01. 타이타닉_Preparation----//"
	"/--A. 준비하기--/"	// 01
colab 에서 github 사본으로 저장을 할때 문제점
	일단 repository 주제를 machine_learning 으로 하고 싶었지만
	적정한 이름의 repository 가 없었다. 그로 인해 새 repository 를 만들어야 했다. 
	헌데 github 에서 만들어고나서 colab 에서 해당 주소로 저장하려 해도 "브랜치"가 없다고 한다. 으음.. 모지
	
	해결1: 일단 github desktop 을 깔아서 new repository 를 빈 repository 만드니까 해결된다. 구글링을 해보니 readme 가 있어서라는데(파일 아무거나 하나 추가였더라)
	
	해결2: github 에서만 하려면 그 상태에서 아무 파일(확장자도 아무거나)을 하나만 추가하면 해결된다. 
		github 해당 repository 에서 "Add file" -> "Create new file" 한다음에는 colab 에서 연동이 된다
	
colab 텍스트
# 아주 큰글
## 다음 사이즈
### 그 다음 사이즈
#### 그 다음 사이즈

일반글
* 별표		// 뛰워쓰라.
- 하이픈
			print("Hello Kaggle") # Pseudo 코드 // 탭이 세번 들어가야 함
			
			
	"/--B. 모듈 및 데이터 가져오기--"	// 02
주요내용
01: 데이터 가져오기	// 복사시 누락되있을 경우 Add Data 를 클릭하여 챌린지데이터에서 가져온다 : 캐글에서임
02: 주요 모듈 임포트	// 기본 모듈, 시각화 모듈, 알고리즘 모듈
03: 보조 링크 파악	// 각 종 외부 링크를 통하여 추가 학습 가능
04: csv를 데이터 프레임으로 전환	// csv 파일을 판다스 데이터 프레임으로 전환

강의 처음 화면은 캐글이다. 검은색이라 눈치 노트북으로 착각했다. 그리고 캐글의 타이타닉에서 "EDIT" 를 눌러야 강의 화면과 동일하게 나온다.!!
 
// 캐글에서 데이터: 위와 같은 주소(../input/titanic/test.csv)가 있습니다. 저 주소를 복사해서 아래에서 사용합니다. 어디에?? 

/*	Part1: 데이터 준비 및 모듈 임포트	*/
# 기본 데이터 정리 및 처리
import numpy as np		// 행렬 등 수학적 처리 모듈
import pandas as pd		// pandas: 데이터 처리 및 조작 모듈, numpy 와 같이 사용할 거다.

# 시각화
import matplotlib.pyplot as plt
%matplotlib inline		// 더 나은 출력 : 그래프, 사운드 애니메이션 등
import seaborn as sns	// 시각화
plt.style.use('seaborn-whitegrid')	// 시본 wiitegrid 스타일로 plot 한다.
import missingno	// 누락된 값을 한눈에 보기 위해

# 전처리 및 머신 러닝 알고리즘
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sgboost import XGBClassifier
from sklearn.ensemble import GradientBosstingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# 도델 튜닝 및 평가
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection

# 경고 제거 (판다스가 에러 메세지를 자주 만들어 내기 때문에 이를 일단 무시하도록 설정)
import sys
import warnings
import warnings		// 중복인데 동영상강의에선 이 4줄이 필요하다고 하네..
warnings.filterwarnings('ignore')




/*	CSV to DF	*/
# csv 를 임포트하여 데이터셋이 판다스 데이터프레임이 되도록 합니다.
# 이 것이 처음하는 사람에게 예상보다 어려울 수 있는데 복사한 것에서 데이터가 전달이 잘 안 되었다면 "+Add Data" 누르시고 'Competition Data'에서 "Titanic Data" 불러온 후 파일을 찍어서 경로 주소 확인해야 함 
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')
# 이제 csv file 들 (test & train) 은 데이터 프레임이 되었습니다.

/*
구글 콜랩엣 사용하실 때는 컴퓨터에 파일을 다운로드 한 후 아래 코드를 입력하면 불러올 수 있게 된다.
	from google.colab import files
	uploaded = files.upload()
그런 다음 아래 코드를 통해서 csv 를 데이터프레임으로 바꿀 수 있게 된다.
	import io
	test = pd.read_csv(io.BytesIO(uploaded['text.csv'))
	train = pd.read_csv(io.BytesIO(uploaded['train.csv']))
*/


"//----02. 타이타닉_EDA----//"
	"/--A. EDA 시작하기--/"// 03
"주요내용"
// EDA 의 이유
	왜 EDA가 필요한지 잠시 생각해본다.
// 데이터 보기
	다양한 데이터 보기 방법을 통하여 데이터 개요를 본다.
// 트레인 및 테스트 Y축 보존	
	추후 사용을 위해 테스트 및 트레인 데이터 크기와 y 축 등을 저장 해 둔다.
// 트레인 및 테스트 데이터 연결
	변환할 때 같이 변환하기 위해 트레인 및 테스트 데이터를 연결하여 data 를 만든다.

"EDA"
	Exploratory Data Analysis	// 탐색적 데이터 분석
	https://eda-ai-lab.tistory.com/13		
	


"데이터 프레임을 보는 다양한 방법"
head() 첫 5행을 볼수 있습니다.
	train.head(n = 3) // 3행을 볼수 있습니다.

tail() 마지막 5행을 볼 수 있습니다.
	train.tail()

describe() 각 열의 통계적인 면을 보여 줍니다.
기본은 연속된 값을 가진 열만 보여주나 (include='all') 로 세팅하면 모두 볼 수 있습니다.
	train.describe(include='all')
https://www.w3resource.com/pandas/dataframe/dataframe-describe.php	// describe 보삼

"exasmples"
	import numpy as np
	import pandas as pd
	s = pd.Series([2,3,4])
	s.describe()
count	3.0			// 3 numbers
mean	3.0			// mean or average : 평균
std		1.0			// standard deviation : 표본 표준편차임(sample standard deviation) 
					// : 우리가 고등학교때 배운 것은 나누기 N 이었지만 이것은 나누기 (N - 1) 을 취한다.
min		2.0
25%		2.5
50%		3.0
75%		3.5
max		4.0
dtype:	float64
	

	s = pd.Series(['p', 'p', 'q', 'r'])
	s.describe()
count	4		// 4 letters
unique	3		// 유일성
top		p		// top letter is p
freq	2		// 빈번한 것 갯수
dtype:	object


	
dtypes 모든 열의 데이터 종류를 보여 줍니다.
	train.dtypes
	info() 는 dtypes 의 좀 더 발전된 개념으로 데이터 타입뿐만 아니라 빈칸이 아닌 갯수까지 보여 줍니다.
	
배경지식	
공분산과 상관계수	
https://m.blog.naver.com/PostView.nhn?blogId=mykepzzang&logNo=220838462884&proxyReferer=https:%2F%2Fwww.google.com%2F


공분산(Covariance)과 상관계수(Correlation)
https://destrudo.tistory.com/15
공분산의 성질
	Cov(X, Y) > 0	// X가 증가 할 때 Y도 증가한다.
	Cov(X, Y) < 0	// X가 증가 할 때 Y는 감소한다.
	Cov(X, Y) = 0	// 공분산이 0이라면 두 변수간에는 아무런 선형 관계가 없으며 두 변수는 서로 독립적인 관계로 볼수 있다.
					// 라고 긍정적으로 해석하기엔 수학적 엄밀함을 들이밀면 다음과 같게 됨...
					// 두 변수가 독립적이다 -> 공분산이 0이 된다.
					// 공분산이 0이 된다고 두 변수가 항상 독립적이라 할 수는 없다. 어디까지나 이건 엄밀하게며 딥러닝에선 거의 항상 독립적으로 봐도 무방할 거라 본다.

	확률변수 x 의 평균(기대값), Y의 평균을 각각
		E(X) = m, E(Y) = v
	이라 했을 때, X, Y의 공분산은 아래와 같다.
		Cov(X, Y) = E((X - m) * (Y - v))
	"즉, 공분산은 X의 편차와 Y의 편차를 곱한것의 평균이라는 뜻이다."
	X, Y 가 같은 즉 자기 자신이라면 어떻게 되나 보면
		Cov(X, X) = E((X - m) * (X - m))		// 편차의 제곱의 평균이 되며 이는 분산이 된다.

다시 정리하자면
X 와 Y가 독립이면 공분산은 0이 된다. // 역은 성립안된다.
공분산의 문제점은 X와 Y의 단위의 크기에 영향을 받는다는 것이다.
"100점 만점인 두 과목의 점수 공분산은 별로 상관성이 부족하한데도 불구하고 100점 만점이기 때문에 큰 공분산이 나오고"
"10점 만점인 두 과목의 점수 공분산은 크게 상관성이 있음에도 불구하고 10점 만점이기 때문에 작은 공분사이 나올수 있다"

이것을 보완하기 위해 상관계수(Correlation) 가 나타난다.
	확률변수의 절대적 크기에 영향을 받지 않도록 단위화 시켰다고 생각하면 된다.
	즉, 분산의 크기만큼 나누었다고 생각하면 된다.
	정의
		p = Cov(X, Y) / sqrt( Var(X) * Var(Y) )
		-1 <= p <= 1
	성질
		1. 상관계수의 절대값은 1을 넘을 수 없다.
		2. 확률변수 X, Y가 독립이라면 상관계수는 0이다.
		3. X와 Y가 선형적 관계라면 상관계수는 1 혹은 -1 이다.
			// 양의 선형관계면 1, 음의 선형관계면 -1
