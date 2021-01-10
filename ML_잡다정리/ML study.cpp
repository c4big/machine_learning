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
	
