1강
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
			
			
2강
주요내용
01: 데이터 가져오기	// 복사시 누락되있을 경우 Add Data 를 클릭하여 챌린지데이터에서 가져온다
02: 주요 모듈 임포트	// 기본 모듈, 시각화 모듈, 알고리즘 모듈
03: 보조 링크 파악	// 각 종 외부 링크를 통하여 추가 학습 가능
04: csv를 데이터 프레임으로 전환	// csv 파일을 판다스 데이터 프레임으로 전환