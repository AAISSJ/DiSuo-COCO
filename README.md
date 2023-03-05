# DiSuo-COCO

- 🥇 1st prize in 2021 AI applied App/Web Development Hackathon 🥇
- Pose Alignment with OpenPose
  - You need to install OpenPose API in your local desktop ( Reference: https://m.blog.naver.com/rhrkdfus/221531159811 / https://hanryang1125.tistory.com/m/2 )

# 기획 의도 및 개발 목표 
## 기획 의도 
1. 코로나로 비대면/재택 근무 증가 & 전자기기 사용시간 증가 
![image](https://user-images.githubusercontent.com/76966915/222941232-5fff9dd5-3add-4279-a8d6-d97acee543c1.png)

2. 거북목 증후군, 척추 측만증 등 잘못된 자세로 인한 환자 증가 - 올바른 자세에 대한 관심 증가 
![image](https://user-images.githubusercontent.com/76966915/222941238-d64e2203-3e05-4514-a5c3-fdf55f8cb8b0.png)

3. 기존의 자세 교정 웹/앱의 한계 - 홈트레이닝에만 집증 & 정적인 사진 분석만 지

## 개발 목표 
1. Pose Estimation을 통해 일상 속에서의 자세 교정을 위한 웹페이지인 DiSuo 개발
2. 자세 교정을 통해 바른 자세 생활화 & 고비용의 수술이 필요한 디스크 관련 질환 예방
![image](https://user-images.githubusercontent.com/76966915/222941239-03da7a7c-f3f0-47d7-8493-35054ad03ef2.png)

3. 정적인 사진과 동적인 영상을 받아 실시간 자세 분석 기능이 있는 웹인 DiSuo 개발

# 웹 구상 
## 주요 기능 
1. Static Version 
   - 사진을 업로드하여 사용자의 자세를 분석하는 모드
   - 여러 자세를 분석하며 올바르지 않은 자세에 대해선 문구로 알림
   - 올바르지 않은 자세에 따른 맞춤 자세 교정 동영상 추천
2. Live Version 
   - 웹 카메라를 이용하여 실시간으로 사용자의 자세를 분석하는 모드
   - 여러 자세를 분석하며 올바르지 않은 자세에 대해선 문구로 알림
   - 올바르지 않은 자세에 따른 맞춤 자세 교정 동영상 추천 / 타이머 기능으로 알람 설정 가능
3. 내 주변 맞춤 병원 추천
   - 주소 입력 시 가까운 위치의 자세 교정 병원 추천(정형외과/한의원/통증의학과)

## 개발 환경 

1. 백엔드
   - 인공지능 : Python OpenCV, OpenPose 
   - 서버 : Python Flask
1. 프론트엔드 : html, css, javascript, bootstrap, jquery

# 사용 기술 
![image](https://user-images.githubusercontent.com/76966915/222941307-19837548-d7cd-4bf3-ba0a-82acdd63f60f.png)
![image](https://user-images.githubusercontent.com/76966915/222941324-36a403fd-21db-467e-ac26-0d9c4a8ab93e.png)
![image](https://user-images.githubusercontent.com/76966915/222941336-72518b08-608f-40f7-8c5b-c98131cddb9c.png)
![image](https://user-images.githubusercontent.com/76966915/222941349-bb1ee141-7f3e-4913-ba9d-45bad112a9b5.png)
![image](https://user-images.githubusercontent.com/76966915/222941353-f3e0a28b-94c9-48f8-b5dc-57bf64cdad5b.png)



# 웹 세부 기능 
![image](https://user-images.githubusercontent.com/76966915/222941367-dccaaf3b-1009-4f62-80d9-b16e81716438.png)
![image](https://user-images.githubusercontent.com/76966915/222941378-5101140b-1fa8-4f16-bc9a-64329d64a572.png)
![image](https://user-images.githubusercontent.com/76966915/222941389-d62fa596-ff8f-4704-ae77-01956b14804e.png)
![image](https://user-images.githubusercontent.com/76966915/222941399-b9ec6d4d-04d2-4cca-b7d2-8f0c8a2613a1.png)
![image](https://user-images.githubusercontent.com/76966915/222941407-9a614d76-f255-4bf9-979a-a3a40d5d12a6.png)
![image](https://user-images.githubusercontent.com/76966915/222941424-bc5ae0df-a33c-4407-8649-12cd38626583.png)
![image](https://user-images.githubusercontent.com/76966915/222941435-c65d4edb-79b4-4091-9c1c-a444f935335a.png)






