import random
import time
import json
import os
import pyperclip
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ================= 설 정 부 분 =================
# 1. 크롬 프로필 경로 (반드시 본인 경로로 수정하세요!)
CHROME_PROFILE_PATH = r"C:\Selenium_Profile"

# 2. 프로필 폴더명 (기본값 Default)
PROFILE_DIRECTORY = "Default"

# 3. 저장할 JSON 파일 경로
JSON_FILE_PATH = r''

# 4. 대상 URL
TARGET_URL = ""

# ===============================================


def get_default_chrome_options():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    return options



def test_args():
    options = get_default_chrome_options()

    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    driver.get('http://selenium.dev')

    driver.quit()



def setup_driver():
    """사용자 프로필을 로드한 크롬 드라이버 설정"""
    print("브라우저 설정을 초기화합니다...")
    
    chrome_options =  webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(f"user-data-dir={CHROME_PROFILE_PATH}")
    
    # 봇 탐지 회피 옵션 (혹시 모를 차단 방지)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    chrome_options.add_experimental_option("detach", True)

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    print("driver success...")
    driver.get(TARGET_URL)
    return driver

def save_to_json(content):
    """결과를 JSON 파일에 저장"""
    data = []
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list): data = [data]
        except:
            data = []

    new_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": "안녕",  # 어떤 질문이었는지도 기록
        "content": content
    }
    data.append(new_entry)

    with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 결과가 {JSON_FILE_PATH}에 저장되었습니다.")





def main():
    print("="*50)
    print("⚠️  주의: 실행 전에 모든 크롬 브라우저 창을 닫아주세요.")
    print("="*50)
    
    try:
        driver = setup_driver()
        print(f"🚀 브라우저 실행 완료! 페이지 로딩 대기 중 (10초)...")
        time.sleep(10) 

        # ==========================================
        # json file import
        user_prompt = "안녕" 
        print(f"🤖 테스트 프롬프트 전송: {user_prompt}")
        # ==========================================

        try:
            # 1. 입력창 찾기 및 입력
            textarea = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[aria-label='Enter a prompt']"))
            )
            textarea.click()
            textarea.clear()
            textarea.send_keys(user_prompt)
            time.sleep(0.5)

            # 2. 실행(Run) 버튼 클릭
            run_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Run']")
            run_button.click()
            print("⏳ 질문 전송 완료. 답변 생성을 기다립니다...")
            
            # 3. 답변 대기
            time.sleep(random.uniform(130, 140))  

            # 4. Copy as text 수행
            turns = driver.find_elements(By.CSS_SELECTOR, "ms-chat-turn")
            if not turns:
                print("❌ 대화 내역을 찾을 수 없습니다.")
                return # 함수 종료
            
            last_turn = turns[-1]

            # 'More options' 버튼 클릭
            more_btn = last_turn.find_element(By.CSS_SELECTOR, "button[aria-label='Open options']")
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(random.uniform(0.5, 1.4)) 

            # 'Copy as text' 버튼 클릭
            copy_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'Copy as text')]]"))
            )
            copy_btn.click()
            print("📋 클립보드 복사 버튼 클릭 완료.")
            time.sleep(random.uniform(0.5, 1.4))

            # 5. 저장
            result_text = pyperclip.paste()
            save_to_json(result_text)
            
            print(f"\n[수집된 데이터 미리보기]\n{result_text[:100]}...")

        except Exception as e:
            print(f"❌ 작업 중 오류 발생: {e}")

    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")
    finally:
        print("테스트 종료.")
        # driver.quit() # 브라우저를 닫으려면 주석 해제

if __name__ == "__main__":
    main()