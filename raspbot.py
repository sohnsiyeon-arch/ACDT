#라이브러리 불러오기
import sys
import time
import serial

import requests # ??

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *

server_ip = '10.165.164.60'

ser = serial.Serial('/dev/ttyACM0', 9600)

# 2. 서버 및 학생 설정
SERVER_URL = f"http://{server_ip}:8000/raspbot"
SERVER_CLEAR_URL = f"http://{server_ip}:8000/alert"

student_num = {
    "Junsang Park": 1,  # A 위치, 이름은 학생 웹이랑 일치해야 함. 
    "Siyeon Sohn": 2    # B 위치
}

last_known_status = {}

current_robot_pos = 1 
speed = 10

#라인트래킹 함수
def line_tracking(duration=3):
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # 센서 데이터 읽기
            try:
                track_data = bot.read_data_array(0x0a, 1)
                track = int(track_data[0])

            x1 = (track >> 3) & 0x01  
            x2 = (track >> 2) & 0x01  
            x3 = (track >> 1) & 0x01  
            x4 = track & 0x01        

            lineL1=x2
            lineL2=x1
            lineR1=x3
	    lineR2=x4

            if lineL1 == 0 and lineL2 == 0 and lineR1 == 0 and lineR2 == 0:
                move_forward(speed)
            elif (lineL2 == 0 or lineL1 == 0) and lineR2 == 0:
                rotate_right(speed)
                time.sleep(0.05)
            elif lineL1 == 0 and (lineR2 == 0 or lineR1 == 0):
                rotate_left(int(speed*1.5))
                time.sleep(0.15)
            elif lineL1 == 0:
                rotate_left(speed)
                time.sleep(0.02)
            elif lineR2 == 0:
                rotate_right(speed)
                time.sleep(0.01)
            elif lineL2 == 0 and lineR1 == 1:
                rotate_left(speed)
            elif lineL2 == 1 and lineR1 == 0:
                rotate_right(speed)
            elif lineL2 == 0 and lineR1 == 0:
                move_forward(speed)
            
            time.sleep(0.01)
    finally:
        stop()
        print("Line tracking finished")

# [시나리오 1] 준상이가 문제일 때
def move_A_to_B():
    print("Moving A -> B")
    line_tracking(5)
    
    for x in range(3):
        move_backward(15)
        time.sleep(0.3)
        stop()
        
        if ser: ser.write(b'n') # 시리얼 명령
        time.sleep(0.5)
        
        move_forward(30)
        time.sleep(0.5)
    stop()    

    time.sleep(0.1)
    move_backward(30)
    time.sleep(0.5)
    rotate_left(10) # B에서는 오른쪽 회전으로 복귀 준비?
    time.sleep(0.8)
    stop()
    line_tracking(0.5)
    stop()

# [시나리오 2] 시연이가 문제일 때
def move_B_to_A():
    print("Moving B -> A")
    line_tracking(5)
    
    for x in range(3):
        move_backward(15)
        time.sleep(0.3)
        stop()
        
        if ser: ser.write(b'n') # 시리얼 명령
        time.sleep(0.5)
        
        move_forward(30)
        time.sleep(0.5)
    stop()    

    time.sleep(0.1)
    move_backward(30)
    time.sleep(0.5)
    rotate_left(10) # B에서는 오른쪽 회전으로 복귀 준비?
    time.sleep(0.8)
    stop()
    line_tracking(0.5)
    stop()

# [시나리오 3] 제자리 경고 (A->A 또는 B->B)
def warn_in_place():
    print("Moving B -> A")
    # 시작 동작 (B->A 코드 참조)
    move_backward(10)
    time.sleep(0.5)
    rotate_left(30)
    time.sleep(0.6)
    
    line_tracking(3)
    perform_warning_action()
    
    # 마무리 동작
    time.sleep(0.1)
    move_backward(30)
    time.sleep(0.5)
    rotate_right(10)
    time.sleep(0.8)
    line_tracking(0.5)
    stop()

# [시나리오 4] 둘다 자고 있을 때
# line_tracking(0.8) ~~ 참조

# ==========================================
# 4. 메인 실행 루프
# ==========================================
print(f"Raspbot 가동 시작 (초기 위치: {current_robot_pos})")

try:
    while True:
        try:
            # 1. 서버 확인
            response = requests.get(SERVER_URL, timeout=0.8)
            alert_board = {}
            if response.status_code == 200:
                alert_board = response.json()

            if not alert_board:
                time.sleep(1)
                continue

            # 2. 경고 처리
            for name, info in alert_board.items():
                current_status = info['status']

                # 상태 변화 감지 및 로깅
                if last_known_status.get(name) != current_status:
                    last_known_status[name] = current_status
                    print(f"\n👀 [감지] {name} -> {current_status}")

                # 나쁜 행동 감지 시 출동
                if current_status in ['phone', 'sleeping', 'staring', 'talking']:
                    
                    target_pos = student_num.get(name, 0)
                    
                    if target_pos != 0:
                        print(f"[출동] {name} ({current_status}) @ Pos {target_pos}")
                        
                        # --- 로봇 이동 로직 ---
                        if current_robot_pos == 1 and target_pos == 2:
                            # A에 있는데 B가 문제 -> 이동
                            move_A_to_B()
                            current_robot_pos = 2
                            
                        elif current_robot_pos == 2 and target_pos == 1:
                            # B에 있는데 A가 문제 -> 이동
                            move_B_to_A()
                            current_robot_pos = 1
                            
                        else:
                            # 같은 위치거나 (1->1, 2->2) 그 외 -> 제자리 경고
                            warn_in_place()
                        
                        print(f"   현재 로봇 위치: {current_robot_pos}")
                        
                        # --- 경고 삭제 및 기억 리셋 ---
                        try:
                            requests.delete(f"{SERVER_CLEAR_URL}/{name}", timeout=0.5)
                            if name in last_known_status:
                                del last_known_status[name]
                            print(f" {name} 처리 완료")
                        except Exception as e:
                            print(f"서버 통신 오류: {e}")
                    
                    else:
                        print(f"위치 모름: {name}")

        except Exception as e:
            # print(f"루프 에러: {e}") # 너무 자주 뜨면 주석 처리
            pass

        time.sleep(1)

except KeyboardInterrupt:
    print("프로그램 종료")
    stop()
    if 'bot' in globals():

        del bot
