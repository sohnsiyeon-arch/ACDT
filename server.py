import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()

# static 폴더 마운트 (CSS 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =================================================================
# 1. 기본 설정 (CORS & 저장소)
# =================================================================
# CORS: 학생들이 각자 다른 와이파이/IP에서 접속해도 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 칠판 (학생 상태 저장소 - 메모리)
student_board = {}
# 경고 받은 학생 기록용 (로봇이 이걸 보고 출동함)
alert_board = {}

# =================================================================
# 2. 데이터 규격 (Schema)
# =================================================================
class StudentData(BaseModel):
    name: str
    status: str
    prob: str
    detail_probs: dict = {} 

class AlertData(BaseModel):
    name: str
    status: str

# =================================================================
# 3. API 엔드포인트
# =================================================================

# [메인화면] 대시보드 웹페이지 띄우기
@app.get("/")
async def show_dashboard():
    if not os.path.exists("dashboard.html"):
        return HTMLResponse("<h1> dashboard.html 파일이 같은 폴더에 없습니다</h1>")
    
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# [받기] 학생이 0.5초마다 데이터 보내는 곳
@app.post("/update")
async def receive_data(data: StudentData):
    student_board[data.name] = {
        "status": data.status,
        "prob": data.prob,
        "details": data.detail_probs
    }
    return {"msg": "success"}

# [보여주기] 대시보드(HTML)가 데이터 가져가는 곳
@app.get("/monitor")
async def get_monitor_data():
    return JSONResponse(content=student_board)

# [삭제] 대시보드에서 특정 학생 모니터링 데이터 삭제 시
@app.delete("/monitor/{name}")
async def clear_student_monitor_data(name: str):
    if name in student_board:
        del student_board[name]
        # 모니터링 끄면 경고도 같이 꺼주는 센스
        if name in alert_board:
            del alert_board[name]
        print(f"🧹 학생 데이터 삭제: {name}")
        return {"msg": f"Monitor data for {name} cleared"}
    return {"msg": f"Monitor data for {name} not found"}, 404

# -----------------------------------------------------------------
# [핵심] 경고 및 로봇 관련 (자동 출동 시스템)
# -----------------------------------------------------------------

# [경고 등록] 학생이 5초 이상 딴짓하면 대시보드가 여기로 보냄
@app.post("/alert")
async def alert_receive_data(jbg: AlertData):
    alert_board[jbg.name] ={
        "status": jbg.status
    }
    print(f"🚨 경고 접수: {jbg.name} -> {jbg.status}")
    return {"msg": "Alert Saved"}

# [로봇 조회] 로봇이 어디로 갈지 계속 물어보는 곳
@app.get("/raspbot")
async def sendto_raspbot():
    # 경고판(alert_board)을 그대로 줍니다. (명단이 있으면 로봇이 알아서 감)
    return JSONResponse(content=alert_board)

# [처리 완료] 로봇이 가서 경고했으면, 명단에서 지우라고 요청하는 곳
@app.delete("/alert/{name}")
async def clear_alert_by_name(name: str):
    # robot_queue가 아니라 alert_board에서 지우도록 수정
    if name in alert_board:
        del alert_board[name]
        print(f"로봇 처리 완료: {name} (명단 삭제)")
    return {"msg": f"Alert for {name} cleared"}

# [상황 종료] 선생님이 '상황 종료' 버튼 누르면 초기화
@app.post("/reset")
async def reset_all_data():
    global alert_board
    
    # 경고판 초기화
    alert_board.clear()
            
    print("[System] 상황 종료! 모든 경고가 초기화되었습니다.")
    return {"msg": "All Alerts Cleared"}

# =================================================================
# 4. 서버 실행
# =================================================================
if __name__ == "__main__":
    print(">>> 서버 가동 시작 (학생들은 이 컴퓨터의 IP 주소를 입력하세요)")
    # host="0.0.0.0"이어야 외부(학생 노트북)에서 접속 가능

    uvicorn.run(app, host="0.0.0.0", port=8000)
