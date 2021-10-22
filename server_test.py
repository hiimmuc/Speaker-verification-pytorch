from fastapi import FastAPI #import class FastAPI() từ thư viện fastapi
import subprocess
import glob
import os
from tqdm import tqdm

app = FastAPI() # gọi constructor và gán vào biến app


@app.get("/") # giống flask, khai báo phương thức get và url
async def root(): # do dùng ASGI nên ở đây thêm async, nếu bên thứ 3 không hỗ trợ thì bỏ async đi
    return {"message": "Hello World"}

def main():
    subprocess.call(f"uvicorn server_test:app --host 0.0.0.0 --port 8111", shell=True)
    
if __name__ == '__main__':
    main()