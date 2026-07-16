import sys
import subprocess
import time
import os

def run():
    print("======================================================================")
    print("                      VoltIQ Starter Process Manager                  ")
    print("======================================================================")
    
    # Define ports
    api_port = "8000"
    streamlit_port = "8501"
    
    # Change CWD to the directory of run.py to run everything relative to it
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Start Backend API
    print(f"[*] Starting FastAPI backend on http://127.0.0.1:{api_port}...")
    api_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app", 
        "--host", "127.0.0.1", "--port", api_port, "--reload"
    ]
    api_process = subprocess.Popen(api_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    # Give the backend API a second to spin up
    time.sleep(1.5)
    
    # Start Streamlit Dashboard
    print(f"[*] Starting Streamlit frontend on http://127.0.0.1:{streamlit_port}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
        "--server.port", streamlit_port, "--server.address", "127.0.0.1"
    ]
    streamlit_process = subprocess.Popen(streamlit_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    print("\n[+] VoltIQ is running! Press Ctrl+C to terminate both servers.")
    
    try:
        while True:
            # Check if either process has terminated
            api_status = api_process.poll()
            st_status = streamlit_process.poll()
            
            if api_status is not None:
                print(f"\n[!] Backend API process stopped unexpectedly with code: {api_status}")
                break
            if st_status is not None:
                print(f"\n[!] Streamlit process stopped unexpectedly with code: {st_status}")
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[!] Shutdown signal received. Terminating servers...")
    finally:
        # Gracefully shut down both subprocesses
        if api_process.poll() is None:
            api_process.terminate()
            api_process.wait()
            print("[+] Backend API terminated.")
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait()
            print("[+] Streamlit frontend terminated.")
        print("[+] Goodbye!")

if __name__ == "__main__":
    run()
