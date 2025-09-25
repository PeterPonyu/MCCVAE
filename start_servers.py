
#!/usr/bin/env python3
"""
Dual Server Launcher
Starts both FastAPI backend and frontend static files server
Runs from project root using module paths like api.main:app
"""

import os
import sys
import time
import signal
import threading
import subprocess
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

class ColoredOutput:
    """Simple colored terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=ColoredOutput.CYAN):
    """Print colored log message"""
    print(f"{color}[LAUNCHER]{ColoredOutput.ENDC} {message}")

class StaticFileHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving static files with proper headers"""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for API requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.end_headers()

def start_frontend_server(frontend_dir="./out", port=3000):
    """Start static file server for frontend"""
    frontend_path = Path(frontend_dir).resolve()
    
    if not frontend_path.exists():
        log(f"❌ Frontend directory not found: {frontend_path}", ColoredOutput.FAIL)
        log(f"💡 Make sure you've run 'npm run build' to generate the 'out' folder", ColoredOutput.WARNING)
        return False
    
    try:
        # Save current directory and change to frontend
        original_dir = os.getcwd()
        os.chdir(frontend_path)
        
        handler = partial(StaticFileHandler, directory=str(frontend_path))
        httpd = HTTPServer(('localhost', port), handler)
        
        log(f"🚀 Frontend server starting at http://localhost:{port}")
        log(f"📁 Serving files from: {frontend_path}")
        
        httpd.serve_forever()
        
    except OSError as e:
        if e.errno == 48:  # Address already in use
            log(f"❌ Port {port} is already in use", ColoredOutput.FAIL)
        else:
            log(f"❌ Frontend server error: {e}", ColoredOutput.FAIL)
        return False
    except KeyboardInterrupt:
        log("🛑 Frontend server stopped")
        os.chdir(original_dir)
        return True
    finally:
        os.chdir(original_dir)

def start_backend_server(api_module="api", port=8000):
    """Start FastAPI backend server using module path"""
    api_path = Path(api_module)
    
    if not api_path.exists():
        log(f"❌ API directory not found: {api_path}", ColoredOutput.FAIL)
        return False
    
    # Look for main.py or app.py in the api folder
    main_files = ['main.py', 'app.py', 'server.py', 'api.py', '__init__.py']
    main_file = None
    app_var = 'app'  # Default FastAPI app variable name
    
    for file in main_files:
        if (api_path / file).exists():
            main_file = file.replace('.py', '')
            break
    
    if not main_file:
        log(f"❌ No FastAPI main file found in {api_path}", ColoredOutput.FAIL)
        log(f"💡 Looking for: {', '.join(main_files)}", ColoredOutput.WARNING)
        return False
    
    # Construct module path: api.main:app
    module_path = f"{api_module}.{main_file}:{app_var}"
    
    try:
        log(f"🚀 Backend API server starting at http://localhost:{port}")
        log(f"📁 Working directory: {os.getcwd()}")
        log(f"🐍 Module path: {module_path}")
        
        # Try uvicorn with module path
        try:
            cmd = [
                'uvicorn', 
                module_path, 
                '--host', '0.0.0.0', 
                '--port', str(port), 
                '--reload'
            ]
            
            log(f"🔧 Command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                cwd=os.getcwd()  # Stay in project root
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"{ColoredOutput.GREEN}[API]{ColoredOutput.ENDC} {line.rstrip()}")
            
        except FileNotFoundError:
            log("⚠️  uvicorn not found - install with: pip install uvicorn", ColoredOutput.WARNING)
            log("💡 Trying alternative: python -m uvicorn", ColoredOutput.WARNING)
            
            try:
                cmd = [
                    'python', '-m', 'uvicorn', 
                    module_path, 
                    '--host', '0.0.0.0', 
                    '--port', str(port), 
                    '--reload'
                ]
                subprocess.run(cmd, cwd=os.getcwd())
            except Exception as e:
                log(f"❌ Failed to start with python -m uvicorn: {e}", ColoredOutput.FAIL)
                return False
            
    except KeyboardInterrupt:
        log("🛑 API server stopped")
        return True
    except Exception as e:
        log(f"❌ API server error: {e}", ColoredOutput.FAIL)
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    log("🔍 Checking dependencies...")
    
    # Check current working directory
    log(f"📁 Working directory: {os.getcwd()}")
    
    # Check if uvicorn is available
    try:
        subprocess.run(['uvicorn', '--version'], capture_output=True, check=True)
        log("✅ uvicorn found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(['python', '-m', 'uvicorn', '--version'], capture_output=True, check=True)
            log("✅ uvicorn found (via python -m)")
        except:
            log("⚠️  uvicorn not found - install with: pip install uvicorn", ColoredOutput.WARNING)
    
    # Check frontend build
    frontend_out = Path('./out')
    if frontend_out.exists():
        log("✅ Frontend 'out' directory found")
        index_html = frontend_out / 'index.html'
        if index_html.exists():
            log("✅ index.html found")
        else:
            log("⚠️  index.html not found in out/ folder", ColoredOutput.WARNING)
    else:
        log("⚠️  Frontend 'out' folder not found - run: npm run build", ColoredOutput.WARNING)
    
    # Check API directory
    api_dir = Path('./api')
    if api_dir.exists():
        log("✅ API directory found")
        
        # Check for main files
        main_files = ['main.py', 'app.py', 'server.py', 'api.py']
        found_files = []
        for file in main_files:
            if (api_dir / file).exists():
                found_files.append(file)
        
        if found_files:
            log(f"✅ Found API files: {', '.join(found_files)}")
        else:
            log("⚠️  No main API files found in api/ directory", ColoredOutput.WARNING)
    else:
        log("⚠️  API directory not found", ColoredOutput.WARNING)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    log("\n🛑 Shutting down servers...", ColoredOutput.WARNING)
    sys.exit(0)

def main():
    """Main launcher function"""
    print(f"""
{ColoredOutput.HEADER}
╔══════════════════════════════════════════════════════════════╗
║                    🚀 DUAL SERVER LAUNCHER                   ║
║                      FastAPI + Next.js                      ║
║                   (Running from project root)               ║
║                                                              ║
║  Backend (FastAPI):  http://localhost:8000                  ║
║  Frontend (Static):  http://localhost:3000                  ║
║  API Docs:           http://localhost:8000/docs             ║
╚══════════════════════════════════════════════════════════════╝
{ColoredOutput.ENDC}
    """)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    check_dependencies()
    
    # Configuration - Stays in project root
    API_MODULE = "api"         # Module name (folder)
    FRONTEND_DIR = "./out"     # Static build folder
    BACKEND_PORT = 8000
    FRONTEND_PORT = 3000
    
    # Start backend server in a separate thread
    backend_thread = threading.Thread(
        target=start_backend_server, 
        args=(API_MODULE, BACKEND_PORT),
        daemon=True
    )
    
    # Start frontend server in a separate thread
    frontend_thread = threading.Thread(
        target=start_frontend_server, 
        args=(FRONTEND_DIR, FRONTEND_PORT),
        daemon=True
    )
    
    try:
        # Start both servers
        log("🎬 Starting API server...")
        backend_thread.start()
        
        time.sleep(3)  # Give backend more time to start
        
        log("🎬 Starting frontend server...")
        frontend_thread.start()
        
        time.sleep(2)  # Give frontend time to start
        
        # Open browser
        log("🌐 Opening browser...")
        webbrowser.open(f'http://localhost:{FRONTEND_PORT}')
        
        log("✅ Both servers are running!")
        log("💡 Press Ctrl+C to stop both servers")
        log(f"📚 API Documentation: http://localhost:{BACKEND_PORT}/docs")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log("👋 Shutting down gracefully...")
    
    except Exception as e:
        log(f"❌ Launcher error: {e}", ColoredOutput.FAIL)

if __name__ == "__main__":
    main()
