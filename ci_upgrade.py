import os
import pty
import sys
import time
import select

def run_as_user(command):
    password = os.environ.get('SUDO_PASSWORD')
    if not password:
        print("Error: SUDO_PASSWORD env var not set")
        sys.exit(1)

    # We use 'su - alexey -c ...' to run as alexey with full environment
    # The 'pty' module allows us to fool 'su' into thinking it's running in a real terminal
    master, slave = pty.openpty()
    
    pid = os.fork()
    if pid == 0:
        # Child process (su)
        os.close(master)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(slave)
        # Execute su
        os.execlp("su", "su", "-", "alexey", "-c", command)
    else:
        # Parent process (controller)
        os.close(slave)
        
        # Simple loop to read output and send password
        password_sent = False
        output_buffer = b""
        
        while True:
            r, w, e = select.select([master], [], [], 1)
            if master in r:
                try:
                    data = os.read(master, 1024)
                    if not data:
                        break
                    output_buffer += data
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                    
                    if not password_sent and (b"Password:" in output_buffer or b"password" in output_buffer.lower()):
                        os.write(master, password.encode() + b"\n")
                        password_sent = True
                        output_buffer = b"" # Reset buffer after sending
                        
                except OSError:
                    break
            else:
                # Check child status
                if os.waitpid(pid, os.WNOHANG)[0] != 0:
                    break
                    
        # Wait for child to exit and return its code
        _, status = os.waitpid(pid, 0)
        exit_code = os.waitstatus_to_exitcode(status)
        sys.exit(exit_code)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ci_upgrade.py <command_string>")
        sys.exit(1)
        
    cmd = sys.argv[1]
    run_as_user(cmd)
