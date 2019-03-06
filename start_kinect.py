import subprocess
com='bin/RelWithDebInfo/Protonect'

if __name__ == '__main__':
    p = subprocess.Popen([com, 'cpu']) # running background.
    p_stdout = p.communicate()[0]
