import sys

class Logger:
    msg = []
    prevmsgl = 0

    @classmethod
    def to_String(self, *m):
        b = ''
        for i in m:
            b += str(i)
        return b
    
    @classmethod
    def add_to_log(self, *m):
        Logger.msg.append(Logger.to_String(*m))

    @classmethod
    def print_log(self):
        for _ in range(Logger.prevmsgl):
            sys.stdout.write("\x1b[1A\x1b[2K")

        str_msg = ""
        for m in Logger.msg:
            str_msg += m + "\n"
        
        sys.stdout.write(str_msg)
        sys.stdout.flush()

        Logger.prevmsgl = len(Logger.msg)
        Logger.msg.clear()
    

# import sys

# def to_String(*m):
#     b = ''
#     for i in m:
#         b += str(i)
#     return b

# def add_to_log(*m):
#     msg.append(to_String(*m))

# def print_log():
#     for _ in range(prevmsgl):
#         sys.stdout.write("\x1b[1A\x1b[2K")

#     str_msg = ""
#     for m in msg:
#         str_msg += m + "\n"
    
#     sys.stdout.write(str_msg)
#     sys.stdout.flush()

#     prevmsgl = len(msg)
#     msg.clear()

# if __name__ == "__main__":
#     msg = []
#     prevmsgl = 0

# import curses

# def to_String(*m):
#     b = ''
#     for i in m:
#         b += str(i)
#     return b

# def add_to_log(*m):
#     msg.append(to_String(*m))

# def print_log():

#     try:
#         for m in msg:
#             stdscr.addstr(0, 0, m)
#     finally:
#         stdscr.refresh()


#     #str_msg = "\0337\n"
#     # print('? .\0338')
#     # print("\0337")
#     for m in msg:
#         print(m)
#         #str_msg += m + "\n"
    
#     print("afq  weferqwfeqeferef", end='')
#     print('\b'*1000)

#     # print(msg)
#     #sys.stdout.write(str_msg)
#     #sys.stdout.write("\b" * len(str_msg))
#     # sys.stdout.flush()
#     msg.clear()

# if __name__ == '__main__':
#     msg = []

#     stdscr = curses.initscr()
#     curses.noecho()
#     curses.cbreak()

# import sys
# import subprocess
# subprocess.run('', shell = True)

# msg = []

# def to_String(*m):
#     b = ''
#     for i in m:
#         b += str(i)
#     return b

# def add_to_log(*m):
#     msg.append(to_String(*m))

# def print_log():
#     #str_msg = "\0337\n"
#     # print('? .\0338')
#     # print("\0337")
#     for m in msg:
#         print(m)
#         #str_msg += m + "\n"
    
#     print("afq  weferqwfeqeferef", end='')
#     print('\b'*1000)

#     # print(msg)
#     #sys.stdout.write(str_msg)
#     #sys.stdout.write("\b" * len(str_msg))
#     # sys.stdout.flush()
#     msg.clear()