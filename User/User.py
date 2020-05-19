from time import ctime, time


class UserException(Exception):
    pass


class User:
    psw_filename = './psw'
    record_filename = './record.txt'
    spilt_sign = ' '

    def __init__(self):
        self.user_name = None
        self.user_psw = self.load_user_psw_pair()

    def login(self, user_name, psw):
        if user_name not in self.user_psw.keys():
            raise UserException('无此用户')
        if self.user_psw[user_name] != psw:
            raise UserException('密码错误')
        else:
            self.user_name = user_name
            return True

    def load_user_psw_pair(self):
        user_psw = dict()
        try:
            with open(self.psw_filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # 此处与定义的self.spilt_sign不同步
                    user, psw = line.split()
                    user_psw[user] = psw
        except FileNotFoundError:
            pass
        return user_psw

    def save_user_psw_file(self):
        with open(self.psw_filename, 'wb') as f:
            for user, psw in self.user_psw.items():
                line = user + self.spilt_sign + psw
                f.write(line.encode())

    def change_psw(self, new_psw):
        self.user_psw[self.user_name] = new_psw
        self.save_user_psw_file()

    def sign_in(self, user_name, psw):
        self.user_name = user_name
        self.change_psw(psw)

    def save_record(self, record):
        with open(self.record_filename, 'a') as f:
            f.write(self.spilt_sign.join((ctime(time()), self.user_name, record)))
