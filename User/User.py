from time import ctime, time


class UserException(Exception):
    pass


class User:
    psw_filename = './psw'
    record_filename = './record.txt'
    spilt_sign = ' '

    def __init__(self):
        self.user_name = None
        self.admin_name = None
        self.user_psw = self.load_user_psw_pair()
        self.last_sign_in_user_name = None

    def login(self, user_name, psw):
        if user_name not in self.user_psw.keys():
            raise UserException('无此用户')
        if self.user_psw[user_name] != psw:
            raise UserException('密码错误')
        else:
            self.user_name = user_name
            self.save_record('登录')
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
                    if self.admin_name is None:
                        self.admin_name = user
        except FileNotFoundError:
            pass
        return user_psw

    def save_user_psw_file(self):
        with open(self.psw_filename, 'wb') as f:
            for user, psw in self.user_psw.items():
                line = user + self.spilt_sign + psw + '\n'
                f.write(line.encode())

    def change_psw(self, new_psw, user_name=None):
        self.user_psw[user_name if user_name else self.user_name] = new_psw
        self.save_user_psw_file()
        self.save_record('修改了 ' + user_name if user_name else self.user_name + ' 的密码')

    def sign_in(self, user_name, psw):
        self.last_sign_in_user_name = user_name
        if self.user_name is None:
            self.user_name = user_name
        self.change_psw(psw, user_name)
        self.save_record('添加了用户 ' + user_name)

    def save_record(self, record):
        with open(self.record_filename, 'a') as f:
            record = self.spilt_sign.join((ctime(time()), self.user_name, record)) + '\n'
            f.write(record)
        return record

    def get_record(self):
        with open(self.record_filename, 'r') as f:
            return f.readlines()

    def delete_user(self, user_name):
        self.user_psw.pop(user_name)
        self.save_user_psw_file()
