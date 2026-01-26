# 2023-11-29    定义常量
# 2024-01-18    使用os, 以兼容Win和Ubuntu
# 2024-04-09    由于集思录机制变化，改为利用当前FIREFOX WIN 用户的profile登录，避免只能获取部分数据
#               需要手动启动Firefox,登录并修改数据页面的“自定义列表”，改为全选
#               增加FIREFOX_PROFILE_PATH定义
# 2024-06-14    增加回测配置定义


'''
Ubuntu 设置PYTHONPATH环境变量：
你也可以通过设置PYTHONPATH环境变量来修改Python的搜索路径。PYTHONPATH环境变量是一个包含若干路径的列表，Python会在这些路径中搜索模块。
打开你的~/.bashrc文件并在其中添加以下行：
export PYTHONPATH="${PYTHONPATH}:/path/to/your/module"

然后运行source ~/.bashrc以更新你的会话以引入新的变量。
注意/path/to/your/module应该被替换为放置constants.py的目录的绝对路径。
请注意，上述两种方法都需要把constants.py所在的目录添加到路径中，而不是constants.py文件本身。 (已编辑)

'''

import os
homedir = "//NUC-SERVER"
pythondir = os.path.join(homedir,'Trading')

LOG_DIR = os.path.join(pythondir, "Log")
DATA_DIR = os.path.join(pythondir, "Data")


# Log Files
DATA_LOG_FILE = os.path.join(LOG_DIR,"data.log")
TRADING_LOG_NAME = 'TRADING'
TRADING_LOG_FILE = os.path.join(LOG_DIR,"trading.log")
DATA_LOG_NAME = 'DATA'
DATA_LOG_FILE = os.path.join(LOG_DIR,"data.log")
QMT_LOG_NAME = 'QMT'
QMT_LOG_FILE = os.path.join(LOG_DIR,"qmt.log")
MarketMonitor_LOG_FILE = os.path.join(LOG_DIR,"marketmonitor.log")
StockHistory_LOG_FILE = os.path.join(LOG_DIR,"StockHistory.log")

# File Names
JSL_DATA_FILE = os.path.join(DATA_DIR,"JSL")
CB_OF_TODAY_FILE = os.path.join(DATA_DIR,"CBofToday")
CB_OF_TODAY_CONV_BIAS = os.path.join(DATA_DIR,"CBofTodayConvBias")
CB_OF_TODAY_NUC_F4 = os.path.join(DATA_DIR,"CBofToday_NUCF4")
#持有封基文件
CYFJ_FILE = os.path.join(DATA_DIR,"CYFJ.xlsx") 

STRATEGY_FILE = os.path.join(DATA_DIR,"StrategyBascket")
TEST_STRATEGY_FILE = os.path.join(DATA_DIR,"Test_StrategyBascket")
LUDE_BASKET_FILE = os.path.join(DATA_DIR,"LudeBascket")
Position_FILE = os.path.join(DATA_DIR,"Position")

STRATEGY_CONV_BIAS = 'CONV_BIAS'           
STRATEGY_CONV_BIAS_V2 = 'CONV_BIAS_V2'       
STRATEGY_LOW_PREM = 'LOW_PREM'
STRATEGY_NUC_F4 = 'NUC_F4'  

#MarketMonitor
StockInventoryFile = os.path.join(DATA_DIR, "持仓净值.xlsx")
CBMonitorFile = os.path.join(DATA_DIR, "cbwatchlist.xlsx")
CONV_PREM_LEVEL = 0.01                                          #溢价率偏离大于3%告警
THS_HOT_CB_FILE = os.path.join(DATA_DIR,"THS_HOT_CB")

import socket
hostname = socket.gethostname()

#增加FIREFOX_PROFILE_PATH定义, 只支持WIN，用hostname区分是否在NUC运行
if hostname == "NUC-QMT1":   #NUC
    FIREFOX_PROFILE_PATH = r"C:\Users\ZhangYi\AppData\Roaming\Mozilla\Firefox\Profiles\r2uq0ae3.default-release"
elif hostname == "NUC-SERVER":
    FIREFOX_PROFILE_PATH = r"C:\Users\Administrator\AppData\Roaming\Mozilla\Firefox\Profiles\emce3zrw.default-release"
else:                       #Ultra 笔记本FIRE
    FIREFOX_PROFILE_PATH = r"C:\Users\zhang\AppData\Roaming\Mozilla\Firefox\Profiles\nyqvch5m.default-release"


#集思录用户名和密码，改为从常量文件引用 2024-04-10
JSL_USER_NAME = 'ilvnet' # 账户名
JSL_PASSWORD = 'bull4ever' # 密码

# 2024-06-14    数据文件集中存放
LUDE_DATA_DIR = os.path.join(DATA_DIR, "Lude")
# 2024-09-21    增加QMT数据目录
QMT_DATA_DIR = os.path.join(DATA_DIR, "QMT")
QMT_JSL_DATA_FILE = os.path.join(QMT_DATA_DIR, "JSL_DATA.xlsx")
QMT_TARGET_CB_FILE = os.path.join(QMT_DATA_DIR, "TargetCB.xlsx")

LUDE_DATA_FILE = os.path.join(LUDE_DATA_DIR, "cb_data.pq")
LUDE_INDEX_FILE = os.path.join(LUDE_DATA_DIR, "index.pq")

# 2024-06-14    增加回测配置文件
CONFIG_DIR = os.path.join(pythondir, "Config")
#策略优化使用的配置文件，包括公用factor定义
BT_CONFIG_FILE = os.path.join(CONFIG_DIR, "BTO_Config.json")
# 2024-08-10    单次策略回测使用的策略配置文件
BT_STRATEGY_FILE = os.path.join(CONFIG_DIR, "BT_Strategy.json")
#2025-01-10    增加资产配置文件
ASSET_CONFIG_FILE = os.path.join(CONFIG_DIR, "资产配置.yaml")

#2024-06-17
TRADING_DAYS_PER_YEAR = 242

#2024-06-27 增加数据库配置为JSON文件

LIDO_CONFIG_FILE = os.path.join(CONFIG_DIR, "lido_config.json")
DB_CONFIG_NAME = 'db_config'
NAS_DB = 'NAS_DB'
NUC_DB = 'NUC_DB'

STRATEGY_CONFIG_NAME = 'strategy_config'

STRATEGY_EVALUATION_FILE_NAME = os.path.join(DATA_DIR, "策略评估")

#企业微信机器人定义 

#QMT 机器人
#QMT_webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4dc9661a-d9bf-4a7d-ba37-de7d16a0da90"
QMT_webhook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2aac393c-a687-47bc-9696-b4c026eae0c5"
#行情监控机器人
MarketMonitor_wehook_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=37917b80-732a-4cbe-8b5b-2fe99040979a"

#MarketMonitor_wehook_url ="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=2aac393c-a687-47bc-9696-b4c026eae0c5"

# 2024-09-13    增加每日复盘代码定义
DAILY_REVIEW_CODE_FILE = os.path.join(CONFIG_DIR, "每日复盘代码定义.yaml")

#2024-09-29    增加Lude策略篮子定义

Lude_Strategies = {
    "禄得策略1": "carl-table-tabs-tab-strategy_config_1",
    "禄得策略2": "carl-table-tabs-tab-strategy_config_2",
    "禄得策略3": "carl-table-tabs-tab-strategy_config_3"
}

Strategy_Config_File = os.path.join(CONFIG_DIR, "Strategy_Config.json")

# 2025-05-12    增加指标工厂配置文件
INDICATOR_FACTORY_CONFIG_FILE = os.path.join(CONFIG_DIR, "Indicator_Factory.yaml")

#2025-03-03 排除强赎时，同时排除满足强赎条件剩余天数
DAYS_TO_REDEEM = 3