'''
#   2024-03-14  用class实现获取数据库数据
#   2025-04-12  采用标准的lidoconfig
#               增加获取持仓标的类
#   2025-10-01  新增10年国债收益率数据库访问类BOND10Y
#   2025-10-09  修正IV 计算当日时，BOND10Y get by date使用SQLAlchemy 参数方法
'''


from datetime import date
from decimal import Decimal  # 2025-10-01  BOND10Y需要处理Decimal类型收益率
from chinese_calendar import is_workday
import numpy as np
import pandas as pd
from urllib.parse import quote
from sqlalchemy import create_engine, text
from LidoConfig import LidoConfig
from CommonFunctions import connect_to_db
from constants import TRADING_LOG_NAME
import logging
import pymysql
# 2025-10-01  引入Union与Iterable用于BOND10Y参数类型注解
from typing import Iterable, Optional, Tuple, Union
from LidoConfig import LidoConfig
from trading_log import qmt_logger
import time

# 设置日志
logger = qmt_logger

# 设置日志
logger = logging.getLogger(TRADING_LOG_NAME)

#转债数据库类
class LidoCBData():

    def __init__(self):
        
        self.lido_config = LidoConfig()
        self.engine, self.conn = self._get_db_connection()

        self.conn = self.engine.connect()

    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            db_config = self.lido_config.get_db_config()
            return connect_to_db(db_config)
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def GetCBData(self,start_date):                                           #从数据库获取start_date之后的转债数据

        query = f"SELECT * FROM CB_DATA WHERE trade_date >=  '{start_date}'"

        return pd.read_sql(query, self.conn)
    
    def GetLastTradeDayCBCode(self):                                          #获取前一交易日的可转债数据的code和stk_code
        query = """
        SELECT code, code_stk 
        FROM CB_DATA 
        WHERE trade_date = (
            SELECT MAX(trade_date) 
            FROM CB_DATA 
            WHERE trade_date < CURDATE()
        )
        """
        return pd.read_sql(query, self.conn)
    
    def CloseDB(self):

        self.engine.dispose()

#持仓数据库类
class LidoPositionData():

    def __init__(self):
        
        self.lido_config = LidoConfig()
        self.engine, self.conn = self._get_db_connection()

        self.conn = self.engine.connect()
        self.position = pd.DataFrame()

    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            db_config = self.lido_config.get_db_config()
            return connect_to_db(db_config)
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def GetLatestPositionData(self,start_date):                                           #从数据库获取start_date之后的转债数据

        query = f"SELECT * FROM 实盘持仓 WHERE 日期 = (SELECT MAX(日期) FROM 实盘持仓)"

        self.position = pd.read_sql(query, self.conn)
        
        return self.position
    
    #获取指定账户的持仓
    def GetLatestPositionDataByAccount(self,account):

        query = f"SELECT * FROM 实盘持仓 WHERE account = '{account}' AND 日期 = (SELECT MAX(日期) FROM 实盘持仓)"

        return pd.read_sql(query, self.conn)
    
    #获取所有可转债持仓
    def GetLatestCB_PositionData(self):

        query = f"SELECT * FROM 实盘持仓 WHERE (代码 LIKE '11%' OR 代码 LIKE '12%') AND 日期 = (SELECT MAX(日期) FROM 实盘持仓)"

        return pd.read_sql(query, self.conn)
    
    #获取所有非可转债持仓，不包括逆回购
    def GetLatestNonCB_PositionData(self):

        query = f"SELECT * FROM 实盘持仓 WHERE (代码 NOT LIKE '11%' AND 代码 NOT LIKE '12%' AND 代码 NOT LIKE '204%' AND 代码 NOT LIKE '999%') AND 日期 = (SELECT MAX(日期) FROM 实盘持仓)"

        return pd.read_sql(query, self.conn)

    def CloseDB(self):

        self.engine.dispose()
        
#A股股票数据库类
class LidoStockData():

    def __init__(self):
        

        self.lido_config = LidoConfig()
        self.engine, self.conn = self._get_db_connection()

        self.conn = self.engine.connect()

    def _get_db_connection(self):
        """获取数据库连接"""
        try:
            db_config = self.lido_config.get_db_config()
            return connect_to_db(db_config)
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def GetStockDataByDate(self,start_date):                                           #从数据库获取start_date之后的股票数据，date 格式2024-04-03

        query = f"SELECT * FROM A_STOCK_HISTORY_BAO WHERE date >=  '{start_date}'"

        return pd.read_sql(query, self.conn)
    
    def GetStockDataByCode(self,code):                                           #从数据库获取code股票数据 code 格式：sh.688196

        query = f"SELECT * FROM A_STOCK_HISTORY_BAO WHERE code =  '{code}'"

        return pd.read_sql(query, self.conn)
    
    def CloseDB(self):

        self.engine.dispose()

"""
集思录可转债数据访问类
功能：获取集思录可转债数据
"""


class JSL_DB_Data:
    """集思录可转债数据访问类"""
    
    def __init__(self):
        self.lido_config = LidoConfig()
        self.conn = self._connect_to_db()
    
    def _connect_to_db(self) -> pymysql.Connection:
        """连接到数据库"""
        try:
            db_config = self.lido_config.get_db_config()
            conn = pymysql.connect(
                host=db_config['host'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                port=int(db_config['port']),
                charset='utf8mb4'
            )
            return conn
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def get_latest_bond_data(self) -> pd.DataFrame:
        """
        获取最新的集思录可转债数据
        
        Returns:
        包含最新可转债数据的DataFrame
        """
        try:
            logger.info("开始获取最新的集思录可转债数据...")
            
            # 首先获取最新的交易日期
            date_query = """
            SELECT MAX(trade_date) as latest_date
            FROM 集思录可转债数据
            """
            
            with self.conn.cursor() as cursor:
                cursor.execute(date_query)
                result = cursor.fetchone()
                if not result or not result[0]:
                    logger.error("未找到最新的交易日期")
                    return pd.DataFrame()
                
                latest_date = result[0]
                logger.info(f"最新交易日期: {latest_date}")
            
            # 获取该日期的所有可转债数据
            data_query = """
            SELECT *
            FROM 集思录可转债数据
            WHERE trade_date = %s
            """
            
            df = pd.read_sql(data_query, self.conn, params=[latest_date])
            logger.info(f"成功读取{len(df)}条可转债记录，日期：{latest_date}")
            
            if df.empty:
                logger.warning("未找到任何可转债数据")
                return pd.DataFrame()
            
            # 检查必要的列是否存在
            required_columns = ['转债名称', '剩余年限', '到期税前收益', '转股溢价率', '现价']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"数据中缺少必要的列: {missing_columns}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"读取可转债数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def update_guarantee_info(self, guarantee_data_dict=None):
        """
        更新数据库中最新日期的集思录可转债数据表中的"担保说明"字段
        
        Parameters:
        guarantee_data_dict: Dict[str, str], 可选
            包含可转债代码到担保信息的映射字典。如果为None，则会自动爬取数据
            
        Returns:
        int: 成功更新的记录数量
        """
        try:
            if not self.conn:
                logger.error("数据库连接不存在")
                return 0
                
            # 获取最新交易日期
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT MAX(trade_date) FROM 集思录可转债数据")
                result = cursor.fetchone()
                if not result or not result[0]:
                    logger.error("未找到最新的交易日期")
                    return 0
                    
                latest_date = result[0]
                logger.info(f"最新交易日期: {latest_date}")
            
            # 如果没有提供担保数据，则需要爬取
            if not guarantee_data_dict:
                # 获取最新日期的可转债数据
                df_bonds = self.get_latest_bond_data()
                
                if df_bonds.empty:
                    logger.error("没有找到可转债数据")
                    return 0
                
                # 创建一个字典存储代码->担保信息的映射
                guarantee_data_dict = {}
                
                # 爬取担保信息
                for _, row in df_bonds.iterrows():
                    bond_code = row['代码']
                    # 移除可能的市场前缀
                    clean_code = bond_code.split('.')[-1] if '.' in bond_code else bond_code
                    
                    # 获取担保信息（这需要实现一个爬取函数）
                    guarantee_info = self._fetch_guarantee_info(clean_code)
                    guarantee_data_dict[bond_code] = guarantee_info
                    
                    # 避免过快请求
                    time.sleep(0.5)
            
            # 检查表中是否存在担保说明字段
            with self.conn.cursor() as cursor:
                cursor.execute("SHOW COLUMNS FROM 集思录可转债数据 LIKE '担保说明'")
                field_exists = cursor.fetchone()
                
                # 如果不存在则添加该字段
                if not field_exists:
                    logger.info("表中不存在'担保说明'字段，正在添加...")
                    cursor.execute("ALTER TABLE 集思录可转债数据 ADD COLUMN 担保说明 VARCHAR(100) DEFAULT NULL")
                    self.conn.commit()
            
            # 更新数据
            update_count = 0
            with self.conn.cursor() as cursor:
                for bond_code, guarantee_info in guarantee_data_dict.items():
                    # 确保担保信息不为None
                    if guarantee_info is None:
                        guarantee_info = "未找到担保信息"
                    
                    # 限制担保信息长度为100个字符
                    if len(guarantee_info) > 100:
                        guarantee_info = guarantee_info[:100]
                        logger.info(f"担保信息长度超过100个字符，已截断: {bond_code}")
                    
                    # 使用参数化查询防止SQL注入
                    update_sql = """
                    UPDATE 集思录可转债数据
                    SET 担保说明 = %s
                    WHERE 代码 = %s AND trade_date = %s
                    """
                    cursor.execute(update_sql, (guarantee_info, bond_code, latest_date))
                    update_count += cursor.rowcount
            
                self.conn.commit()
                logger.info(f"成功更新了{update_count}条可转债担保信息")
            
            return update_count
            
        except Exception as e:
            logger.error(f"更新可转债担保信息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                self.conn.rollback()  # 发生错误时回滚
            except:
                pass
            return 0
    
    def _fetch_guarantee_info(self, bond_code):
        """
        从集思录网站爬取指定可转债的担保信息
        
        Parameters:
        bond_code: str
            可转债代码，不包含市场前缀
            
        Returns:
        str: 担保信息
        """
        import requests
        from bs4 import BeautifulSoup
        import re
        import time
        
        url = f"https://www.jisilu.cn/data/convert_bond_detail/{bond_code}"
        
        # 设置请求头，模拟浏览器访问
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.jisilu.cn/"
        }
        
        try:
            # 发送HTTP请求
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找包含"担保"字段的表格行
            guarantee_element = None
            
            # 方法1：查找包含"担保"文本的表格单元格
            guarantee_cells = soup.find_all(text=re.compile(r'担保'))
            
            # 如果找到了包含"担保"的单元格，获取相邻的单元格内容
            if guarantee_cells:
                for cell in guarantee_cells:
                    parent = cell.parent
                    # 获取下一个兄弟元素（可能是包含担保信息的单元格）
                    next_cell = parent.find_next_sibling()
                    if next_cell:
                        guarantee_element = next_cell.text.strip()
                        break
            
            # 如果上面的方法没找到，尝试其他查找方式
            if not guarantee_element:
                # 方法2：查找data-title属性为"担保"的元素
                guarantee_data = soup.find(attrs={"data-title": "担保"})
                if guarantee_data:
                    guarantee_element = guarantee_data.text.strip()
            
            # 返回担保信息
            return guarantee_element if guarantee_element else "未找到担保信息"
            
        except Exception as e:
            logger.error(f"获取担保信息时出错: {str(e)}")
            return "获取失败"
    
    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")


class THS_DB_Data:
    """集思录可转债数据访问类"""
    
    def __init__(self):
        self.lido_config = LidoConfig()
        self.conn = self._connect_to_db()
    
    def _connect_to_db(self) -> pymysql.Connection:
        """连接到数据库"""
        try:
            db_config = self.lido_config.get_db_config()
            conn = pymysql.connect(
                host=db_config['host'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database'],
                port=int(db_config['port']),
                charset='utf8mb4'
            )
            return conn
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def get_latest_hot_bonds(self) -> pd.DataFrame:
        """
        获取最新的集思录可转债数据
        
        Returns:
        包含最新可转债数据的DataFrame
        """
        try:
            logger.info("开始获取最新的集思录可转债数据...")
            
            # 首先获取最新的交易日期
            date_query = """
            SELECT MAX(trade_date) as latest_date
            FROM THS_HOT_CB
            """
            
            with self.conn.cursor() as cursor:
                cursor.execute(date_query)
                result = cursor.fetchone()
                if not result or not result[0]:
                    logger.error("未找到最新的交易日期")
                    return pd.DataFrame()
                
                latest_date = result[0]
                logger.info(f"最新交易日期: {latest_date}")
            
            # 获取该日期的所有可转债数据
            data_query = """
            SELECT *
            FROM THS_HOT_CB
            WHERE trade_date = %s
            """
            
            df = pd.read_sql(data_query, self.conn, params=[latest_date])
            logger.info(f"成功读取{len(df)}条可转债记录，日期：{latest_date}")
            
            if df.empty:
                logger.warning("未找到任何可转债数据")
                return pd.DataFrame()
            
            # 只保留seq_no数值最大的记录
            if 'seq_no' in df.columns:

                # 获取seq_no的最大值
                max_seq_value = df['seq_no'].max()
                
                # 保留所有seq_no等于最大值的行
                df = df[df['seq_no'] == max_seq_value].reset_index(drop=True)
                
                logger.info(f"已筛选出seq_no最大的记录，seq_no值为: {df['seq_no'].iloc[0]}")
            else:
                logger.warning("数据中不存在seq_no列，无法筛选最大值记录")

            # 检查必要的列是否存在
            required_columns = ['trade_date', 'time', 'name', 'order_no', 'seq_no']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"数据中缺少必要的列: {missing_columns}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"读取可转债数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")


# 2025-10-01  新增10年国债收益率访问类，实现插入和查询功能
class BOND10Y:
    """10年国债收益率数据访问类"""

    def __init__(self) -> None:
        self.lido_config = LidoConfig()
        self.engine, self.conn = self._get_db_connection()
        if self.engine is None:
            raise RuntimeError("BOND10Y 数据库连接失败")
        self.conn = self.engine.connect()

    # 2025-10-01  使用与其他类一致的MySQL配置建立连接
    def _get_db_connection(self):
        """建立数据库连接"""
        try:
            db_config = self.lido_config.get_db_config()
            return connect_to_db(db_config)
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    # 2025-10-01  统一处理日期格式，支持date或字符串
    @staticmethod
    def _normalize_trade_date(trade_date: Union[str, date]) -> str:
        """将日期参数统一格式化为字符串"""
        if isinstance(trade_date, date):
            return trade_date.strftime("%Y-%m-%d")
        return str(trade_date)

    # 2025-10-01  单条插入或更新十年期收益率
    def upsert_record(self, trade_date: Union[str, date], yield_value: Union[float, Decimal, str]) -> None:
        """插入或更新单条收益率记录"""
        trade_date_str = self._normalize_trade_date(trade_date)
        sql = text(
            "INSERT INTO BOND10Y (trade_date, yield) VALUES (:trade_date, :yield_value) "
            "ON DUPLICATE KEY UPDATE yield = :yield_value"
        )
        with self.engine.begin() as conn:
            conn.execute(sql, {"trade_date": trade_date_str, "yield_value": yield_value})

    # 2025-10-01  批量插入或更新十年期收益率
    def upsert_records(self, records: Iterable[Tuple[Union[str, date], Union[float, Decimal, str]]]) -> None:
        """批量插入或更新多条收益率记录"""
        records_list = list(records)
        if not records_list:
            return
        sql = text(
            "INSERT INTO BOND10Y (trade_date, yield) VALUES (:trade_date, :yield_value) "
            "ON DUPLICATE KEY UPDATE yield = :yield_value"
        )
        payload = [
            {"trade_date": self._normalize_trade_date(trade_date), "yield_value": yield_value}
            for trade_date, yield_value in records_list
        ]
        with self.engine.begin() as conn:
            conn.execute(sql, payload)

    # 2025-10-01  查询指定日期收益率
    #   2025-10-09  修正IV 计算当日时，BOND10Y get by date使用SQLAlchemy 参数方法
    def get_by_date(self, trade_date: Union[str, date]) -> pd.DataFrame:
        """查询指定日期的收益率数据"""
        trade_date_str = self._normalize_trade_date(trade_date)

        

        query = text("SELECT trade_date, yield "
               "FROM BOND10Y "
               "WHERE trade_date = :trade_date")

        #query=f"SELECT trade_date, yield FROM BOND10Y WHERE trade_date =  '{trade_date_str}'"
        #query = "SELECT trade_date, yield FROM BOND10Y WHERE trade_date = %s"
        return pd.read_sql(query, self.conn, params={"trade_date": trade_date_str})

    # 2025-10-01  查询全部收益率记录
    def get_all(self) -> pd.DataFrame:
        """查询全部收益率记录"""
        query = "SELECT trade_date, yield FROM BOND10Y ORDER BY trade_date"
        return pd.read_sql(query, self.conn)

    # 2025-10-01  批量查询指定日期区间
    def get_by_dates(self, trade_dates: Iterable[Union[str, date]]) -> pd.DataFrame:
        """按给定日期集合查询收益率数据"""
        normalized = [self._normalize_trade_date(item) for item in trade_dates if item is not None]
        unique_dates = sorted(set(normalized))
        if not unique_dates:
            return pd.DataFrame(columns=["trade_date", "yield"])
        placeholders = ",".join(["%s"] * len(unique_dates))
        query = f"SELECT trade_date, yield FROM BOND10Y WHERE trade_date IN ({placeholders})"
        df = pd.read_sql(query, self.conn, params=unique_dates)
        if df.empty:
            # 2025-10-01 当全部缺失时直接返回空表
            return pd.DataFrame(columns=["trade_date", "yield"])
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime("%Y-%m-%d")
        df = df.sort_values('trade_date')
        lookup = df.set_index('trade_date')['yield']
        filled = []
        last_value: Optional[float] = None
        forward_fill_count = 0
        for trade_date in unique_dates:
            if trade_date in lookup:
                last_value = float(lookup[trade_date])
                filled.append((trade_date, last_value))
            else:
                if last_value is not None:
                    forward_fill_count += 1
                    filled.append((trade_date, last_value))
        filled_df = pd.DataFrame(filled, columns=['trade_date', 'yield'])
        if forward_fill_count > 0:
            logger.warning(
                "BOND10Y: %s 个日期缺失收益率，已使用最近有效值前向填充。示例日期: %s",
                forward_fill_count,
                filled_df['trade_date'].iloc[-1] if not filled_df.empty else 'N/A'
            )
        return filled_df

    # 2025-10-01  析构函数中关闭数据库连接
    def __del__(self) -> None:
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
        except Exception as e:
            logger.error(f"关闭BOND10Y数据库连接失败: {e}")
