"""
Mini QMT 实时行情获取单元测试 (真实环境)

注意：运行此测试前必须确保 Mini QMT 客户端已在后台启动并正常登录。
"""
import pytest
import pandas as pd
import time
from loguru import logger

try:
    from xtquant import xtdata
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False


@pytest.mark.skipif(not HAS_XTQUANT, reason="未安装 xtquant 库")
class TestMiniQMTLive:
    """真实环境下的 Mini QMT 行情测试"""
    
    # 测试用的转债代码 (大秦转债, 中特转债)
    TEST_CODES = ['110095.SH', '123128.SZ']
    
    def test_get_live_quotes_with_full_tick(self):
        """测试使用 get_full_tick 获取实时行情"""
        logger.info(f"--- 测试 get_full_tick (推荐的实时行情获取方式) ---")
        try:
            # 1. 建立推送连接 (非必须，但推荐先订阅)
            for code in self.TEST_CODES:
                xtdata.subscribe_quote(code, period='1d', count=-1)
            
            # 2. 获取实时快照
            # 注意：非交易日可能获取不到最新成交，或者返回上一交易日收盘数据
            data = xtdata.get_full_tick(self.TEST_CODES)
            
            assert isinstance(data, dict), "返回结果应为字典"
            logger.info(f"获取到 {len(data)} 个标的的实时快照")
            
            for code in self.TEST_CODES:
                tick = data.get(code, {})
                if tick:
                    last_price = tick.get('lastPrice', 0)
                    timetag = tick.get('timetag', 'None')
                    logger.info(f"标的: {code} | 最新价: {last_price} | 交易所时间: {timetag}")
                else:
                    logger.warning(f"标的: {code} | 未获取到实时 Tick (可能是非交易日且无缓存)")
                    
        except Exception as e:
            logger.error(f"get_full_tick 执行失败: {e}")
            if "pytest" in sys.modules: pytest.fail(str(e))

    def test_compare_with_market_data_ex(self):
        """对比 get_full_tick (实时) 与 get_market_data_ex (历史缓存)"""
        logger.info(f"--- 对比实时行情与历史缓存数据 ---")
        
        # 1. 下载历史数据
        for code in self.TEST_CODES:
            xtdata.download_history_data(code, period='1d', incrementally=True)
            
        # 2. 获取历史缓存 (K线)
        history_data = xtdata.get_market_data_ex([], self.TEST_CODES, period='1d')
        
        # 3. 获取实时快照 (Tick)
        live_data = xtdata.get_full_tick(self.TEST_CODES)
        
        for code in self.TEST_CODES:
            h_price = 0
            if code in history_data and not history_data[code].empty:
                h_price = history_data[code].iloc[-1]['close']
                h_date = history_data[code].index[-1]
                logger.info(f"[{code}] 历史缓存最新价: {h_price} (日期: {h_date})")
                
            tick = live_data.get(code, {})
            l_price = tick.get('lastPrice', 0)
            l_time = tick.get('timetag', 'None')
            logger.info(f"[{code}] 实时 Tick 最新价: {l_price} (时间: {l_time})")

    def test_get_sector_list(self):
        """测试获取转债板块"""
        logger.info(f"--- 测试获取全市场转债列表 ---")
        for sector_name in ['沪深转债', '可转债']:
            codes = xtdata.get_stock_list_in_sector(sector_name)
            if codes:
                logger.info(f"板块 '{sector_name}' 包含 {len(codes)} 只转债")
                return
        logger.warning("未找到转债板块，请检查券商板块定义")


if __name__ == "__main__":
    if not HAS_XTQUANT:
        print("错误: 请先安装 xtquant")
    else:
        test = TestMiniQMTLive()
        print("\n[开始测试 Mini QMT 行情获取接口]")
        try:
            test.test_get_live_quotes_with_full_tick()
            test.test_compare_with_market_data_ex()
            test.test_get_sector_list()
            print("\n[测试完成]")
        except Exception as e:
            print(f"测试过程出现错误: {e}")
