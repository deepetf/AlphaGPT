"""
测试 LocalDatabaseProvider 改造

运行方式:
1. 使用 API 数据源（默认）:
   python test_local_db.py

2. 使用本地数据库:
   set DATA_SOURCE=local_db
   python test_local_db.py
"""
import os
import sys
import asyncio

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_config():
    """测试配置"""
    from data_pipeline.config import Config
    
    print("=" * 50)
    print("配置测试")
    print("=" * 50)
    print(f"DATA_SOURCE: {Config.DATA_SOURCE}")
    print(f"CB_DB_DSN: {Config.CB_DB_DSN}")
    print()
    return True

async def test_provider_import():
    """测试 Provider 导入"""
    print("=" * 50)
    print("Provider 导入测试")
    print("=" * 50)
    
    try:
        from data_pipeline.providers.local_db import LocalDatabaseProvider
        print("✓ LocalDatabaseProvider 导入成功")
        
        from data_pipeline.providers.birdeye import BirdeyeProvider
        print("✓ BirdeyeProvider 导入成功")
        
        from data_pipeline.data_manager import DataManager
        print("✓ DataManager 导入成功")
        print()
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

async def test_local_db_connection():
    """测试本地数据库连接"""
    from data_pipeline.config import Config
    
    if Config.DATA_SOURCE != "local_db":
        print("跳过本地数据库连接测试 (DATA_SOURCE != 'local_db')")
        print()
        return True
    
    print("=" * 50)
    print("本地数据库连接测试")
    print("=" * 50)
    
    try:
        from data_pipeline.providers.local_db import LocalDatabaseProvider
        
        provider = LocalDatabaseProvider()
        tokens = await provider.get_trending_tokens(limit=5)
        
        print(f"✓ 成功获取 {len(tokens)} 个可转债")
        if tokens:
            print(f"  第一个: {tokens[0]}")
        
        provider.close()
        print()
        return True
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    results = []
    
    results.append(await test_config())
    results.append(await test_provider_import())
    results.append(await test_local_db_connection())
    
    print("=" * 50)
    print("测试结果")
    print("=" * 50)
    if all(results):
        print("✓ 所有测试通过")
    else:
        print("✗ 部分测试失败")

if __name__ == "__main__":
    asyncio.run(main())
