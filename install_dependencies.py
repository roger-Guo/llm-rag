#!/usr/bin/env python3
"""
RAG项目依赖安装脚本
"""
import subprocess
import sys

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装包"""
    print(f"📦 安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def main():
    """主函数"""
    print("🔧 RAG项目依赖检查和安装")
    print("=" * 50)
    
    # 必需的包列表
    required_packages = [
        "sentence-transformers",
        "chromadb", 
        "openai",
        "scikit-learn",
        "numpy"
    ]
    
    # 检查并安装包
    installed_count = 0
    for package in required_packages:
        if check_package(package.replace("-", "_")):
            print(f"✅ {package} 已安装")
            installed_count += 1
        else:
            print(f"❌ {package} 未安装")
            if install_package(package):
                installed_count += 1
    
    print(f"\n📊 安装结果: {installed_count}/{len(required_packages)} 个包已安装")
    
    if installed_count == len(required_packages):
        print("🎉 所有依赖已安装完成！")
        print("\n现在可以运行调试脚本:")
        print("  python debug_interactive.py")
        print("  python test_qa.py")
        print("  python debug_data_loading.py")
    else:
        print("⚠️ 部分依赖安装失败，请手动安装")

if __name__ == "__main__":
    main() 