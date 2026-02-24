#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接读取Excel文件（不依赖openpyxl/xlrd）
通过解析xlsx的XML结构来读取数据
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
import re

def read_xlsx_direct(filepath):
    """直接读取xlsx文件"""
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            # 读取共享字符串
            shared_strings = []
            try:
                with zf.open('xl/sharedStrings.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    # 命名空间
                    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    for si in root.findall('.//main:si', ns):
                        text_elem = si.find('.//main:t', ns)
                        if text_elem is not None:
                            shared_strings.append(text_elem.text or '')
                        else:
                            shared_strings.append('')
            except:
                pass
            
            # 读取工作表，优先选择sheet2（通常包含主要数据），否则选择数据最多的sheet
            sheet_files = [f for f in zf.namelist() if f.startswith('xl/worksheets/sheet') and f.endswith('.xml') and '/_rels/' not in f]
            if not sheet_files:
                return None
            
            # 优先选择sheet2.xml（通常包含主要测试数据）
            sheet_file = None
            for sf in sorted(sheet_files):
                if 'sheet2.xml' in sf:
                    sheet_file = sf
                    break
            
            # 如果没找到sheet2，选择数据最多的sheet
            if sheet_file is None:
                max_rows = 0
                for sf in sorted(sheet_files):
                    try:
                        with zf.open(sf) as test_f:
                            test_tree = ET.parse(test_f)
                            test_root = test_tree.getroot()
                            test_ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                            test_rows = test_root.findall('.//main:row', test_ns)
                            if len(test_rows) > max_rows:
                                max_rows = len(test_rows)
                                sheet_file = sf
                    except:
                        continue
            
            if sheet_file is None:
                sheet_file = sorted(sheet_files)[0]
            
            with zf.open(sheet_file) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                
                # 读取所有行数据（使用字典存储，按行号排序）
                rows_dict = {}
                max_col = 0
            
                for row in root.findall('.//main:row', ns):
                    row_num = int(row.get('r', '0')) - 1  # Excel行号从1开始，转为0开始
                    if row_num < 0:
                        continue
                    
                    # 初始化行数据字典
                    if row_num not in rows_dict:
                        rows_dict[row_num] = {}
                    
                    cells = row.findall('.//main:c', ns)
                    for cell in cells:
                        cell_ref = cell.get('r', '')
                        # 解析列号（A=0, B=1, ..., Z=25, AA=26, ...）
                        col_num = 0
                        col_str = ''
                        for char in cell_ref:
                            if char.isalpha():
                                col_str += char
                            else:
                                break
                        # 转换列字符串为数字
                        for i, char in enumerate(reversed(col_str)):
                            col_num += (ord(char.upper()) - ord('A') + 1) * (26 ** i)
                        col_num -= 1  # 转为0开始
                        
                        if col_num > max_col:
                            max_col = col_num
                        
                        cell_type = cell.get('t', '')
                        value_elem = cell.find('.//main:v', ns)
                        
                        if value_elem is not None:
                            value = value_elem.text
                            if cell_type == 's' and value:  # 共享字符串
                                try:
                                    idx = int(value)
                                    if idx < len(shared_strings):
                                        value = shared_strings[idx]
                                except:
                                    pass
                            rows_dict[row_num][col_num] = value
                        else:
                            rows_dict[row_num][col_num] = ''
                
                # 转换为列表格式（只保留有数据的行）
                rows_data = []
                if rows_dict:
                    max_row = max(rows_dict.keys())
                    for row_num in range(max_row + 1):
                        if row_num in rows_dict:
                            row_data = []
                            for col_num in range(max_col + 1):
                                row_data.append(rows_dict[row_num].get(col_num, ''))
                            rows_data.append(row_data)
                
                if not rows_data:
                    return None
                
                # 转换为DataFrame
                # 第一行作为列名
                if len(rows_data) > 0:
                    max_cols = max(len(r) for r in rows_data)
                    # 补齐列数
                    for r in rows_data:
                        while len(r) < max_cols:
                            r.append('')
                    
                    # 尝试识别列名
                    header_row = None
                    for i, row in enumerate(rows_data[:10]):  # 检查前10行
                        if any(is_numeric(str(v)) for v in row[:5] if v):
                            header_row = i - 1 if i > 0 else 0
                            break
                    
                    if header_row is None:
                        header_row = 0
                    
                    # 提取列名和数据
                    if header_row < len(rows_data):
                        columns = [str(v) if v else f'Col_{i}' for i, v in enumerate(rows_data[header_row])]
                        data_rows = rows_data[header_row + 1:]
                        
                        df = pd.DataFrame(data_rows, columns=columns)
                        
                        # 尝试转换为数值类型
                        for col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                        
                        return df
                
                return None
    except Exception as e:
        print(f"  读取错误: {e}")
        return None

def is_numeric(s):
    """检查字符串是否为数字"""
    try:
        float(str(s))
        return True
    except:
        return False

def read_xls_simple(filepath):
    """简单读取xls文件（尝试作为CSV或文本）"""
    # xls文件格式复杂，这里先返回None，需要xlrd
    return None

if __name__ == "__main__":
    # 测试
    test_file = Path("raw_data/DST-US06-FUDS-0/A1-007-DST-US06-FUDS-0-20120813.xlsx")
    if test_file.exists():
        df = read_xlsx_direct(test_file)
        if df is not None:
            print(f"成功读取: {len(df)} 行, {len(df.columns)} 列")
            print(f"列名: {df.columns.tolist()[:10]}")
            print(f"前5行:\n{df.head()}")
        else:
            print("读取失败")
    else:
        print(f"文件不存在: {test_file}")
