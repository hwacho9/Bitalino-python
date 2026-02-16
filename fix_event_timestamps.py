#!/usr/bin/env python3
"""
이벤트 로그 파일의 누락된 타임스탬프를 수정하는 스크립트

문제: 일부 이벤트 라인에 unix_time, readable_time, sample_index가 누락됨
해결: 전후 이벤트의 타임스탬프를 기반으로 누락된 값을 계산하여 채움
"""

import os
import csv
from datetime import datetime
from pathlib import Path


def parse_timestamp_line(line):
    """타임스탬프가 있는 라인을 파싱"""
    parts = line.strip().split(',')
    if len(parts) >= 4:
        try:
            unix_time = float(parts[0])
            readable_time = parts[1]
            sample_index = int(parts[2])
            label = ','.join(parts[3:]).strip()
            return {
                'unix_time': unix_time,
                'readable_time': readable_time,
                'sample_index': sample_index,
                'label': label
            }
        except (ValueError, IndexError):
            return None
    return None


def is_timestamp_missing(line):
    """라인에 타임스탬프가 없는지 확인"""
    line = line.strip()
    if not line:
        return False
    parts = line.split(',')
    if len(parts) == 1:
        return True
    try:
        float(parts[0])
        return False
    except ValueError:
        return True


def fix_event_log(file_path):
    """이벤트 로그 파일의 누락된 타임스탬프를 수정"""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"  Empty file, skipping")
        return
    
    header = lines[0].strip()
    if not header.startswith('unix_time'):
        print(f"  Invalid header, skipping")
        return
    
    parsed_lines = []
    has_missing = False
    
    for i, line in enumerate(lines):
        if i == 0:
            parsed_lines.append({'type': 'header', 'content': line})
            continue
        
        line_clean = line.strip()
        if not line_clean:
            parsed_lines.append({'type': 'empty', 'content': line})
            continue
        
        parsed = parse_timestamp_line(line)
        if parsed:
            parsed_lines.append({'type': 'complete', 'data': parsed, 'content': line})
        else:
            has_missing = True
            label = line_clean.replace('\r', '').replace('\n', '')
            parsed_lines.append({'type': 'missing', 'label': label, 'content': line})
    
    if not has_missing:
        print(f"  No missing timestamps found")
        return
    
    sampling_rate = 1000
    baseline_duration = 60
    
    for i, item in enumerate(parsed_lines):
        if item['type'] != 'missing':
            continue
        
        label = item['label']
        
        next_complete = None
        for j in range(i + 1, len(parsed_lines)):
            if parsed_lines[j]['type'] == 'complete':
                next_complete = parsed_lines[j]
                break
        
        prev_complete = None
        for j in range(i - 1, -1, -1):
            if parsed_lines[j]['type'] == 'complete':
                prev_complete = parsed_lines[j]
                break
        
        if label == 'PHASE0_BASELINE_START':
            if next_complete:
                next_data = next_complete['data']
                if next_data['label'] == 'PHASE0_BASELINE_END':
                    unix_time = next_data['unix_time'] - baseline_duration
                    sample_index = next_data['sample_index'] - (baseline_duration * sampling_rate)
                    if sample_index < 0:
                        sample_index = 0
                    
                    dt = datetime.fromtimestamp(unix_time)
                    ms = int(dt.microsecond / 1000)
                    readable_time = dt.strftime('%Y-%m-%d %H:%M:%S') + f'.{ms:03d}'
                    
                    item['data'] = {
                        'unix_time': unix_time,
                        'readable_time': readable_time,
                        'sample_index': int(sample_index),
                        'label': label
                    }
                    item['type'] = 'fixed'
                    print(f"  Fixed: {label} at {readable_time}")
        
        elif label in ['START', 'PHASE1_TASK_END', 'CHASE_START']:
            if prev_complete:
                prev_data = prev_complete['data']
                item['data'] = {
                    'unix_time': prev_data['unix_time'],
                    'readable_time': prev_data['readable_time'],
                    'sample_index': prev_data['sample_index'],
                    'label': label
                }
                item['type'] = 'fixed'
                print(f"  Fixed: {label} (same as prev: {prev_data['label']})")
        
        else:
            if prev_complete:
                prev_data = prev_complete['data']
                item['data'] = {
                    'unix_time': prev_data['unix_time'],
                    'readable_time': prev_data['readable_time'],
                    'sample_index': prev_data['sample_index'],
                    'label': label
                }
                item['type'] = 'fixed'
                print(f"  Fixed: {label}")
    
    backup_path = str(file_path) + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  Backup created: {backup_path}")
    
    with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
        for item in parsed_lines:
            if item['type'] == 'header':
                f.write('unix_time, readable_time, sample_index, label\n')
            elif item['type'] == 'empty':
                f.write('\n')
            elif item['type'] in ['complete', 'fixed']:
                data = item['data']
                f.write(f"{data['unix_time']},{data['readable_time']},{data['sample_index']},{data['label']}\n")
            else:
                f.write(item['content'])
    
    print(f"  File updated successfully")


def main():
    base_dir = Path(__file__).parent
    event_files = list(base_dir.glob('**/events_log_*.csv'))
    
    print(f"Found {len(event_files)} event log files\n")
    
    for file_path in event_files:
        fix_event_log(file_path)
        print()
    
    print("Done!")


if __name__ == '__main__':
    main()
