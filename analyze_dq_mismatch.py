#!/usr/bin/env python3
"""
Analyze CPU vs GPU dQ output mismatches from hex dump files.
Summarizes mismatch count, root cause patterns, and location distribution.
"""

import re
import struct
import argparse
from collections import defaultdict

def bf16_to_float(hex_val):
    """Convert bf16 hex value to float."""
    # bf16 is the upper 16 bits of fp32
    val = int(hex_val, 16)
    # Pad with zeros for lower 16 bits
    fp32_bits = val << 16
    return struct.unpack('f', struct.pack('I', fp32_bits))[0]

def parse_hex_file(filepath):
    """Parse the hex dump file and return structured data."""
    data = {}
    current_batch = None
    current_head = None
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Parse batch/head header
            batch_head_match = re.match(r'\+\+\+\+Batch\[(\d+)\]---head\[(\d+)\]\+\+\+\+:', line)
            if batch_head_match:
                current_batch = int(batch_head_match.group(1))
                current_head = int(batch_head_match.group(2))
                if current_batch not in data:
                    data[current_batch] = {}
                if current_head not in data[current_batch]:
                    data[current_batch][current_head] = {}
                continue
            
            # Parse row data: R[XXXX]: 0xYYYY 0xYYYY ...
            row_match = re.match(r'R\[(\d+)\]:\s*(.+)', line)
            if row_match:
                row_idx = int(row_match.group(1))
                hex_values = row_match.group(2).split()
                data[current_batch][current_head][row_idx] = hex_values
    
    return data

def compare_files(cpu_file, gpu_file):
    """Compare two hex dump files and analyze mismatches."""
    print(f"Loading {cpu_file}...")
    cpu_data = parse_hex_file(cpu_file)
    print(f"Loading {gpu_file}...")
    gpu_data = parse_hex_file(gpu_file)
    
    mismatches = []
    total_elements = 0
    
    # Track mismatch patterns
    mismatch_by_batch = defaultdict(int)
    mismatch_by_head = defaultdict(int)
    mismatch_by_row = defaultdict(int)
    mismatch_by_col = defaultdict(int)
    error_magnitudes = []
    
    # Iterate through all data
    for batch in sorted(cpu_data.keys()):
        if batch not in gpu_data:
            print(f"Warning: Batch {batch} missing in GPU data")
            continue
            
        for head in sorted(cpu_data[batch].keys()):
            if head not in gpu_data[batch]:
                print(f"Warning: Head {head} missing in GPU data for batch {batch}")
                continue
                
            for row in sorted(cpu_data[batch][head].keys()):
                if row not in gpu_data[batch][head]:
                    print(f"Warning: Row {row} missing in GPU data for batch {batch}, head {head}")
                    continue
                
                cpu_row = cpu_data[batch][head][row]
                gpu_row = gpu_data[batch][head][row]
                
                for col, (cpu_val, gpu_val) in enumerate(zip(cpu_row, gpu_row)):
                    total_elements += 1
                    if cpu_val != gpu_val:
                        try:
                            cpu_float = bf16_to_float(cpu_val)
                            gpu_float = bf16_to_float(gpu_val)
                            abs_err = abs(cpu_float - gpu_float)
                            rel_err = abs_err / abs(cpu_float) if cpu_float != 0 else float('inf')
                        except:
                            cpu_float = gpu_float = abs_err = rel_err = None
                        
                        mismatch = {
                            'batch': batch,
                            'head': head,
                            'row': row,
                            'col': col,
                            'cpu_hex': cpu_val,
                            'gpu_hex': gpu_val,
                            'cpu_float': cpu_float,
                            'gpu_float': gpu_float,
                            'abs_err': abs_err,
                            'rel_err': rel_err
                        }
                        mismatches.append(mismatch)
                        
                        mismatch_by_batch[batch] += 1
                        mismatch_by_head[head] += 1
                        mismatch_by_row[row] += 1
                        mismatch_by_col[col] += 1
                        if abs_err is not None:
                            error_magnitudes.append(abs_err)
    
    return {
        'total_elements': total_elements,
        'mismatches': mismatches,
        'mismatch_by_batch': dict(mismatch_by_batch),
        'mismatch_by_head': dict(mismatch_by_head),
        'mismatch_by_row': dict(mismatch_by_row),
        'mismatch_by_col': dict(mismatch_by_col),
        'error_magnitudes': error_magnitudes
    }

def print_summary(results):
    """Print analysis summary."""
    mismatches = results['mismatches']
    total = results['total_elements']
    
    print("\n" + "=" * 80)
    print("MISMATCH ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n1. OVERVIEW:")
    print(f"   Total elements compared: {total}")
    print(f"   Total mismatches: {len(mismatches)}")
    print(f"   Mismatch rate: {len(mismatches)/total*100:.4f}%" if total > 0 else "   N/A")
    
    if not mismatches:
        print("\n   No mismatches found - CPU and GPU outputs are identical!")
        return
    
    # Error statistics
    error_mags = results['error_magnitudes']
    if error_mags:
        print(f"\n2. ERROR MAGNITUDE STATISTICS:")
        print(f"   Min absolute error: {min(error_mags):.6e}")
        print(f"   Max absolute error: {max(error_mags):.6e}")
        print(f"   Avg absolute error: {sum(error_mags)/len(error_mags):.6e}")
    
    # Location distribution
    print(f"\n3. MISMATCH LOCATION DISTRIBUTION:")
    
    print(f"\n   By Batch:")
    for batch, count in sorted(results['mismatch_by_batch'].items()):
        print(f"     Batch[{batch:04d}]: {count} mismatches")
    
    print(f"\n   By Head:")
    for head, count in sorted(results['mismatch_by_head'].items()):
        print(f"     Head[{head:04d}]: {count} mismatches")
    
    print(f"\n   By Row (seq position, showing top 20):")
    sorted_rows = sorted(results['mismatch_by_row'].items(), key=lambda x: -x[1])[:20]
    for row, count in sorted_rows:
        print(f"     R[{row:04d}]: {count} mismatches")
    
    print(f"\n   By Column (head dim, showing top 20):")
    sorted_cols = sorted(results['mismatch_by_col'].items(), key=lambda x: -x[1])[:20]
    for col, count in sorted_cols:
        print(f"     Col[{col:03d}]: {count} mismatches")
    
    # Pattern analysis
    print(f"\n4. ROOT CAUSE PATTERN ANALYSIS:")
    
    # Check if mismatches are concentrated in specific hdim ranges
    col_ranges = {'0-63': 0, '64-127': 0, '128-191': 0}
    for m in mismatches:
        col = m['col']
        if col < 64:
            col_ranges['0-63'] += 1
        elif col < 128:
            col_ranges['64-127'] += 1
        else:
            col_ranges['128-191'] += 1
    
    print(f"   Mismatches by head dimension range:")
    for range_name, count in col_ranges.items():
        pct = count / len(mismatches) * 100 if mismatches else 0
        print(f"     hdim {range_name}: {count} ({pct:.1f}%)")
    
    # Check for row patterns (e.g., tile boundaries)
    tile_sizes = [16, 32, 48, 64]
    print(f"\n   Row patterns (tile boundary analysis):")
    for tile_size in tile_sizes:
        at_boundary = sum(1 for m in mismatches if m['row'] % tile_size == 0)
        pct = at_boundary / len(mismatches) * 100 if mismatches else 0
        print(f"     At {tile_size}-row boundaries: {at_boundary} ({pct:.1f}%)")
    
    # First few mismatches detail
    print(f"\n5. FIRST 10 MISMATCHES (DETAILED):")
    for i, m in enumerate(mismatches[:10]):
        print(f"\n   [{i+1}] Batch={m['batch']}, Head={m['head']}, Row={m['row']}, Col={m['col']}")
        print(f"       CPU: {m['cpu_hex']} ({m['cpu_float']:.6f})")
        print(f"       GPU: {m['gpu_hex']} ({m['gpu_float']:.6f})")
        if m['abs_err'] is not None:
            print(f"       Abs Error: {m['abs_err']:.6e}, Rel Error: {m['rel_err']:.4f}")
    
    # Identify potential root cause
    print(f"\n6. POTENTIAL ROOT CAUSE:")
    
    # Check for consistent patterns
    if all(m['row'] >= 48 for m in mismatches):
        print("   -> Mismatches start at row 48+ (second tile block)")
        print("   -> Possible issue with KV loop iteration or dS accumulation")
    
    row_set = set(m['row'] for m in mismatches)
    if len(row_set) == 1:
        row = list(row_set)[0]
        print(f"   -> All mismatches at single row R[{row}]")
        print("   -> Possible issue with specific GEMM tile or boundary handling")
    
    col_set = set(m['col'] for m in mismatches)
    min_col, max_col = min(col_set), max(col_set)
    if min_col >= 128:
        print(f"   -> All mismatches in hdim range [{min_col}, {max_col}]")
        print("   -> Possible issue with D192 head dim handling (hdim > 128)")
    
    # Check if all rows affected have same column pattern
    row_col_pattern = defaultdict(set)
    for m in mismatches:
        row_col_pattern[m['row']].add(m['col'])
    
    unique_col_sets = list(set(tuple(sorted(cols)) for cols in row_col_pattern.values()))
    if len(unique_col_sets) == 1 and len(unique_col_sets[0]) == len(col_set):
        print(f"   -> Same column pattern across all affected rows")
        print(f"   -> Columns affected: {sorted(col_set)[:10]}... (showing first 10)")

def main():
    parser = argparse.ArgumentParser(description='Analyze CPU vs GPU dQ mismatch')
    parser.add_argument('--cpu', default='cpu_dq.hex', help='CPU hex dump file')
    parser.add_argument('--gpu', default='gpu_dq.hex', help='GPU hex dump file')
    parser.add_argument('--output', default=None, help='Output file for detailed results')
    args = parser.parse_args()
    
    results = compare_files(args.cpu, args.gpu)
    print_summary(results)
    
    if args.output:
        import json
        # Convert mismatches for JSON serialization
        output_data = {
            'total_elements': results['total_elements'],
            'num_mismatches': len(results['mismatches']),
            'mismatch_by_batch': results['mismatch_by_batch'],
            'mismatch_by_head': results['mismatch_by_head'],
            'mismatch_by_row': results['mismatch_by_row'],
            'mismatch_by_col': results['mismatch_by_col'],
            'mismatches': results['mismatches'][:100]  # First 100 for brevity
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

if __name__ == '__main__':
    main()
