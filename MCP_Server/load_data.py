import pandas as pd
import numpy as np
from prettytable import PrettyTable

def perf_data_loader():
    df = pd.read_csv('https://raw.githubusercontent.com/candiceT233/spm/main/perf_profiles/updated_master_ior_df.csv')
    df = df[df['totalTime'] >= 0]
    df = df.dropna(subset=['trMiB', 'totalTime', 'numTasks', 'transferSize', 'aggregateFilesizeMB', 'storageType', 'operation'])
    df['trMiB'] = df['trMiB'].astype(float)
    df['totalTime'] = df['totalTime'].astype(float)
    df['numTasks'] = df['numTasks'].astype(int)
    df['transferSize'] = df['transferSize'].astype(int)
    df['aggregateFilesizeMB'] = df['aggregateFilesizeMB'].astype(float)
    df['numNodes'] = df['numNodes'].astype(int)
    df['tasksPerNode'] = df['tasksPerNode'].astype(int)
    df['parallelism'] = df['parallelism'].astype(int)
    return df

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]

def linear_interpolation(df, file_size, storage_type, operation, transfer_size, metric='trMiB'):
    subset = df[(df['storageType'] == storage_type) & 
                (df['operation'] == operation) & 
                (df['transferSize'] == transfer_size)]
    
    if len(subset) < 2:
        return np.nan
    
    # Group by aggregateFilesizeMB and take mean to handle duplicates
    grouped = subset.groupby('aggregateFilesizeMB')[metric].mean().reset_index()
    grouped = grouped.sort_values('aggregateFilesizeMB')
    sizes = grouped['aggregateFilesizeMB'].values
    values = grouped[metric].values
    
    if len(sizes) < 2:
        return np.nan
    
    if file_size < sizes[0]:
        x1, x2 = sizes[0], sizes[1]
        y1, y2 = values[0], values[1]
        if x2 - x1 == 0:
            return y1  # Fallback to nearest value if delta is zero (unlikely after grouping)
        return y1 + (file_size - x1) / (x2 - x1) * (y2 - y1)
    elif file_size > sizes[-1]:
        x1, x2 = sizes[-2], sizes[-1]
        y1, y2 = values[-2], values[-1]
        if x2 - x1 == 0:
            return y1  # Fallback to nearest value
        return y1 + (file_size - x1) / (x2 - x1) * (y2 - y1)
    
    idx = np.searchsorted(sizes, file_size)
    x1, x2 = sizes[idx-1], sizes[idx]
    y1, y2 = values[idx-1], values[idx]
    if x2 - x1 == 0:
        return y1  # Fallback to nearest value
    return y1 + (file_size - x1) / (x2 - x1) * (y2 - y1)

def recommend_storage(df, workflow_params):
    file_size = workflow_params['file_size']
    operation = workflow_params['operation']
    transfer_size = workflow_params['transfer_size']
    needs_persistence = workflow_params['needs_persistence']
    
    storage_types = ['tmpfs', 'beegfs', 'nfs']
    predictions = []
    
    for storage_type in storage_types:
        if needs_persistence and storage_type == 'tmpfs':
            continue
        throughput = linear_interpolation(df, file_size, storage_type, operation, transfer_size, 'trMiB')
        latency = linear_interpolation(df, file_size, storage_type, operation, transfer_size, 'totalTime')
        if not np.isnan(throughput) and not np.isnan(latency):
            predictions.append({
                'storageType': storage_type,
                'throughput_mb_s': throughput,
                'latency_s': latency
            })
    
    if predictions:
        return max(predictions, key=lambda x: x['throughput_mb_s'])
    
    # Fallback to historical best if no predictions
    stats = df.groupby(['storageType', 'operation'])['trMiB'].mean().reset_index()
    stats = stats[stats['operation'] == operation]
    if needs_persistence:
        stats = stats[stats['storageType'] != 'tmpfs']
    if not stats.empty:
        best_storage = stats.loc[stats['trMiB'].idxmax()]
        return {
            'storageType': best_storage['storageType'],
            'throughput_mb_s': best_storage['trMiB'],
            'latency_s': np.nan
        }
    return None

df = perf_data_loader()
for col in ['trMiB', 'totalTime']:
    df = remove_outliers(df, col)

agg_stats = df.groupby(['storageType', 'operation']).agg({
    'trMiB': ['mean', 'median', 'std', 'min', 'max'],
    'totalTime': ['mean', 'median', 'std'],
    'numTasks': ['mean', 'count'],
    'transferSize': ['mean', 'median'],
    'aggregateFilesizeMB': ['mean', 'median']
}).round(2)

agg_stats.columns = [
    'Avg trMiB (MB/s)', 'Median trMiB (MB/s)', 'Std trMiB', 'Min trMiB', 'Max trMiB',
    'Avg Time (s)', 'Median Time (s)', 'Std Time',
    'Avg Tasks', 'Record Count',
    'Avg Transfer Size (B)', 'Median Transfer Size (B)',
    'Avg Aggregate Size (MB)', 'Median Aggregate Size (MB)'
]

agg_stats = agg_stats.reset_index()

table = PrettyTable([
    'Storage Type', 'Operation',
    'Avg trMiB (MB/s)', 'Median trMiB (MB/s)', 'Std trMiB', 'Min trMiB', 'Max trMiB',
    'Avg Time (s)', 'Median Time (s)', 'Std Time',
    'Avg Tasks', 'Record Count',
    'Avg Transfer Size (B)', 'Median Transfer Size (B)',
    'Avg Aggregate Size (MB)', 'Median Aggregate Size (MB)'
])

for _, row in agg_stats.iterrows():
    table.add_row([
        row['storageType'], row['operation'],
        row['Avg trMiB (MB/s)'], row['Median trMiB (MB/s)'], row['Std trMiB'],
        row['Min trMiB'], row['Max trMiB'],
        row['Avg Time (s)'], row['Median Time (s)'], row['Std Time'],
        row['Avg Tasks'], row['Record Count'],
        row['Avg Transfer Size (B)'], row['Median Transfer Size (B)'],
        row['Avg Aggregate Size (MB)'], row['Median Aggregate Size (MB)']
    ])

table.align['Storage Type'] = 'l'
table.align['Operation'] = 'l'
for col in table.field_names[2:]:
    table.align[col] = 'r'
table.sortby = 'Storage Type'

print(table)



# Example workflow parameters
workflow_params_list = [
    # 1. 小文件写操作，持久化
    {
        'file_size': 50,        # 50MB
        'operation': 'write',
        'transfer_size': 1048576,  # 1MB
        'needs_persistence': True
    },
    # 2. 大文件读操作，持久化
    {
        'file_size': 20000,     # 20GB
        'operation': 'read',
        'transfer_size': 4194304,  # 4MB
        'needs_persistence': True
    },
    # 3. 中等文件写操作，不需要持久化 → tmpfs 可选
    {
        'file_size': 5000,      # 5GB
        'operation': 'write',
        'transfer_size': 4194304,  # 4MB
        'needs_persistence': False
    },
    # 4. 极小文件读操作，持久化
    {
        'file_size': 10,        # 10MB
        'operation': 'read',
        'transfer_size': 524288,   # 512KB
        'needs_persistence': True
    },
    # 5. 超大文件写操作，持久化
    {
        'file_size': 100000,    # 100GB
        'operation': 'write',
        'transfer_size': 8388608,  # 8MB
        'needs_persistence': True
    },
    # 6. 小文件读操作，不需要持久化
    {
        'file_size': 200,       # 200MB
        'operation': 'read',
        'transfer_size': 2097152,  # 2MB
        'needs_persistence': False
    },
    # 7. 大文件写操作，小 transfer size
    {
        'file_size': 30000,     # 30GB
        'operation': 'write',
        'transfer_size': 131072,   # 128KB
        'needs_persistence': True
    },
    # 8. 中等文件读操作，持久化
    {
        'file_size': 8000,      # 8GB
        'operation': 'read',
        'transfer_size': 4194304,  # 4MB
        'needs_persistence': True
    }
]
for d in workflow_params_list:
    recommendation = recommend_storage(df, d)
    if recommendation:
        print("\n *********************")
        print(d)
        print("Recommendation:")
        print(f"Recommended Storage System: {recommendation['storageType']}")
        print(f"Predicted Throughput: {recommendation['throughput_mb_s']:.2f} MB/s")
        print(f"Predicted Latency: {recommendation['latency_s']:.4f} s")
        print("\n *********************")
