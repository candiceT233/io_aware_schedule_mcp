import sys
import os
import pandas as pd
import numpy as np
from enum import Enum, auto
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

# --- 1. Environment & Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path adjustment for your specific folder structure
WORKFLOW_ANALYSIS_DIR = os.path.join(CURRENT_DIR, 'spm', 'workflow_analysis')

if WORKFLOW_ANALYSIS_DIR not in sys.path:
    sys.path.append(WORKFLOW_ANALYSIS_DIR)

# Import real core algorithm modules
try:
    from modules.workflow_config import STORAGE_LIST
    from modules.workflow_interpolation import estimate_transfer_rates_for_workflow
    from modules.workflow_spm_calculator import (
        calculate_spm_for_edges, 
        calculate_spm_from_wfg,
        select_best_storage_and_parallelism
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import workflow modules: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Please ensure 'networkx', 'pandas', 'scikit-learn' are installed.", file=sys.stderr)
    sys.exit(1)

# --- 2. Core Service Class (SPM Integration) ---

class StorageRecommenderService:
    def __init__(self):
        self.ior_data = self._load_ior_data()
        print("Storage Service Initialized with Real SPM Algorithms.", file=sys.stderr)

    def _load_ior_data(self) -> pd.DataFrame:
        """Load IOR Benchmark Data"""
        csv_path = os.path.join(WORKFLOW_ANALYSIS_DIR, "..", "perf_profiles", "updated_master_ior_df.csv")
        csv_path = os.path.abspath(csv_path)
        
        if os.path.exists(csv_path):
            print(f"Loading IOR data from: {csv_path}", file=sys.stderr)
            return pd.read_csv(csv_path)
        else:
            print(f"WARNING: IOR benchmark data not found at {csv_path}!", file=sys.stderr)
            # Fallback check
            local_path = os.path.join(CURRENT_DIR, "perf_profiles", "updated_master_ior_df.csv")
            if os.path.exists(local_path):
                 return pd.read_csv(local_path)
            return pd.DataFrame()

    def _construct_workflow_df(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert API request params to workflow_analyzer DataFrame format
        """
        op = params.get("operation", "write").lower()
        
        # Construct single row dataframe simulation workflow CSV structure
        data = {
            "taskName": [params.get("task_name", "adhoc_task")],
            "operation": [op],
            "fileSize": [params.get("file_size", 0)], # MB
            "numNodes": [params.get("numNodes", 1)],
            "tasksPerNode": [params.get("tasksPerNode", 1)],
            "parallelism": [params.get("parallelism", 1)],
            "transferSize": [params.get("transfer_size", 1024*1024)], # Bytes
            "stageOrder": [params.get("stage_order", 1.0)],
            "storageType": ["unknown"], # Placeholder
            # --- CRITICAL FIXES FOR SPM CALCULATOR ---
            "taskPID": [0],  # Dummy PID
            "fileName": [params.get("fileName", "synthetic_data.dat")], 
            "start": [0.0], 
            "end": [1.0],   
            "duration": [1.0],
            # [FIX] Added 'prevTask' to prevent KeyError in graph building
            "prevTask": [params.get("prevTask", None)] 
        }
        
        df = pd.DataFrame(data)
        df['aggregateFilesizeMB'] = df['fileSize'] 
        
        return df

    def run_spm_analysis(self, df: pd.DataFrame, needs_persistence: bool = True) -> Dict[str, Any]:
        """
        Run the full SPM prediction pipeline
        """
        if self.ior_data.empty:
            return {"error": "IOR Benchmark data is missing on server."}

        # 1. Estimate Transfer Rates
        req_parallelism = df['parallelism'].iloc[0]
        allowed_p = sorted(list(set([1, 4, 8, 16, 24, 32, req_parallelism])))
        
        try:
            # Call core interpolation algorithm
            df_with_rates = estimate_transfer_rates_for_workflow(
                df, 
                self.ior_data, 
                STORAGE_LIST, 
                allowed_p, 
                multi_nodes=True, 
                debug=False 
            )
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {"error": f"Interpolation failed: {str(e)}"}

        # 2. Try Graph Analysis (SPM)
        spm_results = {}
        try:
            wfg = calculate_spm_for_edges(df_with_rates, debug=False)
            if len(wfg.edges) > 0:
                spm_results = calculate_spm_from_wfg(wfg, debug=False)
        except Exception as e:
            print(f"Graph analysis warning: {e}", file=sys.stderr)

        # 3. Handle Results (Graph-based OR Fallback for Single Task)
        
        # Scenario A: We have graph results (Chain analysis)
        if spm_results:
            key = list(spm_results.keys())[0]
            result_data = spm_results[key]
            rankings = []
            
            if 'rank' in result_data:
                raw_ranks = result_data.get('rank', {})
                for storage_name, p_dict in raw_ranks.items():
                    if needs_persistence and 'tmpfs' in storage_name.lower(): continue
                    for p_val, metrics in p_dict.items():
                        if isinstance(metrics, (int, float)):
                            score = metrics
                            est_time = 0
                        else:
                            score = metrics.get('final_score', 0)
                            est_time = metrics.get('est_time', 0)
                        
                        throughput = 0
                        file_size = df['fileSize'].iloc[0]
                        if est_time > 0: throughput = file_size / est_time

                        rankings.append({
                            "storage": storage_name,
                            "parallelism": p_val,
                            "score": round(score, 4),
                            "estimated_time_s": round(est_time, 4),
                            "estimated_throughput_mb_s": round(throughput, 2)
                        })
            
            rankings.sort(key=lambda x: x['score'], reverse=True)
            best = rankings[0] if rankings else {"storage": "Unknown", "score": 0}
            
            return {
                "best_option": best.get('storage'),
                "predicted_throughput": f"{best.get('estimated_throughput_mb_s', 0)} MiB/s",
                "all_candidates_ranked": rankings,
                "analysis_type": "chain_spm"
            }

        # Scenario B: No graph edges (Single Task), scan DataFrame columns directly
        else:
            print("No graph edges found. Falling back to direct DataFrame column scan.", file=sys.stderr)
            rankings = []
            row = df_with_rates.iloc[0]
            file_size = row['fileSize']
            
            for col in df_with_rates.columns:
                if col.startswith("estimated_trMiB_"):
                    parts = col.split('_')
                    if len(parts) >= 4:
                        storage = parts[2]
                        p_str = parts[3]
                        if not p_str.endswith('p'): continue
                        
                        try:
                            p_val = int(p_str[:-1])
                        except ValueError: continue
                        
                        throughput = row[col]
                        if pd.isna(throughput) or throughput <= 0: continue
                        if needs_persistence and 'tmpfs' in storage.lower(): continue
                        
                        est_time = file_size / throughput if throughput > 0 else 0
                        
                        rankings.append({
                            "storage": storage,
                            "parallelism": p_val,
                            "score": throughput, 
                            "estimated_time_s": round(est_time, 4),
                            "estimated_throughput_mb_s": round(throughput, 2)
                        })
            
            rankings.sort(key=lambda x: x['estimated_throughput_mb_s'], reverse=True)
            best = rankings[0] if rankings else {"storage": "Insufficient Data", "score": 0}
            
            return {
                "best_option": best.get('storage'),
                "predicted_throughput": f"{best.get('estimated_throughput_mb_s', 0)} MiB/s",
                "all_candidates_ranked": rankings,
                "analysis_type": "single_task_direct"
            }

    # --- Exposed Service Methods ---

    def analyze_single_task(self, file_size_mb, operation, transfer_size, parallelism, num_nodes, needs_persistence, task_name):
        params = {
            "task_name": task_name,
            "file_size": file_size_mb,
            "operation": operation,
            "transfer_size": transfer_size,
            "parallelism": parallelism,
            "numNodes": num_nodes,
            "tasksPerNode": parallelism // max(1, num_nodes),
            "prevTask": None 
        }
        df = self._construct_workflow_df(params)
        return self.run_spm_analysis(df, needs_persistence)

    def analyze_chain(self, p_ctx, c_ctx):
        common_file = "intermediate_data.h5"
        
        # [FIX] Pass transfer_size from context
        p_params = {
            "task_name": p_ctx['task_name'],
            "operation": "write",
            "file_size": p_ctx['file_size'],
            "numNodes": p_ctx['num_nodes'],
            "parallelism": p_ctx['parallelism'],
            "transfer_size": p_ctx.get('transfer_size', 1048576), 
            "stage_order": 1.0,
            "fileName": common_file,
            "prevTask": None 
        }
        
        c_params = {
            "task_name": c_ctx['task_name'],
            "operation": "read",
            "file_size": p_ctx['file_size'],
            "numNodes": c_ctx['num_nodes'],
            "parallelism": c_ctx['parallelism'],
            "transfer_size": c_ctx.get('transfer_size', 1048576),
            "stage_order": 2.0,
            "fileName": common_file,
            "prevTask": p_ctx['task_name'] # [FIX] Explicit link for graph builder
        }
        
        df_p = self._construct_workflow_df(p_params)
        df_c = self._construct_workflow_df(c_params)
        
        combined_df = pd.concat([df_p, df_c], ignore_index=True)
        return self.run_spm_analysis(combined_df, needs_persistence=False)


# --- 3. MCP Server Definition ---

def create_mcp_server() -> FastMCP:
    service = StorageRecommenderService()
    mcp = FastMCP("StorageRecommender")

    @mcp.tool()
    async def recommend_single_task_storage(
        file_size_mb: float,
        operation: str,
        transfer_size_bytes: int,
        parallelism: int = 1,
        num_nodes: int = 1,
        needs_persistence: bool = True,
        task_name: str = "task_1"
    ) -> Dict[str, Any]:
        """Recommend storage for a single task."""
        return service.analyze_single_task(
            file_size_mb, operation, transfer_size_bytes, 
            parallelism, num_nodes, needs_persistence, task_name
        )

    @mcp.tool()
    async def analyze_producer_consumer_transfer(
        producer_task_name: str,
        producer_nodes: int,
        producer_parallelism: int,
        consumer_task_name: str,
        consumer_nodes: int,
        consumer_parallelism: int,
        file_size_mb: float,
        transfer_size_bytes: int = 1048576 # [FIX] Added parameter to tool definition
    ) -> Dict[str, Any]:
        """Analyze best storage for Producer -> Consumer transfer."""
        p_ctx = {
            "task_name": producer_task_name, "num_nodes": producer_nodes, 
            "parallelism": producer_parallelism, "file_size": file_size_mb,
            "transfer_size": transfer_size_bytes
        }
        c_ctx = {
            "task_name": consumer_task_name, "num_nodes": consumer_nodes, 
            "parallelism": consumer_parallelism,
            "transfer_size": transfer_size_bytes
        }
        
        result = service.analyze_chain(p_ctx, c_ctx)
        result['flow'] = f"{producer_task_name} -> {consumer_task_name}"
        return result

    return mcp

if __name__ == "__main__":
    mcp_server = create_mcp_server()
    mcp_server.run(transport="stdio")