import sys
import os
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from mcp.server.fastmcp import FastMCP

# --- 1. Environment & Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_ANALYSIS_DIR = os.path.join(CURRENT_DIR, 'spm', 'workflow_analysis')

if WORKFLOW_ANALYSIS_DIR not in sys.path:
    sys.path.append(WORKFLOW_ANALYSIS_DIR)

try:
    from modules.workflow_config import STORAGE_LIST
    from modules.workflow_interpolation import estimate_transfer_rates_for_workflow
    from modules.workflow_spm_calculator import (
        calculate_spm_for_edges,
        calculate_spm_from_wfg,
    )
except ImportError as e:
    print(f"WARNING: Could not import workflow modules: {e}", file=sys.stderr)
    print("Running in mock mode for testing", file=sys.stderr)
    STORAGE_LIST = ["ssd", "beegfs", "tmpfs"]

# --- 2. Configuration ---
CONTEXT_MODE = os.getenv("CONTEXT_MODE", "stateless").lower()
print(f"[CONFIG] Context Management Mode: {CONTEXT_MODE}", file=sys.stderr)


# --- 3. Metrics Collection ---
@dataclass
class ContextMetrics:
    """收集上下文管理相关指标"""
    mode: str = "unknown"
    total_tool_calls: int = 0
    context_hits: int = 0
    context_misses: int = 0
    total_response_chars: int = 0
    full_response_count: int = 0
    compact_response_count: int = 0
    total_compute_time_ms: float = 0
    total_context_lookup_time_ms: float = 0
    active_workflows: int = 0
    active_sessions: int = 0
    
    def to_dict(self) -> dict:
        total = self.total_tool_calls or 1
        return {
            "mode": self.mode,
            "total_tool_calls": self.total_tool_calls,
            "context_hits": self.context_hits,
            "context_misses": self.context_misses,
            "context_hit_rate": self.context_hits / total,
            "total_response_chars": self.total_response_chars,
            "avg_response_chars": self.total_response_chars / total,
            "compact_ratio": self.compact_response_count / total,
        }

_metrics = ContextMetrics(mode=CONTEXT_MODE)


# --- 4. Workflow Context Data Structures ---
@dataclass
class TaskAnalysis:
    """单个任务的分析结果"""
    task_name: str
    file_size_mb: float
    operation: str
    parallelism: int
    num_nodes: int
    best_storage: str
    predicted_throughput_mb_s: float
    all_rankings: list
    analyzed_at: float = field(default_factory=time.time)


@dataclass 
class WorkflowContext:
    """工作流上下文"""
    workflow_id: str
    tasks: Dict[str, TaskAnalysis] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    chain_analysis: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def add_task(self, analysis: TaskAnalysis):
        self.tasks[analysis.task_name] = analysis
        if analysis.task_name not in self.task_order:
            self.task_order.append(analysis.task_name)
        self.last_accessed_at = time.time()
        self.access_count += 1
    
    def get_summary(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "num_tasks": len(self.tasks),
            "task_names": self.task_order,
            "best_storages": {name: t.best_storage for name, t in self.tasks.items()},
            "has_chain_analysis": self.chain_analysis is not None,
        }
    
    def get_full_context(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "tasks": {name: asdict(t) for name, t in self.tasks.items()},
            "task_order": self.task_order,
            "chain_analysis": self.chain_analysis,
        }


# --- 5. Context Manager Implementations ---
class ContextManager(ABC):
    @abstractmethod
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        pass
    
    @abstractmethod
    def save_workflow(self, context: WorkflowContext):
        pass
    
    @abstractmethod
    def list_workflows(self) -> List[str]:
        pass
    
    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        pass


class StatelessContextManager(ContextManager):
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        return None
    
    def save_workflow(self, context: WorkflowContext):
        pass
    
    def list_workflows(self) -> List[str]:
        return []
    
    def delete_workflow(self, workflow_id: str) -> bool:
        return False
    
    def get_stats(self) -> dict:
        return {"mode": "stateless", "stored_workflows": 0}


class SessionContextManager(ContextManager):
    def __init__(self):
        self._workflows: Dict[str, WorkflowContext] = {}
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        ctx = self._workflows.get(workflow_id)
        if ctx:
            ctx.last_accessed_at = time.time()
            ctx.access_count += 1
        return ctx
    
    def save_workflow(self, context: WorkflowContext):
        self._workflows[context.workflow_id] = context
    
    def list_workflows(self) -> List[str]:
        return list(self._workflows.keys())
    
    def delete_workflow(self, workflow_id: str) -> bool:
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False
    
    def get_stats(self) -> dict:
        return {
            "mode": "session",
            "stored_workflows": len(self._workflows),
            "workflow_ids": list(self._workflows.keys()),
        }


class PersistentContextManager(ContextManager):
    def __init__(self):
        self._redis = None
        self._local_cache: Dict[str, WorkflowContext] = {}
        self._connect_redis()
    
    def _connect_redis(self):
        try:
            import redis
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            self._redis.ping()
            print("[CONTEXT] Redis connected", file=sys.stderr)
        except Exception as e:
            print(f"[CONTEXT] Redis failed: {e}", file=sys.stderr)
            self._redis = None
    
    def _serialize(self, context: WorkflowContext) -> str:
        data = {
            "workflow_id": context.workflow_id,
            "tasks": {name: asdict(t) for name, t in context.tasks.items()},
            "task_order": context.task_order,
            "chain_analysis": context.chain_analysis,
            "created_at": context.created_at,
            "last_accessed_at": context.last_accessed_at,
            "access_count": context.access_count,
        }
        return json.dumps(data)
    
    def _deserialize(self, data: str) -> WorkflowContext:
        d = json.loads(data)
        ctx = WorkflowContext(
            workflow_id=d["workflow_id"],
            task_order=d.get("task_order", []),
            chain_analysis=d.get("chain_analysis"),
            created_at=d.get("created_at", time.time()),
            last_accessed_at=d.get("last_accessed_at", time.time()),
            access_count=d.get("access_count", 0),
        )
        for name, task_data in d.get("tasks", {}).items():
            ctx.tasks[name] = TaskAnalysis(**task_data)
        return ctx
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        if workflow_id in self._local_cache:
            ctx = self._local_cache[workflow_id]
            ctx.last_accessed_at = time.time()
            ctx.access_count += 1
            return ctx
        
        if self._redis:
            try:
                data = self._redis.get(f"workflow:{workflow_id}")
                if data:
                    ctx = self._deserialize(data)
                    ctx.last_accessed_at = time.time()
                    ctx.access_count += 1
                    self._local_cache[workflow_id] = ctx
                    return ctx
            except Exception:
                pass
        return None
    
    def save_workflow(self, context: WorkflowContext):
        self._local_cache[context.workflow_id] = context
        if self._redis:
            try:
                self._redis.setex(
                    f"workflow:{context.workflow_id}",
                    60 * 60 * 24 * 30,
                    self._serialize(context)
                )
            except Exception:
                pass
    
    def list_workflows(self) -> List[str]:
        workflows = set(self._local_cache.keys())
        if self._redis:
            try:
                keys = self._redis.keys("workflow:*")
                workflows.update(k.replace("workflow:", "") for k in keys)
            except Exception:
                pass
        return list(workflows)
    
    def delete_workflow(self, workflow_id: str) -> bool:
        deleted = workflow_id in self._local_cache
        self._local_cache.pop(workflow_id, None)
        if self._redis:
            try:
                self._redis.delete(f"workflow:{workflow_id}")
                deleted = True
            except Exception:
                pass
        return deleted
    
    def get_stats(self) -> dict:
        return {
            "mode": "persistent",
            "local_cache_count": len(self._local_cache),
            "redis_connected": self._redis is not None,
        }


def create_context_manager() -> ContextManager:
    if CONTEXT_MODE == "stateless":
        return StatelessContextManager()
    elif CONTEXT_MODE == "session":
        return SessionContextManager()
    elif CONTEXT_MODE == "persistent":
        return PersistentContextManager()
    else:
        return StatelessContextManager()


# --- 6. Core Service Class ---
class StorageRecommenderService:
    def __init__(self):
        self.ior_data = self._load_ior_data()
        print(f"[SERVICE] Initialized, IOR data rows: {len(self.ior_data)}", file=sys.stderr)

    def _load_ior_data(self) -> pd.DataFrame:
        paths = [
            os.path.join(WORKFLOW_ANALYSIS_DIR, "..", "perf_profiles", "updated_master_ior_df.csv"),
            os.path.join(CURRENT_DIR, "perf_profiles", "updated_master_ior_df.csv"),
            os.path.join(CURRENT_DIR, "spm", "perf_profiles", "updated_master_ior_df.csv"),
        ]
        for path in paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return pd.read_csv(abs_path)
        return pd.DataFrame()

    def _construct_workflow_df(self, params: dict) -> pd.DataFrame:
        op = params.get("operation", "write").lower()
        data = {
            "taskName": [params.get("task_name", "adhoc_task")],
            "operation": [op],
            "fileSize": [params.get("file_size", 0)],
            "numNodes": [params.get("numNodes", 1)],
            "tasksPerNode": [params.get("tasksPerNode", 1)],
            "parallelism": [params.get("parallelism", 1)],
            "transferSize": [params.get("transfer_size", 1048576)],
            "stageOrder": [params.get("stage_order", 1.0)],
            "storageType": ["unknown"],
            "taskPID": [0],
            "fileName": [params.get("fileName", "synthetic_data.dat")],
            "start": [0.0],
            "end": [1.0],
            "duration": [1.0],
            "prevTask": [params.get("prevTask", None)]
        }
        df = pd.DataFrame(data)
        df['aggregateFilesizeMB'] = df['fileSize']
        return df

    def analyze_single_task(
        self, 
        file_size_mb: float, 
        operation: str, 
        transfer_size: int,
        parallelism: int, 
        num_nodes: int, 
        needs_persistence: bool, 
        task_name: str
    ) -> TaskAnalysis:
        if self.ior_data.empty:
            return TaskAnalysis(
                task_name=task_name,
                file_size_mb=file_size_mb,
                operation=operation,
                parallelism=parallelism,
                num_nodes=num_nodes,
                best_storage="ssd" if needs_persistence else "tmpfs",
                predicted_throughput_mb_s=500.0,
                all_rankings=[
                    {"storage": "ssd", "score": 0.9, "throughput_mb_s": 500},
                    {"storage": "beegfs", "score": 0.7, "throughput_mb_s": 300},
                ]
            )
        
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
        result = self._run_spm_analysis(df, needs_persistence)
        
        return TaskAnalysis(
            task_name=task_name,
            file_size_mb=file_size_mb,
            operation=operation,
            parallelism=parallelism,
            num_nodes=num_nodes,
            best_storage=result.get("best_option", "unknown"),
            predicted_throughput_mb_s=float(str(result.get("predicted_throughput", "0 MiB/s")).split()[0]),
            all_rankings=result.get("all_candidates_ranked", [])
        )

    def _run_spm_analysis(self, df: pd.DataFrame, needs_persistence: bool = True) -> dict:
        if self.ior_data.empty:
            return {"error": "IOR Benchmark data is missing."}

        req_parallelism = df['parallelism'].iloc[0]
        allowed_p = sorted(list(set([1, 4, 8, 16, 24, 32, req_parallelism])))

        try:
            df_with_rates = estimate_transfer_rates_for_workflow(
                df, self.ior_data, STORAGE_LIST, allowed_p,
                multi_nodes=True, debug=False
            )
        except Exception as e:
            return {"error": f"Interpolation failed: {str(e)}"}

        spm_results = {}
        try:
            wfg = calculate_spm_for_edges(df_with_rates, debug=False)
            if len(wfg.edges) > 0:
                spm_results = calculate_spm_from_wfg(wfg, debug=False)
        except Exception:
            pass

        if spm_results:
            key = list(spm_results.keys())[0]
            result_data = spm_results[key]
            rankings = []
            if 'rank' in result_data:
                raw_ranks = result_data.get('rank', {})
                for storage_name, p_dict in raw_ranks.items():
                    if needs_persistence and 'tmpfs' in storage_name.lower():
                        continue
                    for p_val, metrics in p_dict.items():
                        score = metrics if isinstance(metrics, (int, float)) else metrics.get('final_score', 0)
                        est_time = 0 if isinstance(metrics, (int, float)) else metrics.get('est_time', 0)
                        throughput = df['fileSize'].iloc[0] / est_time if est_time > 0 else 0
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
            }
        else:
            rankings = []
            row = df_with_rates.iloc[0]
            file_size = row['fileSize']
            for col in df_with_rates.columns:
                if col.startswith("estimated_trMiB_"):
                    parts = col.split('_')
                    if len(parts) < 4 or not parts[3].endswith('p'):
                        continue
                    storage = parts[2]
                    try:
                        p_val = int(parts[3][:-1])
                    except ValueError:
                        continue
                    throughput = row[col]
                    if pd.isna(throughput) or throughput <= 0:
                        continue
                    if needs_persistence and 'tmpfs' in storage.lower():
                        continue
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
            }


# --- 7. MCP Server (Gemini Compatible) ---
def create_mcp_server() -> FastMCP:
    service = StorageRecommenderService()
    context_mgr = create_context_manager()
    mcp = FastMCP("StorageRecommender")
    
    print(f"[MCP] Server created with {CONTEXT_MODE} context management", file=sys.stderr)

    @mcp.tool()
    async def analyze_task(
        workflow_id: str,
        task_name: str,
        file_size_mb: float,
        operation: str,
        parallelism: int = 1,
        num_nodes: int = 1,
        transfer_size_bytes: int = 1048576,
        needs_persistence: bool = True,
    ) -> str:
        """
        Analyze storage recommendation for a single task.
        
        Args:
            workflow_id: Workflow identifier for grouping tasks
            task_name: Name of the task
            file_size_mb: File size in MiB
            operation: "read" or "write"
            parallelism: Number of parallel processes
            num_nodes: Number of compute nodes
            transfer_size_bytes: I/O transfer block size
            needs_persistence: Whether storage must be persistent
        
        Returns:
            JSON string with storage recommendation
        """
        global _metrics
        _metrics.total_tool_calls += 1
        
        # Check context
        ctx = context_mgr.get_workflow(workflow_id)
        
        if ctx and task_name in ctx.tasks:
            _metrics.context_hits += 1
            _metrics.compact_response_count += 1
            existing = ctx.tasks[task_name]
            response = {
                "status": "context_hit",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
                "task_name": task_name,
                "from_context": True,
                "recommendation": {
                    "best_storage": existing.best_storage,
                    "predicted_throughput_mb_s": existing.predicted_throughput_mb_s,
                },
                "hint": f"Use get_task_detail for full rankings",
            }
            result = json.dumps(response)
            _metrics.total_response_chars += len(result)
            return result
        
        # Compute
        _metrics.context_misses += 1
        analysis = service.analyze_single_task(
            file_size_mb, operation, transfer_size_bytes,
            parallelism, num_nodes, needs_persistence, task_name
        )
        
        # Save context
        if ctx is None:
            ctx = WorkflowContext(workflow_id=workflow_id)
        ctx.add_task(analysis)
        context_mgr.save_workflow(ctx)
        _metrics.active_workflows = len(context_mgr.list_workflows())
        
        if CONTEXT_MODE == "stateless":
            _metrics.full_response_count += 1
            response = {
                "status": "computed",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
                "task_name": task_name,
                "recommendation": {
                    "best_storage": analysis.best_storage,
                    "predicted_throughput_mb_s": analysis.predicted_throughput_mb_s,
                },
                "all_rankings": analysis.all_rankings,
            }
        else:
            _metrics.compact_response_count += 1
            response = {
                "status": "computed_and_saved",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
                "task_name": task_name,
                "recommendation": {
                    "best_storage": analysis.best_storage,
                    "predicted_throughput_mb_s": analysis.predicted_throughput_mb_s,
                },
                "context_saved": True,
                "workflow_tasks": ctx.task_order,
            }
        
        result = json.dumps(response)
        _metrics.total_response_chars += len(result)
        return result

    @mcp.tool()
    async def get_task_detail(
        workflow_id: str,
        task_name: str,
    ) -> str:
        """
        Get detailed analysis for a previously analyzed task.
        Requires context (session/persistent mode).
        
        Args:
            workflow_id: Workflow identifier
            task_name: Task name to retrieve
        
        Returns:
            JSON string with full task details including all rankings
        """
        global _metrics
        _metrics.total_tool_calls += 1
        
        ctx = context_mgr.get_workflow(workflow_id)
        
        if ctx is None:
            _metrics.context_misses += 1
            return json.dumps({
                "status": "workflow_not_found",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
            })
        
        if task_name not in ctx.tasks:
            _metrics.context_misses += 1
            return json.dumps({
                "status": "task_not_found",
                "workflow_id": workflow_id,
                "task_name": task_name,
                "available_tasks": ctx.task_order,
            })
        
        _metrics.context_hits += 1
        task = ctx.tasks[task_name]
        return json.dumps({
            "status": "found",
            "context_mode": CONTEXT_MODE,
            "workflow_id": workflow_id,
            "task": asdict(task),
        })

    @mcp.tool()
    async def analyze_workflow(
        workflow_id: str,
        tasks_json: str,
    ) -> str:
        """
        Analyze multiple tasks in a workflow.
        
        Args:
            workflow_id: Workflow identifier
            tasks_json: JSON string array of tasks. Each task should have:
                - name: task name (string)
                - file_size_mb: file size in MiB (number)
                - operation: "read" or "write" (string)
                - parallelism: parallel processes (number, optional, default 1)
                - num_nodes: compute nodes (number, optional, default 1)
                - needs_persistence: require persistent storage (boolean, optional, default true)
                
                Example: '[{"name":"task1","file_size_mb":1024,"operation":"write","parallelism":8}]'
        
        Returns:
            JSON string with analysis results for all tasks
        """
        global _metrics
        _metrics.total_tool_calls += 1
        
        # Parse tasks JSON
        try:
            tasks = json.loads(tasks_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid tasks_json: {str(e)}"})
        
        if not isinstance(tasks, list):
            return json.dumps({"error": "tasks_json must be a JSON array"})
        
        # Check existing context
        ctx = context_mgr.get_workflow(workflow_id)
        if ctx and len(ctx.tasks) == len(tasks):
            all_match = all(t.get("name", t.get("task_name")) in ctx.tasks for t in tasks)
            if all_match:
                _metrics.context_hits += 1
                return json.dumps({
                    "status": "context_hit",
                    "context_mode": CONTEXT_MODE,
                    "workflow_id": workflow_id,
                    "from_context": True,
                    "summary": ctx.get_summary(),
                })
        
        _metrics.context_misses += 1
        
        if ctx is None:
            ctx = WorkflowContext(workflow_id=workflow_id)
        
        results = []
        for task_spec in tasks:
            task_name = task_spec.get("name", task_spec.get("task_name", "unnamed"))
            
            analysis = service.analyze_single_task(
                file_size_mb=task_spec.get("file_size_mb", 1024),
                operation=task_spec.get("operation", "write"),
                transfer_size=task_spec.get("transfer_size_bytes", 1048576),
                parallelism=task_spec.get("parallelism", 1),
                num_nodes=task_spec.get("num_nodes", 1),
                needs_persistence=task_spec.get("needs_persistence", True),
                task_name=task_name,
            )
            ctx.add_task(analysis)
            results.append({
                "task_name": task_name,
                "best_storage": analysis.best_storage,
                "throughput_mb_s": analysis.predicted_throughput_mb_s,
            })
        
        context_mgr.save_workflow(ctx)
        
        if CONTEXT_MODE == "stateless":
            _metrics.full_response_count += 1
            response = {
                "status": "computed",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
                "task_count": len(results),
                "results": results,
            }
        else:
            _metrics.compact_response_count += 1
            response = {
                "status": "computed_and_saved",
                "context_mode": CONTEXT_MODE,
                "workflow_id": workflow_id,
                "task_count": len(results),
                "results": results,
                "context_saved": True,
            }
        
        return json.dumps(response)

    @mcp.tool()
    async def analyze_producer_consumer(
        workflow_id: str,
        producer_name: str,
        producer_file_size_mb: float,
        producer_parallelism: int,
        producer_nodes: int,
        consumer_name: str,
        consumer_parallelism: int,
        consumer_nodes: int,
        needs_persistence: bool = False,
    ) -> str:
        """
        Analyze a producer-consumer pair where producer writes data and consumer reads it.
        
        Args:
            workflow_id: Workflow identifier
            producer_name: Name of producer task (writes data)
            producer_file_size_mb: Size of data produced in MiB
            producer_parallelism: Producer parallel processes
            producer_nodes: Producer compute nodes
            consumer_name: Name of consumer task (reads data)
            consumer_parallelism: Consumer parallel processes
            consumer_nodes: Consumer compute nodes
            needs_persistence: Whether intermediate data needs persistence
        
        Returns:
            JSON string with recommendations for both producer and consumer
        """
        global _metrics
        _metrics.total_tool_calls += 1
        
        ctx = context_mgr.get_workflow(workflow_id)
        if ctx is None:
            ctx = WorkflowContext(workflow_id=workflow_id)
        
        # Analyze producer (write)
        producer_analysis = service.analyze_single_task(
            file_size_mb=producer_file_size_mb,
            operation="write",
            transfer_size=1048576,
            parallelism=producer_parallelism,
            num_nodes=producer_nodes,
            needs_persistence=needs_persistence,
            task_name=producer_name,
        )
        ctx.add_task(producer_analysis)
        
        # Analyze consumer (read same data)
        consumer_analysis = service.analyze_single_task(
            file_size_mb=producer_file_size_mb,
            operation="read",
            transfer_size=1048576,
            parallelism=consumer_parallelism,
            num_nodes=consumer_nodes,
            needs_persistence=needs_persistence,
            task_name=consumer_name,
        )
        ctx.add_task(consumer_analysis)
        
        # Save context
        context_mgr.save_workflow(ctx)
        
        # Determine if same storage is optimal
        same_storage = producer_analysis.best_storage == consumer_analysis.best_storage
        
        response = {
            "status": "analyzed",
            "context_mode": CONTEXT_MODE,
            "workflow_id": workflow_id,
            "producer": {
                "name": producer_name,
                "operation": "write",
                "best_storage": producer_analysis.best_storage,
                "throughput_mb_s": producer_analysis.predicted_throughput_mb_s,
            },
            "consumer": {
                "name": consumer_name,
                "operation": "read",
                "best_storage": consumer_analysis.best_storage,
                "throughput_mb_s": consumer_analysis.predicted_throughput_mb_s,
            },
            "recommendation": {
                "use_same_storage": same_storage,
                "optimal_storage": producer_analysis.best_storage if same_storage else "see individual recommendations",
            },
            "flow": f"{producer_name} (write) -> {consumer_name} (read)",
        }
        
        return json.dumps(response)

    @mcp.tool()
    async def compare_workflows(
        workflow_ids_csv: str,
    ) -> str:
        """
        Compare multiple analyzed workflows. Requires session/persistent context.
        
        Args:
            workflow_ids_csv: Comma-separated workflow IDs to compare
                Example: "workflow_1,workflow_2,workflow_3"
        
        Returns:
            JSON string with comparison results
        """
        global _metrics
        _metrics.total_tool_calls += 1
        
        workflow_ids = [wid.strip() for wid in workflow_ids_csv.split(",")]
        
        if CONTEXT_MODE == "stateless":
            return json.dumps({
                "status": "not_available",
                "context_mode": CONTEXT_MODE,
                "message": "Comparison requires session/persistent context mode",
            })
        
        comparisons = []
        missing = []
        
        for wf_id in workflow_ids:
            ctx = context_mgr.get_workflow(wf_id)
            if ctx:
                _metrics.context_hits += 1
                comparisons.append({
                    "workflow_id": wf_id,
                    "task_count": len(ctx.tasks),
                    "tasks": {
                        name: {
                            "best_storage": t.best_storage,
                            "throughput_mb_s": t.predicted_throughput_mb_s,
                        }
                        for name, t in ctx.tasks.items()
                    },
                })
            else:
                _metrics.context_misses += 1
                missing.append(wf_id)
        
        return json.dumps({
            "status": "compared",
            "context_mode": CONTEXT_MODE,
            "compared": comparisons,
            "missing": missing,
        })

    @mcp.tool()
    async def list_workflows() -> str:
        """
        List all workflows in current context.
        
        Returns:
            JSON string with list of workflow IDs
        """
        workflows = context_mgr.list_workflows()
        return json.dumps({
            "context_mode": CONTEXT_MODE,
            "workflows": workflows,
            "count": len(workflows),
        })

    @mcp.tool()
    async def clear_workflow(workflow_id: str) -> str:
        """
        Clear a workflow from context.
        
        Args:
            workflow_id: Workflow to clear
        
        Returns:
            JSON string with status
        """
        deleted = context_mgr.delete_workflow(workflow_id)
        return json.dumps({
            "status": "deleted" if deleted else "not_found",
            "context_mode": CONTEXT_MODE,
            "workflow_id": workflow_id,
        })

    @mcp.tool()
    async def get_metrics() -> str:
        """
        Get context management metrics for RQ1 validation.
        
        Returns:
            JSON string with metrics
        """
        return json.dumps({
            "metrics": _metrics.to_dict(),
            "context_stats": context_mgr.get_stats(),
        })

    @mcp.tool()
    async def reset_metrics() -> str:
        """
        Reset all metrics counters.
        
        Returns:
            JSON string with previous metrics
        """
        global _metrics
        old = _metrics.to_dict()
        _metrics = ContextMetrics(mode=CONTEXT_MODE)
        return json.dumps({"status": "reset", "previous": old})

    return mcp


if __name__ == "__main__":
    
    mcp_server = create_mcp_server()
    mcp_server.run(transport="stdio")