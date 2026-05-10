"""
Phase 3: Simple Workflow Orchestration

Sequential chain executor for basic multi-step tasks.
Foundation for Phase 4's advanced DAG-based orchestration.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel
from enum import Enum


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """Single step in a workflow."""
    name: str
    func: Optional[Callable] = None  # Function to execute
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class WorkflowResult(BaseModel):
    """Result of workflow execution."""
    workflow_id: str
    status: StepStatus
    steps: List[WorkflowStep]
    total_time: float
    output: Optional[Any] = None


class SimpleWorkflowExecutor:
    """
    Executes sequential workflows step-by-step.
    
    Features:
    - Sequential step execution
    - Error handling with continue/skip options
    - Timing and logging
    - Result aggregation
    """
    
    def __init__(self, max_retries: int = 1, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._workflows: Dict[str, WorkflowResult] = {}
    
    async def execute(self, workflow_id: str, 
                      steps: List[WorkflowStep],
                      on_error: str = "stop") -> WorkflowResult:
        """
        Execute a sequential workflow.
        
        Args:
            workflow_id: Unique identifier for this workflow
            steps: List of steps to execute in order
            on_error: Error handling strategy ("stop", "continue", "retry")
            
        Returns:
            WorkflowResult with all step outcomes
        """
        start_time = time.time()
        executed_steps = []
        overall_status = StepStatus.RUNNING
        
        for i, step in enumerate(steps):
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
            
            try:
                # Execute step
                result = await self._execute_step(step)
                step.result = result
                step.status = StepStatus.COMPLETED
                
            except Exception as e:
                step.error = str(e)
                
                if on_error == "stop":
                    step.status = StepStatus.FAILED
                    overall_status = StepStatus.FAILED
                    executed_steps.append(step)
                    # Skip remaining steps
                    for remaining_step in steps[i+1:]:
                        remaining_step.status = StepStatus.SKIPPED
                        executed_steps.append(remaining_step)
                    break
                    
                elif on_error == "retry":
                    # Retry logic
                    for attempt in range(self.max_retries):
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        try:
                            result = await self._execute_step(step)
                            step.result = result
                            step.status = StepStatus.COMPLETED
                            step.error = None
                            break
                        except Exception as retry_err:
                            step.error = f"Attempt {attempt + 1}: {str(retry_err)}"
                    
                    if step.status != StepStatus.COMPLETED:
                        step.status = StepStatus.FAILED
                        if on_error == "retry":
                            overall_status = StepStatus.FAILED
                            break
                            
                elif on_error == "continue":
                    step.status = StepStatus.FAILED
                    # Continue to next step
            
            step.completed_at = time.time()
            executed_steps.append(step)
        
        # Determine final status
        if overall_status != StepStatus.FAILED:
            failed_steps = [s for s in executed_steps if s.status == StepStatus.FAILED]
            overall_status = StepStatus.FAILED if failed_steps else StepStatus.COMPLETED
        
        total_time = time.time() - start_time
        
        # Aggregate output from successful steps
        output = self._aggregate_output(executed_steps)
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            status=overall_status,
            steps=executed_steps,
            total_time=total_time,
            output=output
        )
        
        self._workflows[workflow_id] = result
        return result
    
    async def _execute_step(self, step: WorkflowStep) -> Any:
        """Execute a single workflow step."""
        if not step.func:
            raise ValueError(f"Step '{step.name}' has no function")
        
        args = step.args or []
        kwargs = step.kwargs or {}
        
        # Handle both sync and async functions
        if asyncio.iscoroutinefunction(step.func):
            return await step.func(*args, **kwargs)
        else:
            return step.func(*args, **kwargs)
    
    def _aggregate_output(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Aggregate results from all successful steps."""
        output = {}
        for step in steps:
            if step.status == StepStatus.COMPLETED and step.result is not None:
                output[step.name] = step.result
        return output
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get workflow result by ID."""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self) -> List[str]:
        """List all workflow IDs."""
        return list(self._workflows.keys())
    
    def clear(self):
        """Clear all stored workflows."""
        self._workflows.clear()


# Helper function to create workflow steps
def create_step(name: str, func: Callable, 
                args: Optional[List[Any]] = None,
                kwargs: Optional[Dict[str, Any]] = None) -> WorkflowStep:
    """Create a workflow step."""
    return WorkflowStep(
        name=name,
        func=func,
        args=args or [],
        kwargs=kwargs or {}
    )


# Global executor instance
_workflow_executor: Optional[SimpleWorkflowExecutor] = None


def get_workflow_executor() -> SimpleWorkflowExecutor:
    """Get or create global workflow executor instance."""
    global _workflow_executor
    if _workflow_executor is None:
        _workflow_executor = SimpleWorkflowExecutor()
    return _workflow_executor
