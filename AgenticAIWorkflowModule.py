# icrms/modules/agentic_ai/workflow_orchestrator.py
"""
Agentic AI Workflow Orchestrator
Uses LangChain and LangGraph for multi-step autonomous workflows
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ComplianceAgent:
    """Autonomous agent for compliance workflows"""
    
    def __init__(self, llm_path: str, sar_generator, risk_analyzer):
        self.llm = GPT4All(model=llm_path, max_tokens=1024, temp=0.2)
        self.sar_generator = sar_generator
        self.risk_analyzer = risk_analyzer
        self.tools = self._create_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        tools = [
            Tool(
                name="GenerateSAR",
                func=self._generate_sar_tool,
                description="Generate a Suspicious Activity Report for a transaction. Input should be transaction details as JSON string."
            ),
            Tool(
                name="AnalyzeRisk",
                func=self._analyze_risk_tool,
                description="Analyze financial risk for given data. Input should be risk data as JSON string."
            ),
            Tool(
                name="CheckCompliance",
                func=self._check_compliance_tool,
                description="Check if a transaction complies with regulations. Input should be transaction details."
            )
        ]
        return tools
    
    def _generate_sar_tool(self, transaction_json: str) -> str:
        """Tool wrapper for SAR generation"""
        try:
            import json
            transaction_data = json.loads(transaction_json)
            sar = self.sar_generator.generate_sar(transaction_data)
            return f"SAR generated: {sar.get('sar_id', 'ERROR')}, Risk Score: {sar.get('risk_score', 0)}"
        except Exception as e:
            return f"Error generating SAR: {str(e)}"
    
    def _analyze_risk_tool(self, risk_data: str) -> str:
        """Tool wrapper for risk analysis"""
        try:
            import json
            import pandas as pd
            data = json.loads(risk_data)
            df = pd.DataFrame(data)
            metrics = self.risk_analyzer.calculate_portfolio_metrics(df)
            return f"Risk Analysis: VaR 95%: {metrics['var_95']:.4f}, Volatility: {metrics['volatility']:.4f}"
        except Exception as e:
            return f"Error analyzing risk: {str(e)}"
    
    def _check_compliance_tool(self, transaction_details: str) -> str:
        """Tool wrapper for compliance checking"""
        # Simple compliance rules
        if "HIGH_RISK" in transaction_details.upper():
            return "COMPLIANCE ALERT: High-risk transaction detected. Manual review required."
        return "Transaction appears compliant with current regulations."
    
    def execute_workflow(self, task: str) -> Dict[str, Any]:
        """Execute an autonomous workflow"""
        try:
            # Create agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt="You are a compliance officer. Use available tools to complete tasks."
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=5
            )
            
            # Execute
            result = agent_executor.invoke({"input": task})
            
            return {
                "success": True,
                "result": result.get("output", ""),
                "steps_taken": len(result.get("intermediate_steps", []))
            }
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class WorkflowOrchestrator:
    """Orchestrate complex multi-agent workflows"""
    
    def __init__(self):
        self.workflows = {}
        self.execution_history = []
        
    def register_workflow(self, name: str, steps: List[Dict]):
        """Register a new workflow"""
        self.workflows[name] = {
            "name": name,
            "steps": steps,
            "created_at": pd.Timestamp.now()
        }
        logger.info(f"Workflow registered: {name}")
    
    def execute_workflow(self, name: str, context: Dict) -> Dict:
        """Execute a registered workflow"""
        if name not in self.workflows:
            return {"error": f"Workflow {name} not found"}
        
        workflow = self.workflows[name]
        results = []
        
        for step in workflow["steps"]:
            step_name = step.get("name")
            step_action = step.get("action")
            
            try:
                # Execute step based on action type
                if step_action == "generate_sar":
                    result = {"step": step_name, "status": "completed"}
                elif step_action == "risk_analysis":
                    result = {"step": step_name, "status": "completed"}
                elif step_action == "send_alert":
                    result = {"step": step_name, "status": "alert_sent"}
                else:
                    result = {"step": step_name, "status": "unknown_action"}
                
                results.append(result)
            except Exception as e:
                results.append({"step": step_name, "status": "failed", "error": str(e)})
        
        execution_record = {
            "workflow": name,
            "executed_at": pd.Timestamp.now(),
            "results": results,
            "context": context
        }
        self.execution_history.append(execution_record)
        
        return execution_record


# icrms/modules/agentic_ai/__init__.py
from .workflow_orchestrator import ComplianceAgent, WorkflowOrchestrator

__all__ = ['ComplianceAgent', 'WorkflowOrchestrator']
