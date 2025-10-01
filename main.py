# icrms/modules/compliance_automation/sar_generator.py
"""
Suspicious Activity Report (SAR) Generator
Uses LLaMA/GPT4All for automated compliance report generation
"""

from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SARGenerator:
    """Generate Suspicious Activity Reports using LLMs"""
    
    def __init__(self, model_path: str = "./icrms/data/models/gpt4all-lora-quantized.bin"):
        self.model_path = model_path
        self.llm = self._initialize_llm()
        self.sar_template = self._create_sar_template()
        
    def _initialize_llm(self):
        """Initialize GPT4All LLM"""
        try:
            llm = GPT4All(
                model=self.model_path,
                max_tokens=2048,
                temp=0.3,
                verbose=False
            )
            logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            raise
    
    def _create_sar_template(self) -> PromptTemplate:
        """Create SAR generation prompt template"""
        template = """
        You are a compliance officer generating a Suspicious Activity Report (SAR).
        
        Transaction Details:
        - Customer ID: {customer_id}
        - Transaction Amount: ${amount}
        - Transaction Type: {transaction_type}
        - Date: {date}
        - Location: {location}
        - Description: {description}
        
        Risk Indicators:
        {risk_indicators}
        
        Generate a professional SAR that includes:
        1. Executive Summary
        2. Suspicious Activity Description
        3. Risk Assessment
        4. Recommended Actions
        5. Regulatory References
        
        SAR Report:
        """
        return PromptTemplate(
            input_variables=["customer_id", "amount", "transaction_type", 
                           "date", "location", "description", "risk_indicators"],
            template=template
        )
    
    def generate_sar(self, transaction_data: Dict) -> Dict[str, any]:
        """Generate SAR from transaction data"""
        try:
            # Prepare risk indicators
            risk_indicators = self._analyze_risk_indicators(transaction_data)
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=self.sar_template)
            
            # Generate SAR
            sar_content = chain.run(
                customer_id=transaction_data.get("customer_id", "UNKNOWN"),
                amount=transaction_data.get("amount", 0),
                transaction_type=transaction_data.get("type", "UNKNOWN"),
                date=transaction_data.get("date", datetime.now().isoformat()),
                location=transaction_data.get("location", "UNKNOWN"),
                description=transaction_data.get("description", ""),
                risk_indicators="\n".join(risk_indicators)
            )
            
            return {
                "sar_id": f"SAR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "generated_date": datetime.now().isoformat(),
                "content": sar_content,
                "transaction_data": transaction_data,
                "risk_score": self._calculate_risk_score(transaction_data),
                "status": "PENDING_REVIEW"
            }
        except Exception as e:
            logger.error(f"SAR generation error: {e}")
            return {"error": str(e)}
    
    def _analyze_risk_indicators(self, transaction_data: Dict) -> List[str]:
        """Analyze and extract risk indicators"""
        indicators = []
        amount = transaction_data.get("amount", 0)
        
        if amount > 10000:
            indicators.append(f"Large transaction amount: ${amount}")
        
        if transaction_data.get("type") == "WIRE_TRANSFER":
            indicators.append("High-risk transaction type: Wire Transfer")
        
        if transaction_data.get("location", "").upper() in ["OFFSHORE", "HIGH_RISK"]:
            indicators.append("Transaction from high-risk jurisdiction")
        
        if transaction_data.get("frequency", 0) > 5:
            indicators.append("Unusual transaction frequency detected")
        
        return indicators if indicators else ["Standard transaction pattern"]
    
    def _calculate_risk_score(self, transaction_data: Dict) -> float:
        """Calculate risk score (0-100)"""
        score = 0.0
        
        # Amount-based scoring
        amount = transaction_data.get("amount", 0)
        if amount > 50000:
            score += 40
        elif amount > 10000:
            score += 25
        elif amount > 5000:
            score += 10
        
        # Type-based scoring
        if transaction_data.get("type") in ["WIRE_TRANSFER", "CRYPTO"]:
            score += 20
        
        # Location-based scoring
        if transaction_data.get("location", "").upper() in ["OFFSHORE", "HIGH_RISK"]:
            score += 30
        
        # Frequency-based scoring
        frequency = transaction_data.get("frequency", 0)
        if frequency > 10:
            score += 10
        
        return min(score, 100.0)


class ComplianceMonitor:
    """Monitor transactions and trigger SAR generation"""
    
    def __init__(self, sar_generator: SARGenerator):
        self.sar_generator = sar_generator
        self.threshold = 10000  # USD
        
    def monitor_transaction(self, transaction: Dict) -> Dict:
        """Monitor single transaction"""
        amount = transaction.get("amount", 0)
        
        if amount >= self.threshold:
            logger.info(f"Suspicious transaction detected: ${amount}")
            sar = self.sar_generator.generate_sar(transaction)
            return {
                "flagged": True,
                "sar_generated": True,
                "sar": sar
            }
        
        return {
            "flagged": False,
            "sar_generated": False
        }
    
    def batch_monitor(self, transactions: List[Dict]) -> List[Dict]:
        """Monitor multiple transactions"""
        results = []
        for transaction in transactions:
            result = self.monitor_transaction(transaction)
            results.append({
                "transaction_id": transaction.get("id"),
                "result": result
            })
        return results


# icrms/modules/compliance_automation/__init__.py
from .sar_generator import SARGenerator, ComplianceMonitor

__all__ = ['SARGenerator', 'ComplianceMonitor']
